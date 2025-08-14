"""
Request queue system for handling concurrent requests and preventing overload
"""
import asyncio
import time
import uuid
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import psutil
import multiprocessing

logger = logging.getLogger(__name__)

@dataclass
class QueuedRequest:
    """Represents a queued request"""
    id: str
    type: str
    priority: int
    payload: dict
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = 'queued'  # queued, processing, completed, failed
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'type': self.type,
            'priority': self.priority,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'retry_count': self.retry_count,
            'error': self.error
        }

class RequestQueue:
    """
    Manages request queuing with priority, concurrency limits, and resource monitoring
    """
    
    def __init__(self, 
                 max_concurrent_requests: int = 10,
                 max_queue_size: int = 100,
                 memory_threshold_percent: int = 80,
                 cpu_threshold_percent: int = 90):
        self.max_concurrent = max_concurrent_requests
        self.max_queue_size = max_queue_size
        self.memory_threshold = memory_threshold_percent
        self.cpu_threshold = cpu_threshold_percent
        
        # Request storage
        self.pending_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.active_requests: Dict[str, QueuedRequest] = {}
        self.completed_requests: Dict[str, QueuedRequest] = {}
        
        # Request handlers
        self.handlers: Dict[str, Callable[[dict], Awaitable[Any]]] = {}
        
        # Thread pool for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)
        
        # Queue processing task
        self.processing_task = None
        self.is_running = False
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'rejected_requests': 0,
            'average_processing_time': 0,
            'current_queue_size': 0
        }
        
    def register_handler(self, request_type: str, handler: Callable[[dict], Awaitable[Any]]):
        """Register a handler for a specific request type"""
        self.handlers[request_type] = handler
        
    async def enqueue(self, 
                     request_type: str,
                     payload: dict,
                     priority: int = 5) -> str:
        """
        Add a request to the queue
        
        Args:
            request_type: Type of request (must have registered handler)
            payload: Request payload
            priority: Priority (1-10, lower is higher priority)
            
        Returns:
            Request ID
            
        Raises:
            ValueError: If queue is full or request type not registered
        """
        if request_type not in self.handlers:
            raise ValueError(f"No handler registered for request type: {request_type}")
            
        # Check queue capacity
        if self.pending_queue.qsize() >= self.max_queue_size:
            self.metrics['rejected_requests'] += 1
            raise ValueError("Request queue is full")
            
        # Check system resources
        if not self._check_system_resources():
            self.metrics['rejected_requests'] += 1
            raise ValueError("System resources exhausted")
            
        # Create request
        request_id = str(uuid.uuid4())
        request = QueuedRequest(
            id=request_id,
            type=request_type,
            priority=priority,
            payload=payload,
            created_at=datetime.utcnow()
        )
        
        # Add to queue (priority, timestamp, request)
        await self.pending_queue.put((
            priority,
            request.created_at.timestamp(),
            request
        ))
        
        self.metrics['total_requests'] += 1
        self.metrics['current_queue_size'] = self.pending_queue.qsize()
        
        logger.info(f"Enqueued request {request_id} of type {request_type} with priority {priority}")
        
        return request_id
        
    async def get_status(self, request_id: str) -> Optional[dict]:
        """Get the status of a request"""
        # Check active requests
        if request_id in self.active_requests:
            return self.active_requests[request_id].to_dict()
            
        # Check completed requests
        if request_id in self.completed_requests:
            return self.completed_requests[request_id].to_dict()
            
        # Check pending queue
        for _, _, request in self.pending_queue._queue:
            if request.id == request_id:
                return request.to_dict()
                
        return None
        
    async def get_result(self, request_id: str, timeout: float = 300) -> Any:
        """
        Wait for a request to complete and return its result
        
        Args:
            request_id: Request ID
            timeout: Maximum time to wait in seconds
            
        Returns:
            Request result
            
        Raises:
            TimeoutError: If request doesn't complete within timeout
            Exception: If request failed
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request_id in self.completed_requests:
                request = self.completed_requests[request_id]
                if request.status == 'failed':
                    raise Exception(f"Request failed: {request.error}")
                return request.result
                
            await asyncio.sleep(0.5)
            
        raise TimeoutError(f"Request {request_id} did not complete within {timeout} seconds")
        
    async def start(self):
        """Start processing the queue"""
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_queue())
        logger.info("Request queue started")
        
    async def stop(self):
        """Stop processing the queue"""
        self.is_running = False
        if self.processing_task:
            await self.processing_task
        self.executor.shutdown(wait=True)
        logger.info("Request queue stopped")
        
    async def _process_queue(self):
        """Main queue processing loop"""
        while self.is_running:
            try:
                # Check if we can process more requests
                if len(self.active_requests) >= self.max_concurrent:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Check system resources
                if not self._check_system_resources():
                    logger.warning("System resources low, pausing queue processing")
                    await asyncio.sleep(5)
                    continue
                    
                # Get next request from queue
                try:
                    priority, timestamp, request = await asyncio.wait_for(
                        self.pending_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                    
                # Process the request
                self.active_requests[request.id] = request
                asyncio.create_task(self._process_request(request))
                
            except Exception as e:
                logger.error(f"Error in queue processing loop: {e}")
                await asyncio.sleep(1)
                
    async def _process_request(self, request: QueuedRequest):
        """Process a single request"""
        try:
            request.status = 'processing'
            request.started_at = datetime.utcnow()
            
            logger.info(f"Processing request {request.id} of type {request.type}")
            
            # Get handler
            handler = self.handlers[request.type]
            
            # Execute handler with timeout
            result = await asyncio.wait_for(
                handler(request.payload),
                timeout=300  # 5 minute timeout
            )
            
            # Update request
            request.status = 'completed'
            request.completed_at = datetime.utcnow()
            request.result = result
            
            # Update metrics
            processing_time = (request.completed_at - request.started_at).total_seconds()
            self._update_average_processing_time(processing_time)
            self.metrics['completed_requests'] += 1
            
            logger.info(f"Completed request {request.id} in {processing_time:.2f} seconds")
            
        except asyncio.TimeoutError:
            await self._handle_request_failure(request, "Request timed out")
            
        except Exception as e:
            await self._handle_request_failure(request, str(e))
            
        finally:
            # Move to completed and remove from active
            self.completed_requests[request.id] = request
            del self.active_requests[request.id]
            self.metrics['current_queue_size'] = self.pending_queue.qsize()
            
    async def _handle_request_failure(self, request: QueuedRequest, error: str):
        """Handle failed request with retry logic"""
        request.error = error
        request.retry_count += 1
        
        if request.retry_count < request.max_retries:
            # Re-queue with lower priority
            logger.warning(f"Request {request.id} failed, retrying ({request.retry_count}/{request.max_retries}): {error}")
            request.status = 'queued'
            await self.pending_queue.put((
                request.priority + 1,  # Lower priority for retries
                datetime.utcnow().timestamp(),
                request
            ))
        else:
            # Final failure
            request.status = 'failed'
            request.completed_at = datetime.utcnow()
            self.metrics['failed_requests'] += 1
            logger.error(f"Request {request.id} failed after {request.retry_count} retries: {error}")
            
    def _check_system_resources(self) -> bool:
        """Check if system has enough resources"""
        try:
            # Check memory
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold:
                logger.warning(f"Memory usage high: {memory.percent}%")
                return False
                
            # Check CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.cpu_threshold:
                logger.warning(f"CPU usage high: {cpu_percent}%")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            return True  # Continue processing on error
            
    def _update_average_processing_time(self, new_time: float):
        """Update average processing time metric"""
        completed = self.metrics['completed_requests']
        if completed == 0:
            self.metrics['average_processing_time'] = new_time
        else:
            current_avg = self.metrics['average_processing_time']
            self.metrics['average_processing_time'] = (
                (current_avg * (completed - 1) + new_time) / completed
            )
            
    def get_metrics(self) -> dict:
        """Get queue metrics"""
        return {
            **self.metrics,
            'active_requests': len(self.active_requests),
            'queue_capacity': f"{self.pending_queue.qsize()}/{self.max_queue_size}"
        }
        
    async def clear_completed(self, older_than_minutes: int = 60):
        """Clear completed requests older than specified minutes"""
        cutoff_time = datetime.utcnow().timestamp() - (older_than_minutes * 60)
        
        to_remove = []
        for request_id, request in self.completed_requests.items():
            if request.completed_at and request.completed_at.timestamp() < cutoff_time:
                to_remove.append(request_id)
                
        for request_id in to_remove:
            del self.completed_requests[request_id]
            
        logger.info(f"Cleared {len(to_remove)} completed requests older than {older_than_minutes} minutes")