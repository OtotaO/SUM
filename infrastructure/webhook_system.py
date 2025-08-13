"""
Webhook System for SUM Platform
================================

Production-grade webhook infrastructure with:
- HMAC signature verification for security
- Exponential backoff retry logic
- Circuit breaker pattern for resilience
- Event-driven architecture
- Comprehensive logging and monitoring

This enables third-party integrations and real-time notifications
while maintaining security and reliability.

Author: ototao
License: Apache License 2.0
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set
from urllib.parse import urlparse
import aiohttp
from aiohttp import ClientTimeout, ClientError
import uuid

logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Enumeration of all webhook events in the SUM platform"""
    
    # Document events
    DOCUMENT_SUMMARIZED = "document.summarized"
    DOCUMENT_ANALYZED = "document.analyzed"
    DOCUMENT_PROCESSED = "document.processed"
    
    # Knowledge events
    THOUGHT_CAPTURED = "thought.captured"
    INSIGHT_GENERATED = "insight.generated"
    KNOWLEDGE_DENSIFIED = "knowledge.densified"
    BREAKTHROUGH_DETECTED = "breakthrough.detected"
    
    # Memory events
    MEMORY_STORED = "memory.stored"
    PATTERN_DETECTED = "pattern.detected"
    MEMORY_COMPRESSED = "memory.compressed"
    
    # Temporal events
    TEMPORAL_PATTERN_FOUND = "temporal.pattern_found"
    CONCEPT_EVOLVED = "concept.evolved"
    
    # Predictive events
    NEED_PREDICTED = "need.predicted"
    SUGGESTION_GENERATED = "suggestion.generated"
    
    # System events
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"


@dataclass
class WebhookConfig:
    """Configuration for a webhook endpoint"""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    url: str = ""
    events: List[WebhookEvent] = field(default_factory=list)
    secret: str = ""
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_base: float = 1.0  # Base delay in seconds
    retry_delay_max: float = 60.0  # Maximum delay in seconds
    
    # Circuit breaker configuration
    circuit_breaker_threshold: int = 5  # Failures before opening circuit
    circuit_breaker_timeout: float = 300.0  # Seconds before trying again
    circuit_breaker_state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    circuit_breaker_last_failure: Optional[datetime] = None
    
    # Request configuration
    timeout_seconds: float = 10.0
    
    # Metadata
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    custom_headers: Dict[str, str] = field(default_factory=dict)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for webhook resilience.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests blocked
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 success_threshold: int = 2):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            success_threshold: Successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = datetime.now()
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            # Check if we should try recovery
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self._transition_to("HALF_OPEN")
                    return True
            return False
        
        if self.state == "HALF_OPEN":
            return True
        
        return False
    
    def record_success(self):
        """Record successful execution"""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._transition_to("CLOSED")
        elif self.state == "CLOSED":
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed execution"""
        self.last_failure_time = datetime.now()
        
        if self.state == "CLOSED":
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self._transition_to("OPEN")
        elif self.state == "HALF_OPEN":
            self._transition_to("OPEN")
    
    def _transition_to(self, new_state: str):
        """Transition to new state"""
        logger.info(f"Circuit breaker transitioning from {self.state} to {new_state}")
        self.state = new_state
        self.last_state_change = datetime.now()
        
        if new_state == "CLOSED":
            self.failure_count = 0
            self.success_count = 0
        elif new_state == "HALF_OPEN":
            self.success_count = 0


class WebhookManager:
    """
    Central webhook management system with security and resilience.
    
    Features:
    - HMAC signature verification
    - Exponential backoff retries
    - Circuit breaker pattern
    - Event filtering and routing
    - Comprehensive logging
    - Performance metrics
    """
    
    def __init__(self, 
                 max_webhooks_per_event: int = 10,
                 global_timeout: float = 30.0,
                 enable_async: bool = True):
        """
        Initialize webhook manager.
        
        Args:
            max_webhooks_per_event: Maximum webhooks per event type
            global_timeout: Global timeout for all webhook calls
            enable_async: Enable asynchronous webhook delivery
        """
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.event_webhooks: Dict[WebhookEvent, List[str]] = defaultdict(list)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        self.max_webhooks_per_event = max_webhooks_per_event
        self.global_timeout = global_timeout
        self.enable_async = enable_async
        
        # Metrics
        self.metrics = {
            'total_webhooks_sent': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'retries_attempted': 0,
            'circuit_breaks': 0,
            'avg_response_time': 0.0,
            'response_times': deque(maxlen=1000)  # Keep last 1000 response times
        }
        
        # Event queue for async processing
        self.event_queue: asyncio.Queue = None
        self.processing_task = None
        
        if self.enable_async:
            self._start_async_processor()
    
    def register_webhook(self, 
                         url: str,
                         events: List[WebhookEvent],
                         secret: str,
                         description: str = "",
                         **kwargs) -> str:
        """
        Register a new webhook endpoint.
        
        Args:
            url: Webhook endpoint URL
            events: List of events to subscribe to
            secret: Secret key for HMAC verification
            description: Optional description
            **kwargs: Additional configuration options
        
        Returns:
            Webhook ID
        
        Raises:
            ValueError: If URL is invalid or too many webhooks registered
        """
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid webhook URL: {url}")
        
        if parsed.scheme not in ['http', 'https']:
            raise ValueError(f"URL scheme must be http or https: {url}")
        
        # Check limits
        for event in events:
            if len(self.event_webhooks[event]) >= self.max_webhooks_per_event:
                raise ValueError(
                    f"Maximum webhooks ({self.max_webhooks_per_event}) "
                    f"already registered for event {event.value}"
                )
        
        # Create webhook configuration
        config = WebhookConfig(
            url=url,
            events=events,
            secret=secret,
            description=description,
            **kwargs
        )
        
        # Register webhook
        self.webhooks[config.id] = config
        
        # Map events to webhook
        for event in events:
            self.event_webhooks[event].append(config.id)
        
        # Create circuit breaker
        self.circuit_breakers[config.id] = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            recovery_timeout=config.circuit_breaker_timeout
        )
        
        logger.info(f"Registered webhook {config.id} for events {[e.value for e in events]}")
        
        return config.id
    
    def unregister_webhook(self, webhook_id: str) -> bool:
        """
        Unregister a webhook.
        
        Args:
            webhook_id: ID of webhook to unregister
        
        Returns:
            True if successfully unregistered
        """
        if webhook_id not in self.webhooks:
            return False
        
        config = self.webhooks[webhook_id]
        
        # Remove from event mappings
        for event in config.events:
            if webhook_id in self.event_webhooks[event]:
                self.event_webhooks[event].remove(webhook_id)
        
        # Remove webhook and circuit breaker
        del self.webhooks[webhook_id]
        del self.circuit_breakers[webhook_id]
        
        logger.info(f"Unregistered webhook {webhook_id}")
        
        return True
    
    async def trigger_event(self,
                           event: WebhookEvent,
                           payload: Dict[str, Any],
                           metadata: Optional[Dict[str, Any]] = None):
        """
        Trigger webhooks for an event.
        
        Args:
            event: Event type
            payload: Event payload
            metadata: Optional metadata
        """
        # Add event metadata
        event_data = {
            'event': event.value,
            'timestamp': datetime.now().isoformat(),
            'payload': payload
        }
        
        if metadata:
            event_data['metadata'] = metadata
        
        # Get registered webhooks for this event
        webhook_ids = self.event_webhooks.get(event, [])
        
        if not webhook_ids:
            logger.debug(f"No webhooks registered for event {event.value}")
            return
        
        logger.info(f"Triggering {len(webhook_ids)} webhooks for event {event.value}")
        
        # Process webhooks
        if self.enable_async:
            # Queue for async processing
            await self.event_queue.put((event, event_data, webhook_ids))
        else:
            # Process synchronously
            await self._process_webhooks(webhook_ids, event_data)
    
    async def _process_webhooks(self,
                               webhook_ids: List[str],
                               event_data: Dict[str, Any]):
        """Process webhooks for an event"""
        tasks = []
        
        for webhook_id in webhook_ids:
            if webhook_id not in self.webhooks:
                continue
            
            config = self.webhooks[webhook_id]
            
            if not config.active:
                logger.debug(f"Skipping inactive webhook {webhook_id}")
                continue
            
            # Check circuit breaker
            circuit_breaker = self.circuit_breakers[webhook_id]
            if not circuit_breaker.can_execute():
                logger.warning(f"Circuit breaker OPEN for webhook {webhook_id}")
                self.metrics['circuit_breaks'] += 1
                continue
            
            # Create task for webhook delivery
            task = self._deliver_webhook(config, event_data, circuit_breaker)
            tasks.append(task)
        
        # Execute all webhook deliveries
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _deliver_webhook(self,
                              config: WebhookConfig,
                              event_data: Dict[str, Any],
                              circuit_breaker: CircuitBreaker):
        """
        Deliver webhook with retries and error handling.
        
        Args:
            config: Webhook configuration
            event_data: Event data to send
            circuit_breaker: Circuit breaker for this webhook
        """
        self.metrics['total_webhooks_sent'] += 1
        
        # Generate signature
        signature = self._generate_signature(config.secret, event_data)
        
        # Prepare headers
        headers = {
            'Content-Type': 'application/json',
            'X-SUM-Signature': signature,
            'X-SUM-Event': event_data['event'],
            'X-SUM-Webhook-ID': config.id,
            'X-SUM-Timestamp': event_data['timestamp']
        }
        
        # Add custom headers
        headers.update(config.custom_headers)
        
        # Prepare request
        timeout = ClientTimeout(total=config.timeout_seconds)
        
        # Attempt delivery with retries
        retry_count = 0
        last_error = None
        
        while retry_count <= config.max_retries:
            try:
                start_time = time.time()
                
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        config.url,
                        json=event_data,
                        headers=headers
                    ) as response:
                        response_time = time.time() - start_time
                        self.metrics['response_times'].append(response_time)
                        
                        if response.status >= 200 and response.status < 300:
                            # Success
                            config.success_count += 1
                            config.last_triggered = datetime.now()
                            circuit_breaker.record_success()
                            self.metrics['successful_deliveries'] += 1
                            
                            logger.info(
                                f"Successfully delivered webhook {config.id} "
                                f"(status: {response.status}, time: {response_time:.2f}s)"
                            )
                            return
                        else:
                            # HTTP error
                            error_body = await response.text()
                            last_error = f"HTTP {response.status}: {error_body[:200]}"
                            logger.warning(
                                f"Webhook {config.id} returned error: {last_error}"
                            )
            
            except ClientError as e:
                last_error = str(e)
                logger.warning(f"Webhook {config.id} connection error: {last_error}")
            
            except Exception as e:
                last_error = str(e)
                logger.error(f"Unexpected error delivering webhook {config.id}: {last_error}")
            
            # Retry logic
            if retry_count < config.max_retries:
                retry_delay = min(
                    config.retry_delay_base * (2 ** retry_count),
                    config.retry_delay_max
                )
                logger.info(f"Retrying webhook {config.id} in {retry_delay}s (attempt {retry_count + 1})")
                await asyncio.sleep(retry_delay)
                retry_count += 1
                self.metrics['retries_attempted'] += 1
            else:
                break
        
        # All retries failed
        config.failure_count += 1
        circuit_breaker.record_failure()
        self.metrics['failed_deliveries'] += 1
        
        logger.error(
            f"Failed to deliver webhook {config.id} after {retry_count} retries: {last_error}"
        )
    
    def _generate_signature(self, secret: str, data: Dict[str, Any]) -> str:
        """
        Generate HMAC signature for webhook security.
        
        Args:
            secret: Secret key
            data: Data to sign
        
        Returns:
            Hex-encoded HMAC-SHA256 signature
        """
        message = json.dumps(data, sort_keys=True, separators=(',', ':'))
        signature = hmac.new(
            secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
    
    def verify_signature(self, 
                         secret: str,
                         signature: str,
                         data: Dict[str, Any]) -> bool:
        """
        Verify webhook signature (for receiving webhooks).
        
        Args:
            secret: Secret key
            signature: Received signature
            data: Received data
        
        Returns:
            True if signature is valid
        """
        expected = self._generate_signature(secret, data)
        return hmac.compare_digest(expected, signature)
    
    def _start_async_processor(self):
        """Start asynchronous webhook processor"""
        self.event_queue = asyncio.Queue()
        self.processing_task = asyncio.create_task(self._async_processor())
    
    async def _async_processor(self):
        """Process webhook events asynchronously"""
        while True:
            try:
                # Get event from queue
                event, event_data, webhook_ids = await self.event_queue.get()
                
                # Process webhooks
                await self._process_webhooks(webhook_ids, event_data)
                
            except Exception as e:
                logger.error(f"Error in async webhook processor: {e}")
    
    def get_webhook_status(self, webhook_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a webhook.
        
        Args:
            webhook_id: Webhook ID
        
        Returns:
            Status dictionary or None if not found
        """
        if webhook_id not in self.webhooks:
            return None
        
        config = self.webhooks[webhook_id]
        circuit_breaker = self.circuit_breakers[webhook_id]
        
        return {
            'id': config.id,
            'url': config.url,
            'active': config.active,
            'events': [e.value for e in config.events],
            'created_at': config.created_at.isoformat(),
            'last_triggered': config.last_triggered.isoformat() if config.last_triggered else None,
            'success_count': config.success_count,
            'failure_count': config.failure_count,
            'circuit_breaker_state': circuit_breaker.state,
            'description': config.description
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get webhook system metrics"""
        avg_response_time = (
            sum(self.metrics['response_times']) / len(self.metrics['response_times'])
            if self.metrics['response_times'] else 0.0
        )
        
        return {
            'total_webhooks_registered': len(self.webhooks),
            'active_webhooks': sum(1 for w in self.webhooks.values() if w.active),
            'total_webhooks_sent': self.metrics['total_webhooks_sent'],
            'successful_deliveries': self.metrics['successful_deliveries'],
            'failed_deliveries': self.metrics['failed_deliveries'],
            'success_rate': (
                self.metrics['successful_deliveries'] / self.metrics['total_webhooks_sent']
                if self.metrics['total_webhooks_sent'] > 0 else 0.0
            ),
            'retries_attempted': self.metrics['retries_attempted'],
            'circuit_breaks': self.metrics['circuit_breaks'],
            'avg_response_time': avg_response_time,
            'webhooks_by_event': {
                event.value: len(webhook_ids)
                for event, webhook_ids in self.event_webhooks.items()
            }
        }
    
    def list_webhooks(self, 
                      event: Optional[WebhookEvent] = None,
                      active_only: bool = False) -> List[Dict[str, Any]]:
        """
        List registered webhooks.
        
        Args:
            event: Filter by event type
            active_only: Only show active webhooks
        
        Returns:
            List of webhook status dictionaries
        """
        webhooks = []
        
        for webhook_id, config in self.webhooks.items():
            if active_only and not config.active:
                continue
            
            if event and event not in config.events:
                continue
            
            webhooks.append(self.get_webhook_status(webhook_id))
        
        return webhooks
    
    async def shutdown(self):
        """Gracefully shutdown webhook manager"""
        if self.processing_task:
            self.processing_task.cancel()
            
        # Wait for pending webhooks to complete
        if self.event_queue:
            while not self.event_queue.empty():
                await asyncio.sleep(0.1)
        
        logger.info("Webhook manager shutdown complete")


# Global webhook manager instance
webhook_manager = WebhookManager()


# Convenience function for triggering events
async def trigger_webhook_event(event: WebhookEvent, 
                                payload: Dict[str, Any],
                                **metadata):
    """
    Trigger a webhook event.
    
    Args:
        event: Event type
        payload: Event payload
        **metadata: Additional metadata
    """
    await webhook_manager.trigger_event(event, payload, metadata)


# Example usage and integration points
if __name__ == "__main__":
    # Example: Register a webhook
    webhook_id = webhook_manager.register_webhook(
        url="https://example.com/webhooks/sum",
        events=[WebhookEvent.DOCUMENT_SUMMARIZED, WebhookEvent.INSIGHT_GENERATED],
        secret="your-secret-key",
        description="Example webhook for testing"
    )
    
    print(f"Registered webhook: {webhook_id}")
    
    # Example: Trigger an event
    async def example_trigger():
        await trigger_webhook_event(
            WebhookEvent.DOCUMENT_SUMMARIZED,
            {
                'document_id': '123',
                'summary': 'This is a summary',
                'keywords': ['ai', 'machine learning'],
                'processing_time': 0.5
            },
            user_id='user123',
            source='api'
        )
    
    # Run example
    asyncio.run(example_trigger())
    
    # Get metrics
    print("Webhook Metrics:", webhook_manager.get_metrics())
