#!/usr/bin/env python3
"""
multimodal_integration.py - Integration layer for Multi-Modal Processing in SUM

This module provides seamless integration between:
- Multi-modal processing engine
- Predictive intelligence system
- Zero-friction capture
- Knowledge graph generation
- Real-time processing pipeline

Features:
- Automatic content routing
- Progressive enhancement
- Cross-modal correlation
- Intelligent caching
- Batch processing with progress
- Real-time streaming support

Author: ototao
License: Apache License 2.0
"""

import os
import json
import time
import logging
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, PriorityQueue
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict
import tempfile

# Import SUM components
from multimodal_engine import MultiModalEngine, EnhancedProcessingResult, ExtendedContentType
from predictive_intelligence import PredictiveIntelligenceSystem
from knowledge_graph_visualizer import KnowledgeGraphVisualizer
from capture.capture_engine import CaptureEngine
from sum_engines import HierarchicalDensificationEngine

# Import additional dependencies
try:
    import aiofiles
    ASYNC_FILES = True
except ImportError:
    ASYNC_FILES = False
    logging.warning("Async file support not available. Install: pip install aiofiles")

try:
    import websockets
    WEBSOCKET_SUPPORT = True
except ImportError:
    WEBSOCKET_SUPPORT = False
    logging.warning("WebSocket support not available. Install: pip install websockets")

try:
    from tqdm.asyncio import tqdm as async_tqdm
    from tqdm import tqdm
    PROGRESS_SUPPORT = True
except ImportError:
    PROGRESS_SUPPORT = False
    logging.warning("Progress bars not available. Install: pip install tqdm")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingRequest:
    """Request for multi-modal processing."""
    request_id: str
    file_path: str
    priority: int = 5  # 1-10, higher is more urgent
    timestamp: datetime = field(default_factory=datetime.now)
    options: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Any] = None
    
    def __lt__(self, other):
        """For priority queue sorting."""
        return self.priority > other.priority


@dataclass
class ProcessingProgress:
    """Progress tracking for processing operations."""
    request_id: str
    status: str  # queued, processing, completed, error
    progress: float  # 0.0 to 1.0
    current_step: str
    eta_seconds: Optional[float] = None
    result: Optional[EnhancedProcessingResult] = None
    error: Optional[str] = None


@dataclass
class CrossModalInsight:
    """Insight derived from multiple content sources."""
    insight_id: str
    content_sources: List[str]  # File paths
    insight_type: str  # connection, pattern, summary, etc.
    description: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiModalIntegration:
    """
    Integration layer for seamless multi-modal processing in SUM.
    
    Coordinates between all components for intelligent content processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the integration layer."""
        self.config = config or {}
        
        # Core components
        self.multimodal_engine = MultiModalEngine(config.get('multimodal', {}))
        self.predictive_system = PredictiveIntelligenceSystem(config.get('predictive', {}))
        self.hierarchical_engine = HierarchicalDensificationEngine()
        
        # Processing infrastructure
        self.processing_queue = PriorityQueue()
        self.progress_tracking = {}
        self.processing_history = defaultdict(list)
        
        # Cross-modal correlation
        self.content_graph = defaultdict(list)
        self.insights_cache = {}
        
        # Background processing
        self.executor = asyncio.new_event_loop()
        self.processing_thread = threading.Thread(target=self._run_processing_loop, daemon=True)
        self.processing_thread.start()
        
        # Real-time support
        self.subscribers = defaultdict(list)
        self.websocket_server = None
        
        logger.info("MultiModalIntegration initialized successfully")
    
    def _run_processing_loop(self):
        """Run the async processing loop in background thread."""
        asyncio.set_event_loop(self.executor)
        self.executor.run_until_complete(self._processing_loop())
    
    async def _processing_loop(self):
        """Main processing loop for handling requests."""
        while True:
            try:
                # Check for new requests
                if not self.processing_queue.empty():
                    request = self.processing_queue.get()
                    await self._process_request(request)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    async def _process_request(self, request: ProcessingRequest):
        """Process a single request with progress tracking."""
        try:
            # Update progress
            progress = ProcessingProgress(
                request_id=request.request_id,
                status='processing',
                progress=0.0,
                current_step='Initializing'
            )
            self._update_progress(progress)
            
            # Detect content type
            content_type = self.multimodal_engine.detect_content_type(request.file_path)
            progress.current_step = f'Processing {content_type.value} content'
            progress.progress = 0.2
            self._update_progress(progress)
            
            # Process with multi-modal engine
            result = await self._process_with_engine(request, progress)
            
            # Enhance with predictive intelligence
            progress.current_step = 'Applying predictive intelligence'
            progress.progress = 0.6
            self._update_progress(progress)
            
            enhanced_result = await self._enhance_with_predictions(result, request)
            
            # Cross-modal correlation
            progress.current_step = 'Finding cross-modal connections'
            progress.progress = 0.8
            self._update_progress(progress)
            
            insights = await self._find_cross_modal_insights(enhanced_result, request)
            
            # Complete processing
            progress.status = 'completed'
            progress.progress = 1.0
            progress.result = enhanced_result
            progress.current_step = 'Processing complete'
            self._update_progress(progress)
            
            # Store in history
            self.processing_history[request.file_path].append({
                'timestamp': datetime.now(),
                'result': enhanced_result,
                'insights': insights
            })
            
            # Trigger callback if provided
            if request.callback:
                await self._trigger_callback(request.callback, enhanced_result, insights)
            
            # Notify subscribers
            await self._notify_subscribers(request.request_id, progress)
            
        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {e}")
            progress = ProcessingProgress(
                request_id=request.request_id,
                status='error',
                progress=0.0,
                current_step='Error occurred',
                error=str(e)
            )
            self._update_progress(progress)
    
    async def _process_with_engine(self, request: ProcessingRequest, progress: ProcessingProgress) -> EnhancedProcessingResult:
        """Process file with multi-modal engine."""
        # Use async processing if available
        if hasattr(self.multimodal_engine, 'process_file_async'):
            result = await self.multimodal_engine.process_file_async(
                request.file_path,
                **request.options
            )
        else:
            # Fall back to sync processing in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.multimodal_engine.process_file,
                request.file_path,
                **request.options
            )
        
        return result
    
    async def _enhance_with_predictions(self, result: EnhancedProcessingResult, request: ProcessingRequest) -> EnhancedProcessingResult:
        """Enhance result with predictive intelligence."""
        try:
            # Get user profile
            user_id = request.options.get('user_id', 'default')
            
            # Create thought from result
            thought_content = result.extracted_text[:1000] if result.extracted_text else ""
            
            # Process with predictive system
            prediction_result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.predictive_system.process_thought,
                thought_content,
                {
                    'source': request.file_path,
                    'content_type': result.content_type.value,
                    'metadata': result.metadata
                }
            )
            
            # Add predictions to result
            if prediction_result:
                result.metadata['predictions'] = {
                    'suggested_connections': prediction_result.get('connections', []),
                    'relevant_insights': prediction_result.get('insights', []),
                    'next_steps': prediction_result.get('next_steps', [])
                }
                
                # Extract entities and concepts
                result.entities = prediction_result.get('entities', [])
                result.concepts = prediction_result.get('concepts', [])
        
        except Exception as e:
            logger.warning(f"Failed to enhance with predictions: {e}")
        
        return result
    
    async def _find_cross_modal_insights(self, result: EnhancedProcessingResult, request: ProcessingRequest) -> List[CrossModalInsight]:
        """Find insights across different content types."""
        insights = []
        
        try:
            # Get recent processing history
            recent_results = []
            for file_path, history in self.processing_history.items():
                if file_path != request.file_path:
                    for item in history[-5:]:  # Last 5 items
                        if (datetime.now() - item['timestamp']).days < 7:  # Within a week
                            recent_results.append((file_path, item['result']))
            
            # Find connections
            for other_path, other_result in recent_results:
                # Compare concepts
                if result.concepts and other_result.concepts:
                    common_concepts = set(result.concepts) & set(other_result.concepts)
                    if common_concepts:
                        insight = CrossModalInsight(
                            insight_id=hashlib.md5(f"{request.file_path}:{other_path}".encode()).hexdigest()[:8],
                            content_sources=[request.file_path, other_path],
                            insight_type='concept_connection',
                            description=f"Common concepts found: {', '.join(common_concepts)}",
                            confidence=len(common_concepts) / max(len(result.concepts), len(other_result.concepts)),
                            metadata={
                                'concepts': list(common_concepts),
                                'content_types': [result.content_type.value, other_result.content_type.value]
                            }
                        )
                        insights.append(insight)
                
                # Cross-modal patterns
                if result.content_type.value == 'audio' and other_result.content_type.value == 'image':
                    # Example: Meeting recording + whiteboard photo
                    if 'meeting' in result.metadata.get('content_type', '') and other_result.visual_elements:
                        insight = CrossModalInsight(
                            insight_id=hashlib.md5(f"meeting_visual:{request.file_path}:{other_path}".encode()).hexdigest()[:8],
                            content_sources=[request.file_path, other_path],
                            insight_type='meeting_visualization',
                            description="Meeting discussion matches whiteboard content",
                            confidence=0.7,
                            metadata={
                                'meeting_topics': result.metadata.get('key_topics', []),
                                'visual_elements': other_result.visual_elements
                            }
                        )
                        insights.append(insight)
            
            # Cache insights
            self.insights_cache[request.file_path] = insights
            
        except Exception as e:
            logger.warning(f"Failed to find cross-modal insights: {e}")
        
        return insights
    
    def _update_progress(self, progress: ProcessingProgress):
        """Update progress tracking."""
        self.progress_tracking[progress.request_id] = progress
        
        # Log significant updates
        if progress.status in ['completed', 'error']:
            logger.info(f"Request {progress.request_id}: {progress.status}")
    
    async def _trigger_callback(self, callback, result: EnhancedProcessingResult, insights: List[CrossModalInsight]):
        """Trigger callback with results."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(result, insights)
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    callback,
                    result,
                    insights
                )
        except Exception as e:
            logger.error(f"Error in callback: {e}")
    
    async def _notify_subscribers(self, request_id: str, progress: ProcessingProgress):
        """Notify subscribers of progress updates."""
        subscribers = self.subscribers.get(request_id, [])
        for subscriber in subscribers:
            try:
                await subscriber(progress)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    # Public API methods
    
    def process_file(self, file_path: str, priority: int = 5, **options) -> str:
        """
        Queue a file for processing.
        
        Returns request_id for tracking.
        """
        request_id = hashlib.md5(f"{file_path}:{time.time()}".encode()).hexdigest()[:16]
        
        request = ProcessingRequest(
            request_id=request_id,
            file_path=file_path,
            priority=priority,
            options=options
        )
        
        # Initial progress
        progress = ProcessingProgress(
            request_id=request_id,
            status='queued',
            progress=0.0,
            current_step='Queued for processing'
        )
        self._update_progress(progress)
        
        # Queue request
        self.processing_queue.put(request)
        
        logger.info(f"Queued {file_path} for processing with ID {request_id}")
        
        return request_id
    
    def process_batch(self, file_paths: List[str], priority: int = 5, **options) -> List[str]:
        """Queue multiple files for processing."""
        request_ids = []
        
        for file_path in file_paths:
            request_id = self.process_file(file_path, priority, **options)
            request_ids.append(request_id)
        
        return request_ids
    
    def get_progress(self, request_id: str) -> Optional[ProcessingProgress]:
        """Get progress for a specific request."""
        return self.progress_tracking.get(request_id)
    
    def get_result(self, request_id: str) -> Optional[EnhancedProcessingResult]:
        """Get result if processing is complete."""
        progress = self.get_progress(request_id)
        if progress and progress.status == 'completed':
            return progress.result
        return None
    
    async def process_stream(self, file_path: str, chunk_size: int = 1024*1024) -> AsyncIterator[ProcessingProgress]:
        """Process file with streaming updates."""
        request_id = self.process_file(file_path, priority=8)  # Higher priority for streaming
        
        # Subscribe to updates
        queue = asyncio.Queue()
        
        async def update_handler(progress):
            await queue.put(progress)
        
        self.subscribers[request_id].append(update_handler)
        
        try:
            # Yield progress updates
            while True:
                progress = await queue.get()
                yield progress
                
                if progress.status in ['completed', 'error']:
                    break
        finally:
            # Cleanup subscription
            self.subscribers[request_id].remove(update_handler)
    
    def get_cross_modal_insights(self, file_path: str) -> List[CrossModalInsight]:
        """Get cached cross-modal insights for a file."""
        return self.insights_cache.get(file_path, [])
    
    def get_processing_history(self, file_path: str) -> List[Dict[str, Any]]:
        """Get processing history for a file."""
        return self.processing_history.get(file_path, [])
    
    async def start_websocket_server(self, host: str = 'localhost', port: int = 8765):
        """Start WebSocket server for real-time updates."""
        if not WEBSOCKET_SUPPORT:
            logger.error("WebSocket support not available")
            return
        
        async def handler(websocket, path):
            # Handle WebSocket connections
            request_id = None
            try:
                async for message in websocket:
                    data = json.loads(message)
                    
                    if data['type'] == 'subscribe':
                        request_id = data['request_id']
                        
                        async def ws_update(progress):
                            await websocket.send(json.dumps({
                                'type': 'progress',
                                'data': {
                                    'request_id': progress.request_id,
                                    'status': progress.status,
                                    'progress': progress.progress,
                                    'current_step': progress.current_step
                                }
                            }))
                        
                        self.subscribers[request_id].append(ws_update)
                    
                    elif data['type'] == 'process':
                        # Process file via WebSocket
                        file_path = data['file_path']
                        request_id = self.process_file(file_path, **data.get('options', {}))
                        
                        await websocket.send(json.dumps({
                            'type': 'queued',
                            'request_id': request_id
                        }))
            
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                if request_id and request_id in self.subscribers:
                    # Clean up subscription
                    self.subscribers[request_id] = [
                        s for s in self.subscribers[request_id] 
                        if not asyncio.iscoroutinefunction(s) or s.__name__ != 'ws_update'
                    ]
        
        self.websocket_server = await websockets.serve(handler, host, port)
        logger.info(f"WebSocket server started on ws://{host}:{port}")
    
    def generate_processing_report(self) -> Dict[str, Any]:
        """Generate comprehensive processing report."""
        report = {
            'total_processed': sum(len(h) for h in self.processing_history.values()),
            'active_requests': self.processing_queue.qsize(),
            'content_types': defaultdict(int),
            'average_processing_times': defaultdict(list),
            'cross_modal_insights': len(self.insights_cache),
            'recent_activity': []
        }
        
        # Analyze history
        for file_path, history in self.processing_history.items():
            for item in history:
                result = item['result']
                content_type = result.content_type.value
                report['content_types'][content_type] += 1
                report['average_processing_times'][content_type].append(result.processing_time)
        
        # Calculate averages
        for content_type, times in report['average_processing_times'].items():
            report['average_processing_times'][content_type] = sum(times) / len(times) if times else 0
        
        # Recent activity
        all_items = []
        for file_path, history in self.processing_history.items():
            for item in history:
                all_items.append({
                    'file': file_path,
                    'timestamp': item['timestamp'],
                    'content_type': item['result'].content_type.value
                })
        
        all_items.sort(key=lambda x: x['timestamp'], reverse=True)
        report['recent_activity'] = all_items[:10]
        
        return report


# Example usage and demo
if __name__ == "__main__":
    import sys
    
    # Initialize integration
    integration = MultiModalIntegration({
        'multimodal': {
            'whisper_model': 'base',
            'cache_dir': './cache/integration'
        },
        'predictive': {
            'model_name': 'all-MiniLM-L6-v2'
        }
    })
    
    # Demo async processing
    async def demo_async():
        """Demo async processing capabilities."""
        print("Multi-Modal Integration Demo")
        print("=" * 50)
        
        if len(sys.argv) > 1:
            file_paths = sys.argv[1:]
            
            # Process files
            print(f"\nProcessing {len(file_paths)} files...")
            request_ids = integration.process_batch(file_paths, priority=7)
            
            # Track progress
            for request_id in request_ids:
                print(f"\nTracking request: {request_id}")
                
                async for progress in integration.process_stream(file_paths[request_ids.index(request_id)]):
                    print(f"  [{progress.progress*100:.0f}%] {progress.current_step}")
                    
                    if progress.status == 'completed':
                        result = progress.result
                        print(f"\nCompleted processing: {result.content_type.value}")
                        print(f"Confidence: {result.confidence_score:.2f}")
                        
                        # Show insights
                        insights = integration.get_cross_modal_insights(file_paths[request_ids.index(request_id)])
                        if insights:
                            print(f"\nCross-modal insights found:")
                            for insight in insights:
                                print(f"  - {insight.description}")
                    
                    elif progress.status == 'error':
                        print(f"\nError: {progress.error}")
        
        # Generate report
        print("\n" + "=" * 50)
        print("Processing Report:")
        report = integration.generate_processing_report()
        print(f"Total processed: {report['total_processed']}")
        print(f"Active requests: {report['active_requests']}")
        print(f"Content types: {dict(report['content_types'])}")
        
        # Start WebSocket server for demo
        if '--server' in sys.argv:
            print("\nStarting WebSocket server...")
            await integration.start_websocket_server()
            print("Server running. Press Ctrl+C to stop.")
            await asyncio.Future()  # Run forever
    
    # Run demo
    asyncio.run(demo_async())