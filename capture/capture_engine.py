"""
Capture Engine - Zero-Friction Content Capture System

Revolutionary capture system that makes collecting and processing thoughts
as natural as breathing. Integrates with the optimized SumEngine for
sub-second processing.

Author: ototao (optimized with Claude Code)
License: Apache License 2.0
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

# Import the optimized SumEngine
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine import SumEngine

logger = logging.getLogger(__name__)


class CaptureSource(Enum):
    """Source types for captured content."""
    GLOBAL_HOTKEY = "global_hotkey"
    BROWSER_EXTENSION = "browser_extension"
    MOBILE_VOICE = "mobile_voice"
    MOBILE_OCR = "mobile_ocr"
    EMAIL = "email"
    API_WEBHOOK = "api_webhook"


class CaptureStatus(Enum):
    """Processing status for captured content."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CaptureRequest:
    """Represents a content capture request."""
    id: str
    text: str
    source: CaptureSource
    context: Dict[str, Any]
    timestamp: float
    status: CaptureStatus = CaptureStatus.PENDING
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CaptureResult:
    """Result of processing a capture request."""
    request_id: str
    summary: str
    keywords: List[str]
    concepts: Optional[List[str]]
    processing_time: float
    algorithm_used: str
    confidence_score: float


class CaptureEngine:
    """
    Zero-friction capture engine with intelligent processing.
    
    Features:
    - Sub-second processing for most content
    - Context-aware summarization
    - Background processing with callbacks
    - Cross-platform compatibility
    - Beautiful progress indication
    """
    
    def __init__(self, max_workers: int = 4):
        """Initialize capture engine with optimized SumEngine backend."""
        self.sum_engine = SumEngine()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._requests: Dict[str, CaptureRequest] = {}
        self._results: Dict[str, CaptureResult] = {}
        self._callbacks: List[Callable[[CaptureResult], None]] = []
        self._lock = threading.Lock()
        
        # Performance tracking
        self._stats = {
            'total_captures': 0,
            'successful_captures': 0,
            'average_processing_time': 0.0,
            'captures_by_source': {},
            'active_requests': 0
        }
        
        logger.info("CaptureEngine initialized with SumEngine backend")
    
    def capture_text(self, 
                    text: str, 
                    source: CaptureSource,
                    context: Optional[Dict[str, Any]] = None,
                    callback: Optional[Callable[[CaptureResult], None]] = None) -> str:
        """
        Capture and process text with zero-friction experience.
        
        Args:
            text: Content to capture and process
            source: Source of the capture (hotkey, browser, etc.)
            context: Additional context for processing
            callback: Optional callback for async result delivery
            
        Returns:
            request_id: Unique identifier for tracking the request
        """
        # Generate unique request ID
        request_id = f"capture_{int(time.time() * 1000)}_{source.value}"
        
        # Create capture request
        request = CaptureRequest(
            id=request_id,
            text=text,
            source=source,
            context=context or {},
            timestamp=time.time(),
            metadata=self._extract_metadata(text, source, context)
        )
        
        with self._lock:
            self._requests[request_id] = request
            self._stats['total_captures'] += 1
            self._stats['active_requests'] += 1
            
            # Track by source
            source_key = source.value
            self._stats['captures_by_source'][source_key] = (
                self._stats['captures_by_source'].get(source_key, 0) + 1
            )
        
        # Process asynchronously for zero-friction experience
        future = self.executor.submit(self._process_capture, request, callback)
        
        logger.info(f"Capture request {request_id} submitted from {source.value}")
        return request_id
    
    def _process_capture(self, 
                        request: CaptureRequest, 
                        callback: Optional[Callable[[CaptureResult], None]] = None):
        """Process capture request with intelligent algorithms."""
        start_time = time.time()
        
        try:
            # Update status
            request.status = CaptureStatus.PROCESSING
            
            # Context-aware processing parameters
            processing_params = self._get_processing_params(request)
            
            # Use optimized SumEngine for processing
            result = self.sum_engine.summarize(
                text=request.text,
                max_length=processing_params['max_length'],
                algorithm=processing_params['algorithm']
            )
            
            # Calculate confidence score based on various factors
            confidence = self._calculate_confidence(request, result)
            
            # Create capture result
            processing_time = time.time() - start_time
            capture_result = CaptureResult(
                request_id=request.id,
                summary=result['summary'],
                keywords=result['keywords'],
                concepts=result.get('concepts'),
                processing_time=processing_time,
                algorithm_used=result['algorithm_used'],
                confidence_score=confidence
            )
            
            # Store result
            with self._lock:
                self._results[request.id] = capture_result
                request.status = CaptureStatus.COMPLETED
                self._stats['successful_captures'] += 1
                self._stats['active_requests'] -= 1
                
                # Update average processing time
                total_successful = self._stats['successful_captures']
                current_avg = self._stats['average_processing_time']
                self._stats['average_processing_time'] = (
                    (current_avg * (total_successful - 1) + processing_time) / total_successful
                )
            
            # Execute callback if provided
            if callback:
                callback(capture_result)
            
            # Execute registered callbacks
            for cb in self._callbacks:
                try:
                    cb(capture_result)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            logger.info(f"Capture {request.id} processed successfully in {processing_time:.3f}s")
            
        except Exception as e:
            # Handle processing failure
            request.status = CaptureStatus.FAILED
            with self._lock:
                self._stats['active_requests'] -= 1
            
            logger.error(f"Capture processing failed for {request.id}: {e}", exc_info=True)
    
    def _extract_metadata(self, 
                         text: str, 
                         source: CaptureSource, 
                         context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metadata from capture request."""
        metadata = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'source': source.value,
            'timestamp': time.time()
        }
        
        # Add context-specific metadata
        if context:
            if source == CaptureSource.BROWSER_EXTENSION:
                metadata.update({
                    'url': context.get('url'),
                    'page_title': context.get('title'),
                    'selection_length': context.get('selection_length', len(text))
                })
            elif source == CaptureSource.EMAIL:
                metadata.update({
                    'sender': context.get('sender'),
                    'subject': context.get('subject'),
                    'email_type': context.get('type', 'unknown')
                })
        
        return metadata
    
    def _get_processing_params(self, request: CaptureRequest) -> Dict[str, Any]:
        """Get context-aware processing parameters."""
        params = {
            'max_length': 100,
            'algorithm': 'auto'
        }
        
        # Adjust based on source and context
        if request.source == CaptureSource.GLOBAL_HOTKEY:
            # Prioritize speed for hotkey captures
            params['algorithm'] = 'fast'
            params['max_length'] = 80
            
        elif request.source == CaptureSource.BROWSER_EXTENSION:
            # Context-aware processing for web content
            if 'article' in request.context.get('page_type', '').lower():
                params['algorithm'] = 'quality'
                params['max_length'] = 150
            elif 'email' in request.context.get('page_type', '').lower():
                params['algorithm'] = 'fast'
                params['max_length'] = 100
                
        elif request.source == CaptureSource.EMAIL:
            # Email-specific processing
            email_type = request.context.get('type', '')
            if email_type == 'newsletter':
                params['algorithm'] = 'hierarchical'
                params['max_length'] = 200
            else:
                params['algorithm'] = 'quality'
                params['max_length'] = 120
        
        # Adjust based on content length
        word_count = len(request.text.split())
        if word_count > 1000:
            params['algorithm'] = 'hierarchical'
            params['max_length'] = min(params['max_length'] * 2, 300)
        
        return params
    
    def _calculate_confidence(self, request: CaptureRequest, result: Dict[str, Any]) -> float:
        """Calculate confidence score for the processing result."""
        base_confidence = 0.8
        
        # Adjust based on compression ratio
        stats = result.get('stats', {})
        compression_ratio = stats.get('compression_ratio', 0.1)
        
        if 0.1 <= compression_ratio <= 0.3:
            confidence_adjustment = 0.1  # Good compression
        elif compression_ratio < 0.05:
            confidence_adjustment = -0.2  # Over-compression
        else:
            confidence_adjustment = -0.1  # Under-compression
        
        # Adjust based on keyword extraction quality
        keywords = result.get('keywords', [])
        if len(keywords) >= 3:
            confidence_adjustment += 0.1
        
        # Adjust based on source reliability
        if request.source == CaptureSource.GLOBAL_HOTKEY:
            confidence_adjustment += 0.05  # Direct user selection
        elif request.source == CaptureSource.EMAIL:
            confidence_adjustment -= 0.05  # May have formatting issues
        
        return max(0.1, min(1.0, base_confidence + confidence_adjustment))
    
    def get_result(self, request_id: str) -> Optional[CaptureResult]:
        """Get processing result by request ID."""
        with self._lock:
            return self._results.get(request_id)
    
    def get_request_status(self, request_id: str) -> Optional[CaptureStatus]:
        """Get current status of a capture request."""
        with self._lock:
            request = self._requests.get(request_id)
            return request.status if request else None
    
    def register_callback(self, callback: Callable[[CaptureResult], None]):
        """Register a callback for all capture results."""
        self._callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get capture engine performance statistics."""
        with self._lock:
            return self._stats.copy()
    
    def shutdown(self):
        """Gracefully shutdown the capture engine."""
        logger.info("Shutting down CaptureEngine...")
        self.executor.shutdown(wait=True)
        logger.info("CaptureEngine shutdown complete")


# Global instance for easy access
capture_engine = CaptureEngine()