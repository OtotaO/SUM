"""
service_registry.py - Service Registry and Dependency Injection

Clean service management following Carmack's principles:
- Lazy initialization for fast startup
- Thread-safe singleton pattern
- Clear service interfaces
- Minimal dependencies

Author: ototao
License: Apache License 2.0
"""

import logging
from threading import Lock
from typing import Optional, Dict, Any


logger = logging.getLogger(__name__)


class ServiceRegistry:
    """
    Thread-safe service registry with lazy initialization.
    
    Manages all application services and their dependencies.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._services = {}
        self._initializers = {}
        self._service_lock = Lock()
        self._initialized = True
    
    def register_initializer(self, service_name: str, initializer):
        """
        Register a service initializer (lazy loading).
        
        Args:
            service_name: Name of the service
            initializer: Callable that returns service instance
        """
        with self._service_lock:
            self._initializers[service_name] = initializer
    
    def get_service(self, service_name: str) -> Optional[Any]:
        """
        Get or initialize a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service instance or None if not available
        """
        with self._service_lock:
            # Return existing service
            if service_name in self._services:
                return self._services[service_name]
            
            # Initialize service if initializer exists
            if service_name in self._initializers:
                try:
                    service = self._initializers[service_name]()
                    self._services[service_name] = service
                    logger.info(f"Initialized service: {service_name}")
                    return service
                except Exception as e:
                    logger.error(f"Failed to initialize {service_name}: {e}")
                    return None
            
            return None
    
    def get_all_services(self) -> Dict[str, Any]:
        """Get all initialized services."""
        with self._service_lock:
            return self._services.copy()
    
    def clear(self):
        """Clear all services (useful for testing)."""
        with self._service_lock:
            self._services.clear()


# Global registry instance
registry = ServiceRegistry()


def register_service_initializers():
    """Register all service initializers."""
    
    # Simple summarizer (always available)
    def init_simple_summarizer():
        from summarization_engine import SimpleSUM
        return SimpleSUM()
    
    # Advanced summarizer (lazy-loaded)
    def init_advanced_summarizer():
        from summarization_engine import MagnumOpusSUM
        return MagnumOpusSUM()
    
    # Hierarchical engine (lazy-loaded)
    def init_hierarchical_engine():
        from summarization_engine import HierarchicalDensificationEngine
        return HierarchicalDensificationEngine()
    
    # Streaming engine (lazy-loaded)
    def init_streaming_engine():
        from streaming_engine import StreamingHierarchicalEngine, StreamingConfig
        config = StreamingConfig()
        return StreamingHierarchicalEngine(config)
    
    # Topic modeler (lazy-loaded)
    def init_topic_modeler():
        from Models.topic_modeling import TopicModeler
        return TopicModeler()
    
    # Adaptive compression engine (lazy-loaded)
    def init_adaptive_engine():
        try:
            from adaptive_compression import AdaptiveCompressionEngine
            return AdaptiveCompressionEngine()
        except ImportError:
            logger.warning("Adaptive compression not available")
            return None
    
    # Life compression system (lazy-loaded)
    def init_life_system():
        try:
            from life_compression_system import LifeCompressionSystem
            return LifeCompressionSystem()
        except ImportError:
            logger.warning("Life compression not available")
            return None
    
    # AI engine (lazy-loaded)
    def init_ai_engine():
        try:
            from ai_models import HybridAIEngine
            return HybridAIEngine()
        except ImportError:
            logger.warning("AI models not available")
            return None
    
    # Register all initializers
    registry.register_initializer('simple_summarizer', init_simple_summarizer)
    registry.register_initializer('advanced_summarizer', init_advanced_summarizer)
    registry.register_initializer('hierarchical_engine', init_hierarchical_engine)
    registry.register_initializer('streaming_engine', init_streaming_engine)
    registry.register_initializer('topic_modeler', init_topic_modeler)
    registry.register_initializer('adaptive_engine', init_adaptive_engine)
    registry.register_initializer('life_system', init_life_system)
    registry.register_initializer('ai_engine', init_ai_engine)


# Initialize service registry on import
register_service_initializers()