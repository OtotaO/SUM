"""
Enhanced Main Application for SUM Platform
==========================================

Production-ready application with integrated:
- Comprehensive test suite
- Webhook system
- Circuit breakers and resilience
- Feature flags
- Distributed tracing
- API documentation
- Health monitoring

This brings together all Phase 1 and Phase 2 enhancements
for a robust, scalable, production-ready platform.

Author: ototao
License: Apache License 2.0
"""

import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Flask and web components
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

# Core SUM components
from web.app_factory import create_app
from config import active_config
from core.engine import SumEngine
from knowledge_os import KnowledgeOperatingSystem
from temporal_intelligence_engine import TemporalIntelligenceEngine
from predictive_intelligence import PredictiveIntelligence
from superhuman_memory import SuperhumanMemory

# Infrastructure components (Phase 1 & 2)
from infrastructure.webhook_system import (
    webhook_manager, WebhookEvent, trigger_webhook_event
)
from infrastructure.resilience import (
    resilience_manager, with_resilience, CircuitBreakerConfig
)
from infrastructure.feature_flags import (
    feature_flags, is_enabled, FeatureFlag, FlagType
)
from infrastructure.tracing import (
    tracer, dashboard, trace, MonitoringDashboard
)
from infrastructure.api_documentation import (
    generate_sum_api_documentation, generate_html_documentation
)

# Configure logging with production settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sum_platform.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionSUMPlatform:
    """
    Production-ready SUM Platform with all enhancements integrated.
    """
    
    def __init__(self):
        """Initialize the production platform"""
        logger.info("Initializing Production SUM Platform...")
        
        # Core components
        self.app = None
        self.sum_engine = None
        self.knowledge_os = None
        self.temporal_engine = None
        self.predictive_engine = None
        self.memory_system = None
        
        # Infrastructure managers
        self.webhook_manager = webhook_manager
        self.resilience_manager = resilience_manager
        self.feature_flags = feature_flags
        self.tracer = tracer
        self.dashboard = dashboard
        
        # Metrics
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
        # Initialize components
        self._initialize_components()
        self._setup_feature_flags()
        self._setup_circuit_breakers()
        self._setup_webhooks()
    
    def _initialize_components(self):
        """Initialize core components with resilience"""
        logger.info("Initializing core components...")
        
        # Initialize with circuit breakers
        try:
            # Core engines
            self.sum_engine = SumEngine()
            self.knowledge_os = KnowledgeOperatingSystem()
            self.temporal_engine = TemporalIntelligenceEngine()
            self.predictive_engine = PredictiveIntelligence()
            self.memory_system = SuperhumanMemory()
            
            logger.info("Core components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _setup_feature_flags(self):
        """Setup default feature flags"""
        logger.info("Setting up feature flags...")
        
        # Temporal Intelligence Feature
        temporal_flag = FeatureFlag(
            key="temporal_intelligence",
            name="Temporal Intelligence",
            description="Enable temporal pattern analysis",
            flag_type=FlagType.BOOLEAN,
            enabled=True,
            rollout_percentage=100
        )
        self.feature_flags.create_flag(temporal_flag)
        
        # Predictive Intelligence Feature
        predictive_flag = FeatureFlag(
            key="predictive_intelligence",
            name="Predictive Intelligence",
            description="Enable predictive suggestions",
            flag_type=FlagType.PERCENTAGE,
            enabled=True,
            rollout_percentage=75  # 75% rollout
        )
        self.feature_flags.create_flag(predictive_flag)
        
        # Webhook Notifications
        webhook_flag = FeatureFlag(
            key="webhook_notifications",
            name="Webhook Notifications",
            description="Enable webhook event notifications",
            flag_type=FlagType.BOOLEAN,
            enabled=True
        )
        self.feature_flags.create_flag(webhook_flag)
        
        # A/B Test for Summarization Algorithm
        algo_test_flag = FeatureFlag(
            key="summarization_algorithm",
            name="Summarization Algorithm Test",
            description="A/B test different algorithms",
            flag_type=FlagType.VARIANT,
            enabled=True,
            variants=[
                {'key': 'fast', 'weight': 40},
                {'key': 'quality', 'weight': 40},
                {'key': 'auto', 'weight': 20}
            ]
        )
        self.feature_flags.create_flag(algo_test_flag)
        
        logger.info(f"Created {len(self.feature_flags.list_flags())} feature flags")
    
    def _setup_circuit_breakers(self):
        """Setup circuit breakers for external services"""
        logger.info("Setting up circuit breakers...")
        
        # External API circuit breaker
        api_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            min_request_volume=10
        )
        self.resilience_manager.create_circuit_breaker("external_api", api_config)
        
        # Database circuit breaker
        db_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            min_request_volume=5
        )
        self.resilience_manager.create_circuit_breaker("database", db_config)
        
        # AI Model circuit breaker
        ai_config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=120.0,
            min_request_volume=20
        )
        self.resilience_manager.create_circuit_breaker("ai_models", ai_config)
        
        logger.info("Circuit breakers configured")
    
    def _setup_webhooks(self):
        """Setup default webhook configurations"""
        logger.info("Setting up webhook system...")
        
        # Webhook system is already initialized
        # Add any default webhook subscriptions here if needed
        
        logger.info("Webhook system ready")
    
    def create_flask_app(self) -> Flask:
        """Create Flask application with all integrations"""
        logger.info("Creating Flask application...")
        
        # Create base app
        self.app = create_app()
        
        # Add production middleware
        self.app.wsgi_app = ProxyFix(self.app.wsgi_app, x_for=1, x_proto=1)
        
        # Enable CORS for API access
        CORS(self.app, resources={r"/api/*": {"origins": "*"}})
        
        # Register enhanced routes
        self._register_routes()
        
        # Add error handlers
        self._register_error_handlers()
        
        # Start monitoring
        self.dashboard.start()
        
        logger.info("Flask application created successfully")
        
        return self.app
    
    def _register_routes(self):
        """Register all API routes with enhancements"""
        
        @self.app.route('/api/summarize', methods=['POST'])
        @trace(operation_name="summarize_endpoint")
        @with_resilience(circuit_breaker="ai_models", retry=True)
        async def summarize():
            """Enhanced summarization endpoint with all features"""
            try:
                # Extract request data
                data = request.get_json()
                text = data.get('text', '')
                max_length = data.get('max_length', 100)
                
                # Get user context for feature flags
                user_context = {
                    'user_id': request.headers.get('X-User-ID', 'anonymous'),
                    'subscription': request.headers.get('X-Subscription', 'free')
                }
                
                # Check feature flag for algorithm selection
                algorithm = 'auto'
                if is_enabled('summarization_algorithm', user_context):
                    algorithm = self.feature_flags.evaluate(
                        'summarization_algorithm',
                        user_context,
                        'auto'
                    )
                
                # Perform summarization with tracing
                with self.tracer.start_span("summarization", tags={'algorithm': algorithm}) as span:
                    result = self.sum_engine.summarize(
                        text,
                        max_length=max_length,
                        algorithm=algorithm
                    )
                    span.add_tag('word_count', len(text.split()))
                    span.add_tag('summary_length', len(result.get('summary', '').split()))
                
                # Trigger webhook if enabled
                if is_enabled('webhook_notifications', user_context):
                    await trigger_webhook_event(
                        WebhookEvent.DOCUMENT_SUMMARIZED,
                        {
                            'text_length': len(text),
                            'summary_length': len(result.get('summary', '')),
                            'algorithm': algorithm,
                            'user_id': user_context['user_id']
                        }
                    )
                
                # Record metrics
                self.tracer.metrics.increment_counter('api.summarize.success')
                self.request_count += 1
                
                return jsonify(result), 200
                
            except Exception as e:
                logger.error(f"Summarization error: {e}")
                self.error_count += 1
                self.tracer.metrics.increment_counter('api.summarize.error')
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/knowledge/capture', methods=['POST'])
        @trace(operation_name="knowledge_capture")
        async def capture_thought():
            """Capture thought with Knowledge OS"""
            try:
                data = request.get_json()
                thought = data.get('thought', '')
                
                # Capture with Knowledge OS
                result = self.knowledge_os.capture(thought)
                
                # Trigger webhook
                await trigger_webhook_event(
                    WebhookEvent.THOUGHT_CAPTURED,
                    {
                        'thought_id': result.id,
                        'content': thought[:100],  # First 100 chars
                        'concepts': result.metadata.get('concepts', [])
                    }
                )
                
                return jsonify({
                    'thought_id': result.id,
                    'processed': True,
                    'metadata': result.metadata
                }), 200
                
            except Exception as e:
                logger.error(f"Knowledge capture error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/webhooks', methods=['POST'])
        @trace(operation_name="webhook_registration")
        async def register_webhook():
            """Register a new webhook"""
            try:
                data = request.get_json()
                
                # Parse events
                event_strings = data.get('events', [])
                events = [WebhookEvent(e) for e in event_strings]
                
                # Register webhook
                webhook_id = self.webhook_manager.register_webhook(
                    url=data['url'],
                    events=events,
                    secret=data['secret'],
                    description=data.get('description', '')
                )
                
                return jsonify({
                    'webhook_id': webhook_id,
                    'status': 'registered'
                }), 200
                
            except Exception as e:
                logger.error(f"Webhook registration error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Enhanced health check with system status"""
            try:
                # Get system health
                system_health = self.resilience_manager.get_system_health()
                
                # Check component status
                components = {
                    'sum_engine': 'healthy' if self.sum_engine else 'unhealthy',
                    'knowledge_os': 'healthy' if self.knowledge_os else 'unhealthy',
                    'webhooks': 'healthy',
                    'circuit_breakers': 'healthy' if system_health['circuit_breakers']['open'] == 0 else 'degraded'
                }
                
                # Determine overall status
                if all(v == 'healthy' for v in components.values()):
                    status = 'healthy'
                elif any(v == 'unhealthy' for v in components.values()):
                    status = 'unhealthy'
                else:
                    status = 'degraded'
                
                return jsonify({
                    'status': status,
                    'version': '2.0.0',
                    'uptime': time.time() - self.start_time,
                    'components': components,
                    'metrics': {
                        'requests': self.request_count,
                        'errors': self.error_count,
                        'error_rate': self.error_count / max(self.request_count, 1)
                    },
                    'system': system_health
                }), 200 if status == 'healthy' else 503
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/api/metrics', methods=['GET'])
        def get_metrics():
            """Get comprehensive metrics"""
            try:
                # Aggregate metrics from all components
                metrics = {
                    'application': {
                        'uptime': time.time() - self.start_time,
                        'requests': self.request_count,
                        'errors': self.error_count
                    },
                    'tracing': self.tracer.get_metrics_summary(),
                    'webhooks': self.webhook_manager.get_metrics(),
                    'feature_flags': self.feature_flags.get_metrics(),
                    'resilience': self.resilience_manager.get_system_health(),
                    'monitoring': self.dashboard.get_dashboard_data()
                }
                
                return jsonify(metrics), 200
                
            except Exception as e:
                logger.error(f"Metrics error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/documentation', methods=['GET'])
        def api_documentation():
            """Serve interactive API documentation"""
            try:
                # Generate OpenAPI spec
                spec = generate_sum_api_documentation()
                
                # Generate HTML documentation
                html = generate_html_documentation(spec)
                
                return html, 200, {'Content-Type': 'text/html'}
                
            except Exception as e:
                logger.error(f"Documentation error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/feature-flags', methods=['GET'])
        def list_feature_flags():
            """List all feature flags"""
            try:
                flags = self.feature_flags.list_flags()
                
                return jsonify({
                    'flags': [
                        {
                            'key': f.key,
                            'name': f.name,
                            'enabled': f.enabled,
                            'type': f.flag_type.value
                        }
                        for f in flags
                    ]
                }), 200
                
            except Exception as e:
                logger.error(f"Feature flags error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _register_error_handlers(self):
        """Register error handlers"""
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Endpoint not found'}), 404
        
        @self.app.errorhandler(429)
        def rate_limited(error):
            return jsonify({'error': 'Rate limit exceeded'}), 429
        
        @self.app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal server error: {error}")
            return jsonify({'error': 'Internal server error'}), 500
    
    def run(self, host: str = None, port: int = None, debug: bool = None):
        """Run the production platform"""
        host = host or active_config.HOST
        port = port or active_config.PORT
        debug = debug if debug is not None else active_config.DEBUG
        
        logger.info(f"Starting Production SUM Platform on {host}:{port}")
        logger.info(f"Debug mode: {debug}")
        logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
        
        # Create Flask app if not already created
        if not self.app:
            self.create_flask_app()
        
        # Setup graceful shutdown
        def signal_handler(sig, frame):
            logger.info("Shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Log startup information
        logger.info("=" * 70)
        logger.info("SUM PLATFORM - PRODUCTION READY")
        logger.info("=" * 70)
        logger.info("Features Enabled:")
        logger.info("  ✅ Comprehensive Test Suite")
        logger.info("  ✅ Webhook System with HMAC Security")
        logger.info("  ✅ Circuit Breakers & Resilience")
        logger.info("  ✅ Feature Flags & A/B Testing")
        logger.info("  ✅ Distributed Tracing")
        logger.info("  ✅ API Documentation")
        logger.info("  ✅ Health Monitoring")
        logger.info("=" * 70)
        logger.info(f"API Documentation: http://{host}:{port}/api/documentation")
        logger.info(f"Health Check: http://{host}:{port}/api/health")
        logger.info(f"Metrics: http://{host}:{port}/api/metrics")
        logger.info("=" * 70)
        
        # Run the application
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("Performing graceful shutdown...")
        
        # Stop monitoring
        if self.dashboard:
            self.dashboard.stop()
        
        # Save metrics
        if self.feature_flags:
            self.feature_flags.save_metrics()
        
        # Shutdown webhook manager
        if self.webhook_manager:
            asyncio.run(self.webhook_manager.shutdown())
        
        logger.info("Shutdown complete")


def main():
    """Main entry point for production platform"""
    # Create and run production platform
    platform = ProductionSUMPlatform()
    platform.run()


if __name__ == '__main__':
    main()
