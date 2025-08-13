"""
health.py - Health Check and Monitoring API

Provides endpoints for:
- System health checks
- Component status monitoring
- Performance metrics
- Resource usage tracking

Author: SUM Development Team
License: Apache License 2.0
"""

import os
import time
import psutil
import logging
from flask import Blueprint, jsonify
from datetime import datetime, timedelta
from typing import Dict, Any, List
import traceback

from memory.semantic_memory import get_semantic_memory_engine
from memory.knowledge_graph import get_knowledge_graph_engine
from application.feedback_system import get_feedback_system
from Utils.error_handler import error_monitor

logger = logging.getLogger(__name__)
health_bp = Blueprint('health', __name__)


class HealthMonitor:
    """Monitors system health and component status."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.last_check_time = time.time()
        self.component_status = {}
        self.performance_metrics = {
            'response_times': [],
            'max_history': 1000
        }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': time.time() - self.start_time,
            'components': {},
            'resources': self._check_resources(),
            'performance': self._get_performance_metrics()
        }
        
        # Check each component
        components_to_check = [
            ('semantic_memory', self._check_semantic_memory),
            ('knowledge_graph', self._check_knowledge_graph),
            ('feedback_system', self._check_feedback_system),
            ('file_system', self._check_file_system),
            ('api', self._check_api_health)
        ]
        
        all_healthy = True
        for component_name, check_func in components_to_check:
            try:
                component_health = check_func()
                health_status['components'][component_name] = component_health
                if component_health['status'] != 'healthy':
                    all_healthy = False
            except Exception as e:
                logger.error(f"Error checking {component_name}: {e}")
                health_status['components'][component_name] = {
                    'status': 'error',
                    'message': str(e),
                    'last_check': datetime.utcnow().isoformat()
                }
                all_healthy = False
        
        # Overall status
        if not all_healthy:
            health_status['status'] = 'degraded'
        
        if health_status['resources']['memory_percent'] > 90:
            health_status['status'] = 'critical'
            health_status['warnings'] = health_status.get('warnings', [])
            health_status['warnings'].append('High memory usage detected')
        
        self.last_check_time = time.time()
        return health_status
    
    def _check_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Process-specific info
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024 * 1024 * 1024),
                'process_memory_mb': process_memory.rss / (1024 * 1024),
                'thread_count': process.num_threads()
            }
        except Exception as e:
            logger.error(f"Error checking resources: {e}")
            return {
                'error': str(e),
                'cpu_percent': 0,
                'memory_percent': 0
            }
    
    def _check_semantic_memory(self) -> Dict[str, Any]:
        """Check semantic memory component health."""
        try:
            engine = get_semantic_memory_engine()
            
            # Try a simple operation
            test_embedding = engine.generate_embedding("health check")
            
            # Get stats
            stats = engine.get_stats() if hasattr(engine, 'get_stats') else {}
            
            return {
                'status': 'healthy',
                'message': 'Semantic memory operational',
                'stats': stats,
                'embedding_dim': len(test_embedding) if test_embedding else 0,
                'last_check': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Semantic memory error: {str(e)}',
                'last_check': datetime.utcnow().isoformat()
            }
    
    def _check_knowledge_graph(self) -> Dict[str, Any]:
        """Check knowledge graph component health."""
        try:
            engine = get_knowledge_graph_engine()
            
            # Get stats
            stats = engine.get_stats() if hasattr(engine, 'get_stats') else {}
            
            return {
                'status': 'healthy',
                'message': 'Knowledge graph operational',
                'stats': stats,
                'backend': engine.backend_type if hasattr(engine, 'backend_type') else 'unknown',
                'last_check': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Knowledge graph error: {str(e)}',
                'last_check': datetime.utcnow().isoformat()
            }
    
    def _check_feedback_system(self) -> Dict[str, Any]:
        """Check feedback system health."""
        try:
            system = get_feedback_system()
            
            # Get insights
            insights = system.get_insights()
            
            return {
                'status': 'healthy',
                'message': 'Feedback system operational',
                'total_feedback': insights.get('total_feedback_count', 0),
                'average_rating': insights.get('average_rating', 0),
                'last_check': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Feedback system error: {str(e)}',
                'last_check': datetime.utcnow().isoformat()
            }
    
    def _check_file_system(self) -> Dict[str, Any]:
        """Check file system health."""
        try:
            # Check critical directories
            from config import Config
            
            directories = {
                'uploads': Config.UPLOADS_DIR,
                'temp': Config.TEMP_DIR,
                'output': Config.OUTPUT_DIR
            }
            
            all_accessible = True
            issues = []
            
            for name, path in directories.items():
                if not os.path.exists(path):
                    issues.append(f"{name} directory missing: {path}")
                    all_accessible = False
                elif not os.access(path, os.W_OK):
                    issues.append(f"{name} directory not writable: {path}")
                    all_accessible = False
            
            return {
                'status': 'healthy' if all_accessible else 'unhealthy',
                'message': 'File system accessible' if all_accessible else '; '.join(issues),
                'directories_checked': len(directories),
                'last_check': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'File system error: {str(e)}',
                'last_check': datetime.utcnow().isoformat()
            }
    
    def _check_api_health(self) -> Dict[str, Any]:
        """Check API health metrics."""
        try:
            # Get error statistics
            error_stats = error_monitor.get_error_stats()
            
            # Calculate success rate
            total_requests = self.request_count
            error_count = error_stats.get('total_errors', 0)
            success_rate = ((total_requests - error_count) / total_requests * 100) if total_requests > 0 else 100
            
            return {
                'status': 'healthy' if success_rate > 95 else 'degraded',
                'message': f'API success rate: {success_rate:.1f}%',
                'total_requests': total_requests,
                'error_count': error_count,
                'error_rate': error_stats.get('error_rate', 0),
                'last_check': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'API health check error: {str(e)}',
                'last_check': datetime.utcnow().isoformat()
            }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        if not self.performance_metrics['response_times']:
            return {
                'average_response_time_ms': 0,
                'p95_response_time_ms': 0,
                'p99_response_time_ms': 0
            }
        
        times = sorted(self.performance_metrics['response_times'])
        count = len(times)
        
        return {
            'average_response_time_ms': sum(times) / count,
            'p95_response_time_ms': times[int(count * 0.95)],
            'p99_response_time_ms': times[int(count * 0.99)],
            'request_count': self.request_count
        }
    
    def record_request(self, response_time_ms: float):
        """Record a request for metrics."""
        self.request_count += 1
        self.performance_metrics['response_times'].append(response_time_ms)
        
        # Limit history size
        if len(self.performance_metrics['response_times']) > self.performance_metrics['max_history']:
            self.performance_metrics['response_times'] = \
                self.performance_metrics['response_times'][-self.performance_metrics['max_history']:]


# Global health monitor instance
health_monitor = HealthMonitor()


# Routes
@health_bp.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    try:
        # Quick health check
        return jsonify({
            'status': 'ok',
            'timestamp': datetime.utcnow().isoformat(),
            'service': 'SUM Knowledge Crystallization System'
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 503


@health_bp.route('/health/detailed', methods=['GET'])
def detailed_health_check():
    """Detailed health check with component status."""
    try:
        health_status = health_monitor.check_system_health()
        
        # Determine HTTP status code
        if health_status['status'] == 'healthy':
            status_code = 200
        elif health_status['status'] == 'degraded':
            status_code = 200  # Still operational
        else:
            status_code = 503
        
        return jsonify(health_status), status_code
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 503


@health_bp.route('/health/ready', methods=['GET'])
def readiness_check():
    """Readiness probe for container orchestration."""
    try:
        # Check if all critical components are ready
        health_status = health_monitor.check_system_health()
        
        critical_components = ['semantic_memory', 'knowledge_graph']
        all_ready = all(
            health_status['components'].get(comp, {}).get('status') == 'healthy'
            for comp in critical_components
        )
        
        if all_ready:
            return jsonify({
                'ready': True,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        else:
            return jsonify({
                'ready': False,
                'reason': 'Critical components not ready',
                'timestamp': datetime.utcnow().isoformat()
            }), 503
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return jsonify({
            'ready': False,
            'error': str(e)
        }), 503


@health_bp.route('/health/live', methods=['GET'])
def liveness_check():
    """Liveness probe for container orchestration."""
    try:
        # Simple check that the service is responding
        return jsonify({
            'alive': True,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return jsonify({
            'alive': False,
            'error': str(e)
        }), 503


@health_bp.route('/metrics', methods=['GET'])
def metrics():
    """Expose metrics in Prometheus format."""
    try:
        # Get current metrics
        health_status = health_monitor.check_system_health()
        resources = health_status['resources']
        performance = health_status['performance']
        
        # Format as Prometheus metrics
        metrics_lines = [
            '# HELP sum_uptime_seconds Time since service start',
            '# TYPE sum_uptime_seconds counter',
            f'sum_uptime_seconds {health_status["uptime_seconds"]:.2f}',
            '',
            '# HELP sum_cpu_usage_percent CPU usage percentage',
            '# TYPE sum_cpu_usage_percent gauge',
            f'sum_cpu_usage_percent {resources.get("cpu_percent", 0):.2f}',
            '',
            '# HELP sum_memory_usage_percent Memory usage percentage',
            '# TYPE sum_memory_usage_percent gauge',
            f'sum_memory_usage_percent {resources.get("memory_percent", 0):.2f}',
            '',
            '# HELP sum_request_count Total number of requests',
            '# TYPE sum_request_count counter',
            f'sum_request_count {performance.get("request_count", 0)}',
            '',
            '# HELP sum_response_time_milliseconds Response time in milliseconds',
            '# TYPE sum_response_time_milliseconds histogram',
            f'sum_response_time_milliseconds_avg {performance.get("average_response_time_ms", 0):.2f}',
            f'sum_response_time_milliseconds_p95 {performance.get("p95_response_time_ms", 0):.2f}',
            f'sum_response_time_milliseconds_p99 {performance.get("p99_response_time_ms", 0):.2f}',
        ]
        
        return '\n'.join(metrics_lines), 200, {'Content-Type': 'text/plain'}
    except Exception as e:
        logger.error(f"Metrics generation failed: {e}")
        return f"# Error generating metrics: {e}", 500


# Middleware to track response times
@health_bp.before_app_request
def before_request():
    """Record request start time."""
    from flask import g
    g.request_start_time = time.time()


@health_bp.after_app_request
def after_request(response):
    """Record request completion."""
    from flask import g
    if hasattr(g, 'request_start_time'):
        response_time_ms = (time.time() - g.request_start_time) * 1000
        health_monitor.record_request(response_time_ms)
    return response


# Export blueprint
__all__ = ['health_bp', 'health_monitor']