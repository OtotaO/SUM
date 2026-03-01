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
from flask import Blueprint, jsonify, send_file, request
from datetime import datetime, timedelta
from typing import Dict, Any, List
import traceback
from pathlib import Path

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
            'uptime_seconds': int(time.time() - self.start_time),
            'checks': {}
        }
        
        # Check system resources
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            health_status['system'] = {
                'memory': {
                    'used_percent': memory.percent,
                    'available_mb': memory.available / (1024 * 1024),
                    'total_mb': memory.total / (1024 * 1024)
                },
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count()
                },
                'disk': self._check_disk_usage()
            }
            
            # Resource thresholds
            if memory.percent > 90:
                health_status['status'] = 'degraded'
                health_status['issues'] = health_status.get('issues', [])
                health_status['issues'].append('High memory usage')
            
            if cpu_percent > 80:
                health_status['status'] = 'degraded'
                health_status['issues'] = health_status.get('issues', [])
                health_status['issues'].append('High CPU usage')
                
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            health_status['status'] = 'degraded'
            health_status['system'] = {'error': str(e)}
        
        # Check core components
        health_status['components'] = self._check_components()
        
        # Overall status
        if any(comp.get('status') == 'unhealthy' 
               for comp in health_status['components'].values()):
            health_status['status'] = 'unhealthy'
        
        return health_status
    
    def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage for data directories."""
        disk_info = {}
        
        try:
            # Check main disk
            disk = psutil.disk_usage('/')
            disk_info['main'] = {
                'used_percent': disk.percent,
                'free_gb': disk.free / (1024**3),
                'total_gb': disk.total / (1024**3)
            }
            
            # Check data directories
            data_dirs = ['uploads', 'temp', 'Data', 'Output']
            for dir_name in data_dirs:
                if os.path.exists(dir_name):
                    try:
                        size = sum(
                            os.path.getsize(os.path.join(dirpath, filename))
                            for dirpath, _, filenames in os.walk(dir_name)
                            for filename in filenames
                        ) / (1024 * 1024)  # MB
                        disk_info[dir_name] = {'size_mb': round(size, 2)}
                    except:
                        disk_info[dir_name] = {'error': 'Unable to calculate size'}
                        
        except Exception as e:
            logger.error(f"Error checking disk usage: {e}")
            disk_info['error'] = str(e)
            
        return disk_info
    
    def _check_components(self) -> Dict[str, Dict[str, Any]]:
        """Check status of core components."""
        components = {}
        
        # Check summarization engine
        try:
            from summarization_engine import BasicSummarizationEngine
            engine = BasicSummarizationEngine()
            test_result = engine.process_text("Test", {"maxTokens": 10})
            components['summarization_engine'] = {
                'status': 'healthy' if test_result else 'unhealthy',
                'type': 'BasicSummarizationEngine'
            }
        except Exception as e:
            components['summarization_engine'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Check file processing
        try:
            from utils.universal_file_processor import UniversalFileProcessor
            processor = UniversalFileProcessor()
            components['file_processor'] = {
                'status': 'healthy',
                'supported_formats': len(processor.supported_extensions)
            }
        except Exception as e:
            components['file_processor'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Check NLTK resources
        try:
            from infrastructure.nltk_manager import NLTKResourceManager
            nltk_manager = NLTKResourceManager()
            required = ['punkt', 'stopwords', 'vader_lexicon']
            missing = nltk_manager.check_missing_resources(required)
            components['nltk_resources'] = {
                'status': 'healthy' if not missing else 'degraded',
                'missing': missing
            }
        except Exception as e:
            components['nltk_resources'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        return components
    
    def record_request(self, response_time: float):
        """Record request metrics."""
        self.request_count += 1
        self.performance_metrics['response_times'].append(response_time)
        
        # Keep only recent history
        max_history = self.performance_metrics['max_history']
        if len(self.performance_metrics['response_times']) > max_history:
            self.performance_metrics['response_times'] = \
                self.performance_metrics['response_times'][-max_history:]
    
    def record_error(self):
        """Record error occurrence."""
        self.error_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        response_times = self.performance_metrics['response_times']
        
        if response_times:
            avg_response = sum(response_times) / len(response_times)
            min_response = min(response_times)
            max_response = max(response_times)
        else:
            avg_response = min_response = max_response = 0
        
        uptime = time.time() - self.start_time
        requests_per_minute = (self.request_count / uptime) * 60 if uptime > 0 else 0
        
        return {
            'uptime_seconds': int(uptime),
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate': (self.error_count / self.request_count * 100) 
                         if self.request_count > 0 else 0,
            'requests_per_minute': round(requests_per_minute, 2),
            'response_times': {
                'average_ms': round(avg_response * 1000, 2),
                'min_ms': round(min_response * 1000, 2),
                'max_ms': round(max_response * 1000, 2),
                'sample_size': len(response_times)
            }
        }


# Global health monitor instance
health_monitor = HealthMonitor()


@health_bp.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    try:
        health_status = health_monitor.check_system_health()
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return jsonify(health_status), status_code
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 503


@health_bp.route('/health/detailed', methods=['GET'])
def detailed_health():
    """Detailed health check with all component statuses."""
    try:
        health_status = health_monitor.check_system_health()
        metrics = health_monitor.get_metrics()
        
        detailed = {
            **health_status,
            'metrics': metrics,
            'version': '1.0.0',
            'environment': os.getenv('ENVIRONMENT', 'development')
        }
        
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return jsonify(detailed), status_code
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 503


@health_bp.route('/metrics', methods=['GET'])
def metrics():
    """Get performance metrics."""
    try:
        return jsonify(health_monitor.get_metrics())
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return jsonify({'error': str(e)}), 500


@health_bp.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint for uptime monitoring."""
    return jsonify({'pong': True, 'timestamp': time.time()})


@health_bp.route('/openapi.yaml', methods=['GET'])
@health_bp.route('/openapi.json', methods=['GET'])
def get_openapi_spec():
    """Serve OpenAPI specification."""
    spec_path = Path(__file__).parent.parent / 'openapi.yaml'
    
    if not spec_path.exists():
        return jsonify({'error': 'OpenAPI specification not found'}), 404
        
    # Return YAML or JSON based on request
    if request.path.endswith('.json'):
        # Convert YAML to JSON
        try:
            import yaml
            with open(spec_path, 'r') as f:
                spec_data = yaml.safe_load(f)
            return jsonify(spec_data)
        except Exception as e:
            logger.error(f"Error converting OpenAPI spec to JSON: {e}")
            return jsonify({'error': 'Failed to load specification'}), 500
    else:
        # Return YAML file
        return send_file(
            spec_path,
            mimetype='application/x-yaml',
            as_attachment=False,
            download_name='openapi.yaml'
        )


# Middleware to track request metrics
@health_bp.before_app_request
def before_request():
    """Record request start time."""
    from flask import g
    g.start_time = time.time()


@health_bp.after_app_request
def after_request(response):
    """Record request metrics."""
    from flask import g
    if hasattr(g, 'start_time'):
        response_time = time.time() - g.start_time
        health_monitor.record_request(response_time)
        
        if response.status_code >= 500:
            health_monitor.record_error()
    
    return response