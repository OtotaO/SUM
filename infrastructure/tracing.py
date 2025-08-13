"""
Distributed Tracing and Monitoring for SUM Platform
====================================================

Production-grade observability infrastructure with:
- OpenTelemetry integration for distributed tracing
- Comprehensive metrics collection
- Performance monitoring
- Error tracking
- Request correlation
- Custom dashboards

Based on industry standards like Jaeger, Prometheus, and Grafana.

Author: ototao
License: Apache License 2.0
"""

import asyncio
import functools
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import threading
import traceback

try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenTelemetry not installed. Using fallback tracing.")

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class TraceSpan:
    """
    Represents a span in a distributed trace.
    """
    
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str = ""
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"
    error: Optional[str] = None
    
    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self, error: Optional[Exception] = None):
        """Finish the span"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        if error:
            self.status = "ERROR"
            self.error = str(error)
            self.logs.append({
                'timestamp': self.end_time,
                'level': 'ERROR',
                'message': str(error),
                'traceback': traceback.format_exc()
            })
    
    def add_tag(self, key: str, value: Any):
        """Add a tag to the span"""
        self.tags[key] = value
    
    def add_log(self, message: str, level: str = "INFO", **kwargs):
        """Add a log entry to the span"""
        log_entry = {
            'timestamp': time.time(),
            'level': level,
            'message': message,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export"""
        return {
            'span_id': self.span_id,
            'trace_id': self.trace_id,
            'parent_span_id': self.parent_span_id,
            'operation_name': self.operation_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'tags': self.tags,
            'logs': self.logs,
            'status': self.status,
            'error': self.error,
            'metrics': self.metrics
        }


class TraceContext:
    """
    Thread-local trace context for managing current trace and span.
    """
    
    def __init__(self):
        self._local = threading.local()
    
    @property
    def current_trace(self) -> Optional[str]:
        """Get current trace ID"""
        return getattr(self._local, 'trace_id', None)
    
    @current_trace.setter
    def current_trace(self, trace_id: str):
        """Set current trace ID"""
        self._local.trace_id = trace_id
    
    @property
    def current_span(self) -> Optional[TraceSpan]:
        """Get current span"""
        return getattr(self._local, 'span', None)
    
    @current_span.setter
    def current_span(self, span: Optional[TraceSpan]):
        """Set current span"""
        self._local.span = span
    
    @property
    def span_stack(self) -> List[TraceSpan]:
        """Get span stack"""
        if not hasattr(self._local, 'span_stack'):
            self._local.span_stack = []
        return self._local.span_stack
    
    def push_span(self, span: TraceSpan):
        """Push span to stack"""
        self.span_stack.append(span)
        self.current_span = span
    
    def pop_span(self) -> Optional[TraceSpan]:
        """Pop span from stack"""
        if self.span_stack:
            span = self.span_stack.pop()
            self.current_span = self.span_stack[-1] if self.span_stack else None
            return span
        return None
    
    def clear(self):
        """Clear context"""
        self._local.trace_id = None
        self._local.span = None
        self._local.span_stack = []


class MetricsCollector:
    """
    Collects and aggregates metrics.
    """
    
    def __init__(self, 
                 window_size: int = 1000,
                 aggregation_interval: float = 60.0):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Size of sliding window for metrics
            aggregation_interval: Interval for aggregation in seconds
        """
        self.window_size = window_size
        self.aggregation_interval = aggregation_interval
        
        # Metrics storage
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.summaries: Dict[str, Dict[str, Any]] = {}
        
        # Metadata
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Aggregation
        self.last_aggregation = time.time()
        self.aggregated_metrics: Dict[str, Any] = {}
    
    def increment_counter(self, 
                         name: str,
                         value: float = 1.0,
                         tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self.lock:
            key = self._make_key(name, tags)
            self.counters[key] += value
            
            # Update metadata
            if key not in self.metric_metadata:
                self.metric_metadata[key] = {
                    'type': MetricType.COUNTER,
                    'name': name,
                    'tags': tags or {},
                    'created_at': time.time()
                }
    
    def set_gauge(self,
                 name: str,
                 value: float,
                 tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        with self.lock:
            key = self._make_key(name, tags)
            self.gauges[key] = value
            
            # Update metadata
            if key not in self.metric_metadata:
                self.metric_metadata[key] = {
                    'type': MetricType.GAUGE,
                    'name': name,
                    'tags': tags or {},
                    'created_at': time.time()
                }
    
    def record_histogram(self,
                        name: str,
                        value: float,
                        tags: Optional[Dict[str, str]] = None):
        """Record a histogram value"""
        with self.lock:
            key = self._make_key(name, tags)
            self.histograms[key].append(value)
            
            # Update metadata
            if key not in self.metric_metadata:
                self.metric_metadata[key] = {
                    'type': MetricType.HISTOGRAM,
                    'name': name,
                    'tags': tags or {},
                    'created_at': time.time()
                }
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Create metric key from name and tags"""
        if not tags:
            return name
        
        tag_str = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}:{tag_str}"
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile from values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def aggregate_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics for reporting"""
        with self.lock:
            current_time = time.time()
            
            # Check if aggregation is needed
            if current_time - self.last_aggregation < self.aggregation_interval:
                return self.aggregated_metrics
            
            aggregated = {
                'timestamp': current_time,
                'interval': current_time - self.last_aggregation,
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histograms': {},
                'metadata': self.metric_metadata.copy()
            }
            
            # Aggregate histograms
            for key, values in self.histograms.items():
                if values:
                    aggregated['histograms'][key] = {
                        'count': len(values),
                        'sum': sum(values),
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'p50': self._calculate_percentile(list(values), 50),
                        'p95': self._calculate_percentile(list(values), 95),
                        'p99': self._calculate_percentile(list(values), 99)
                    }
            
            self.aggregated_metrics = aggregated
            self.last_aggregation = current_time
            
            return aggregated
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.aggregate_metrics()
    
    def reset(self):
        """Reset all metrics"""
        with self.lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.summaries.clear()
            self.metric_metadata.clear()
            self.aggregated_metrics = {}


class DistributedTracer:
    """
    Distributed tracing system with OpenTelemetry support.
    """
    
    def __init__(self,
                 service_name: str = "sum-platform",
                 enable_export: bool = True,
                 otlp_endpoint: Optional[str] = None):
        """
        Initialize distributed tracer.
        
        Args:
            service_name: Name of the service
            enable_export: Enable trace export
            otlp_endpoint: OTLP collector endpoint
        """
        self.service_name = service_name
        self.enable_export = enable_export
        self.otlp_endpoint = otlp_endpoint or "localhost:4317"
        
        # Trace storage
        self.traces: Dict[str, List[TraceSpan]] = defaultdict(list)
        self.active_spans: Dict[str, TraceSpan] = {}
        
        # Context management
        self.context = TraceContext()
        
        # Metrics
        self.metrics = MetricsCollector()
        
        # Initialize OpenTelemetry if available
        if OPENTELEMETRY_AVAILABLE and enable_export:
            self._init_opentelemetry()
        else:
            self.tracer = None
    
    def _init_opentelemetry(self):
        """Initialize OpenTelemetry"""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "1.0.0"
            })
            
            # Initialize tracer
            provider = TracerProvider(resource=resource)
            
            if self.enable_export:
                # Add OTLP exporter
                exporter = OTLPSpanExporter(
                    endpoint=self.otlp_endpoint,
                    insecure=True
                )
                processor = BatchSpanProcessor(exporter)
                provider.add_span_processor(processor)
            
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(__name__)
            
            logger.info(f"OpenTelemetry initialized with endpoint {self.otlp_endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self.tracer = None
    
    @contextmanager
    def start_span(self,
                  operation_name: str,
                  tags: Optional[Dict[str, Any]] = None,
                  child_of: Optional[TraceSpan] = None):
        """
        Start a new span.
        
        Args:
            operation_name: Name of the operation
            tags: Tags to add to span
            child_of: Parent span
        
        Yields:
            TraceSpan object
        """
        # Use OpenTelemetry if available
        if self.tracer:
            with self.tracer.start_as_current_span(operation_name) as otel_span:
                if tags:
                    for key, value in tags.items():
                        otel_span.set_attribute(key, str(value))
                
                # Create fallback span for metrics
                span = self._create_span(operation_name, tags, child_of)
                
                try:
                    yield span
                    span.finish()
                except Exception as e:
                    span.finish(error=e)
                    otel_span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                finally:
                    self._finish_span(span)
        
        else:
            # Fallback implementation
            span = self._create_span(operation_name, tags, child_of)
            
            try:
                yield span
                span.finish()
            except Exception as e:
                span.finish(error=e)
                raise
            finally:
                self._finish_span(span)
    
    def _create_span(self,
                    operation_name: str,
                    tags: Optional[Dict[str, Any]] = None,
                    child_of: Optional[TraceSpan] = None) -> TraceSpan:
        """Create a new span"""
        # Determine trace ID
        if child_of:
            trace_id = child_of.trace_id
        elif self.context.current_trace:
            trace_id = self.context.current_trace
        else:
            trace_id = uuid.uuid4().hex
            self.context.current_trace = trace_id
        
        # Determine parent span
        parent_span_id = None
        if child_of:
            parent_span_id = child_of.span_id
        elif self.context.current_span:
            parent_span_id = self.context.current_span.span_id
        
        # Create span
        span = TraceSpan(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            tags=tags or {}
        )
        
        # Add to active spans
        self.active_spans[span.span_id] = span
        
        # Push to context
        self.context.push_span(span)
        
        # Record metric
        self.metrics.increment_counter(
            "traces.spans.started",
            tags={'operation': operation_name}
        )
        
        return span
    
    def _finish_span(self, span: TraceSpan):
        """Finish a span"""
        # Pop from context
        self.context.pop_span()
        
        # Remove from active spans
        if span.span_id in self.active_spans:
            del self.active_spans[span.span_id]
        
        # Add to completed traces
        self.traces[span.trace_id].append(span)
        
        # Record metrics
        self.metrics.increment_counter(
            "traces.spans.finished",
            tags={'operation': span.operation_name, 'status': span.status}
        )
        
        if span.duration:
            self.metrics.record_histogram(
                "traces.spans.duration",
                span.duration,
                tags={'operation': span.operation_name}
            )
    
    def inject_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Inject trace context into headers for propagation.
        
        Args:
            headers: Existing headers
        
        Returns:
            Headers with trace context
        """
        if self.context.current_trace:
            headers['X-Trace-Id'] = self.context.current_trace
        
        if self.context.current_span:
            headers['X-Span-Id'] = self.context.current_span.span_id
            headers['X-Parent-Span-Id'] = self.context.current_span.parent_span_id or ''
        
        return headers
    
    def extract_headers(self, headers: Dict[str, str]):
        """
        Extract trace context from headers.
        
        Args:
            headers: Request headers
        """
        trace_id = headers.get('X-Trace-Id')
        if trace_id:
            self.context.current_trace = trace_id
        
        parent_span_id = headers.get('X-Parent-Span-Id')
        if parent_span_id:
            # Create continuation span
            span = TraceSpan(
                trace_id=trace_id or uuid.uuid4().hex,
                parent_span_id=parent_span_id
            )
            self.context.current_span = span
    
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace"""
        return self.traces.get(trace_id, [])
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        metrics = self.metrics.get_metrics()
        
        # Add trace-specific metrics
        metrics['traces'] = {
            'total_traces': len(self.traces),
            'active_spans': len(self.active_spans),
            'total_spans': sum(len(spans) for spans in self.traces.values())
        }
        
        return metrics


class MonitoringDashboard:
    """
    Real-time monitoring dashboard data provider.
    """
    
    def __init__(self,
                 tracer: DistributedTracer,
                 update_interval: float = 5.0):
        """
        Initialize monitoring dashboard.
        
        Args:
            tracer: Distributed tracer instance
            update_interval: Dashboard update interval
        """
        self.tracer = tracer
        self.update_interval = update_interval
        
        # Dashboard data
        self.dashboard_data = {
            'system': {},
            'application': {},
            'traces': {},
            'errors': [],
            'alerts': []
        }
        
        # Update thread
        self.update_thread = None
        self.running = False
    
    def start(self):
        """Start dashboard updates"""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def stop(self):
        """Stop dashboard updates"""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
    
    def _update_loop(self):
        """Update dashboard data periodically"""
        while self.running:
            try:
                self._update_dashboard()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Dashboard update failed: {e}")
    
    def _update_dashboard(self):
        """Update dashboard data"""
        import psutil
        
        # System metrics
        self.dashboard_data['system'] = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            },
            'network': {
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv
            }
        }
        
        # Application metrics
        metrics = self.tracer.get_metrics_summary()
        self.dashboard_data['application'] = metrics
        
        # Recent traces
        recent_traces = []
        for trace_id, spans in list(self.tracer.traces.items())[-10:]:
            trace_summary = {
                'trace_id': trace_id,
                'span_count': len(spans),
                'duration': max(s.duration or 0 for s in spans),
                'errors': sum(1 for s in spans if s.status == "ERROR")
            }
            recent_traces.append(trace_summary)
        
        self.dashboard_data['traces'] = {
            'recent': recent_traces,
            'active_count': len(self.tracer.active_spans)
        }
        
        # Error tracking
        errors = []
        for trace_id, spans in self.tracer.traces.items():
            for span in spans:
                if span.status == "ERROR":
                    errors.append({
                        'trace_id': trace_id,
                        'span_id': span.span_id,
                        'operation': span.operation_name,
                        'error': span.error,
                        'timestamp': span.end_time
                    })
        
        self.dashboard_data['errors'] = errors[-50:]  # Keep last 50 errors
        
        # Generate alerts
        self._generate_alerts()
    
    def _generate_alerts(self):
        """Generate alerts based on thresholds"""
        alerts = []
        
        # High CPU alert
        if self.dashboard_data['system']['cpu_percent'] > 80:
            alerts.append({
                'level': 'WARNING',
                'type': 'HIGH_CPU',
                'message': f"CPU usage is {self.dashboard_data['system']['cpu_percent']}%",
                'timestamp': time.time()
            })
        
        # High memory alert
        if self.dashboard_data['system']['memory']['percent'] > 80:
            alerts.append({
                'level': 'WARNING',
                'type': 'HIGH_MEMORY',
                'message': f"Memory usage is {self.dashboard_data['system']['memory']['percent']}%",
                'timestamp': time.time()
            })
        
        # High error rate alert
        if self.dashboard_data['errors']:
            recent_errors = [e for e in self.dashboard_data['errors'] 
                           if e['timestamp'] > time.time() - 60]
            if len(recent_errors) > 10:
                alerts.append({
                    'level': 'ERROR',
                    'type': 'HIGH_ERROR_RATE',
                    'message': f"{len(recent_errors)} errors in last minute",
                    'timestamp': time.time()
                })
        
        self.dashboard_data['alerts'] = alerts
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data.copy()


# Global tracer instance
tracer = DistributedTracer()

# Global monitoring dashboard
dashboard = MonitoringDashboard(tracer)


# Decorator for tracing functions
def trace(operation_name: Optional[str] = None, **tags):
    """
    Decorator to trace function execution.
    
    Args:
        operation_name: Name of operation (defaults to function name)
        **tags: Tags to add to span
    """
    def decorator(func):
        op_name = operation_name or func.__name__
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tracer.start_span(op_name, tags=tags) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.add_log(f"Error in {op_name}: {str(e)}", level="ERROR")
                    raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.start_span(op_name, tags=tags) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.add_log(f"Error in {op_name}: {str(e)}", level="ERROR")
                    raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Example usage
if __name__ == "__main__":
    # Start monitoring dashboard
    dashboard.start()
    
    # Example traced function
    @trace(component="example")
    def process_document(doc_id: str):
        """Example function with tracing"""
        with tracer.start_span("fetch_document") as span:
            span.add_tag("doc_id", doc_id)
            time.sleep(0.1)  # Simulate work
        
        with tracer.start_span("analyze_document") as span:
            time.sleep(0.2)  # Simulate work
            span.add_log("Analysis complete")
        
        return {"doc_id": doc_id, "status": "processed"}
    
    # Process some documents
    for i in range(5):
        try:
            result = process_document(f"doc_{i}")
            print(f"Processed: {result}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Get metrics
    print("\nMetrics Summary:")
    print(json.dumps(tracer.get_metrics_summary(), indent=2))
    
    # Get dashboard data
    print("\nDashboard Data:")
    print(json.dumps(dashboard.get_dashboard_data(), indent=2))
    
    # Stop dashboard
    dashboard.stop()
