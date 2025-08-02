#!/usr/bin/env python3
"""
monitoring_dashboard.py - Real-time Monitoring Dashboard for SUM

Provides comprehensive monitoring of system performance, security events,
and operational metrics with real-time updates.

Author: SUM Development Team
License: Apache License 2.0
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging
import psutil
from collections import defaultdict, deque

# Flask for web dashboard
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit

# SUM components
from performance_optimizer import PerformanceMonitor, MemoryOptimizer
from security_utils import SecurityMonitor, RateLimiter
from error_handling import ErrorHandler

logger = logging.getLogger(__name__)


@dataclass
class SystemHealth:
    """System health metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    active_connections: int
    response_time_ms: float
    error_rate: float
    uptime_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'disk_percent': self.disk_percent,
            'active_connections': self.active_connections,
            'response_time_ms': self.response_time_ms,
            'error_rate': self.error_rate,
            'uptime_seconds': self.uptime_seconds,
            'health_status': self._get_health_status()
        }
    
    def _get_health_status(self) -> str:
        """Determine overall health status."""
        if self.cpu_percent > 90 or self.memory_percent > 95 or self.error_rate > 0.1:
            return "critical"
        elif self.cpu_percent > 70 or self.memory_percent > 80 or self.error_rate > 0.05:
            return "warning"
        else:
            return "healthy"


class MonitoringCollector:
    """Collect metrics from various SUM components."""
    
    def __init__(self):
        self.start_time = time.time()
        self.performance_monitor = PerformanceMonitor("SystemMonitor")
        self.security_monitor = SecurityMonitor()
        self.rate_limiter = RateLimiter()
        self.error_handler = ErrorHandler("MonitoringCollector")
        
        # Metrics storage
        self.health_history: deque = deque(maxlen=1000)
        self.request_metrics: deque = deque(maxlen=5000)
        self.error_counts = defaultdict(int)
        
        # Real-time counters
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
    
    def collect_system_health(self) -> SystemHealth:
        """Collect current system health metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network connections
        connections = len(psutil.net_connections())
        
        # Calculate average response time
        recent_requests = list(self.request_metrics)[-100:]  # Last 100 requests
        avg_response_time = sum(r['response_time'] for r in recent_requests) / len(recent_requests) if recent_requests else 0
        
        # Calculate error rate
        recent_errors = sum(1 for r in recent_requests if not r['success']) if recent_requests else 0
        error_rate = recent_errors / len(recent_requests) if recent_requests else 0
        
        # Uptime
        uptime = time.time() - self.start_time
        
        health = SystemHealth(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_percent=disk.percent,
            active_connections=connections,
            response_time_ms=avg_response_time * 1000,
            error_rate=error_rate,
            uptime_seconds=uptime
        )
        
        self.health_history.append(health)
        return health
    
    def record_request(self, endpoint: str, method: str, response_time: float, 
                      success: bool, status_code: int = 200):
        """Record API request metrics."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_processing_time += response_time
        
        request_data = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'method': method,
            'response_time': response_time,
            'success': success,
            'status_code': status_code
        }
        
        self.request_metrics.append(request_data)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        current_health = self.collect_system_health()
        
        # Recent health data for charts
        recent_health = list(self.health_history)[-50:]  # Last 50 data points
        
        # Performance summary
        perf_summary = self.performance_monitor.get_summary()
        
        # Security summary
        security_summary = self.security_monitor.get_threat_summary(hours=24)
        
        # Rate limiter stats
        rate_limit_stats = self.rate_limiter.get_stats()
        
        # Top endpoints by requests
        endpoint_counts = defaultdict(int)
        endpoint_avg_times = defaultdict(list)
        
        for req in list(self.request_metrics)[-1000:]:  # Last 1000 requests
            endpoint = req['endpoint']
            endpoint_counts[endpoint] += 1
            endpoint_avg_times[endpoint].append(req['response_time'])
        
        top_endpoints = [
            {
                'endpoint': endpoint,
                'requests': count,
                'avg_time_ms': sum(endpoint_avg_times[endpoint]) / len(endpoint_avg_times[endpoint]) * 1000
            }
            for endpoint, count in sorted(endpoint_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        return {
            'current_health': current_health.to_dict(),
            'health_history': [h.to_dict() for h in recent_health],
            'performance_summary': perf_summary,
            'security_summary': security_summary,
            'rate_limit_stats': rate_limit_stats,
            'request_stats': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
                'avg_response_time_ms': (self.total_processing_time / self.total_requests * 1000) if self.total_requests > 0 else 0
            },
            'top_endpoints': top_endpoints,
            'timestamp': datetime.now().isoformat()
        }


# Global monitoring instance
monitoring_collector = MonitoringCollector()

# Flask app for dashboard
app = Flask(__name__)
app.config['SECRET_KEY'] = 'monitoring_dashboard_secret'
socketio = SocketIO(app, cors_allowed_origins="*")


# Dashboard HTML template
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SUM Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.0/socket.io.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; margin: 10px 0; }
        .healthy { color: #27ae60; }
        .warning { color: #f39c12; }
        .critical { color: #e74c3c; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
        .refresh-indicator { position: fixed; top: 20px; right: 20px; padding: 10px; background: #3498db; color: white; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 SUM System Monitoring Dashboard</h1>
        <p>Real-time monitoring of performance, security, and health metrics</p>
        <div id="refreshIndicator" class="refresh-indicator" style="display: none;">Updating...</div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <h3>System Health</h3>
            <div id="healthStatus" class="metric-value healthy">
                <span class="status-indicator" style="background: #27ae60;"></span>
                Healthy
            </div>
            <div>CPU: <span id="cpuPercent">0</span>%</div>
            <div>Memory: <span id="memoryPercent">0</span>%</div>
            <div>Disk: <span id="diskPercent">0</span>%</div>
        </div>

        <div class="metric-card">
            <h3>Request Statistics</h3>
            <div class="metric-value" id="totalRequests">0</div>
            <div>Success Rate: <span id="successRate">0</span>%</div>
            <div>Avg Response: <span id="avgResponseTime">0</span>ms</div>
            <div>Failed: <span id="failedRequests">0</span></div>
        </div>

        <div class="metric-card">
            <h3>Security Events</h3>
            <div class="metric-value" id="securityEvents">0</div>
            <div>Blocked IPs: <span id="blockedIPs">0</span></div>
            <div>Rate Limited: <span id="rateLimited">0</span></div>
            <div>Threats: <span id="threats">0</span></div>
        </div>

        <div class="metric-card">
            <h3>Performance</h3>
            <div class="metric-value" id="uptime">0h 0m</div>
            <div>Active Connections: <span id="activeConnections">0</span></div>
            <div>Memory Usage: <span id="memoryUsage">0</span>MB</div>
            <div>Cache Hit Rate: <span id="cacheHitRate">0</span>%</div>
        </div>
    </div>

    <div class="chart-container">
        <h3>System Health Trends</h3>
        <canvas id="healthChart" width="400" height="200"></canvas>
    </div>

    <div class="chart-container">
        <h3>Response Time Trends</h3>
        <canvas id="responseChart" width="400" height="200"></canvas>
    </div>

    <div class="chart-container">
        <h3>Top Endpoints</h3>
        <table id="endpointsTable">
            <thead>
                <tr>
                    <th>Endpoint</th>
                    <th>Requests</th>
                    <th>Avg Response (ms)</th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>

    <script>
        const socket = io();
        
        // Chart configurations
        const healthChart = new Chart(document.getElementById('healthChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'CPU %',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Memory %',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true, max: 100 }
                }
            }
        });

        const responseChart = new Chart(document.getElementById('responseChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Response Time (ms)',
                    data: [],
                    borderColor: '#27ae60',
                    backgroundColor: 'rgba(39, 174, 96, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });

        // Update dashboard with new data
        function updateDashboard(data) {
            const health = data.current_health;
            const stats = data.request_stats;
            const security = data.security_summary;
            
            // System health
            document.getElementById('healthStatus').innerHTML = 
                `<span class="status-indicator" style="background: ${getHealthColor(health.health_status)};"></span>${health.health_status.toUpperCase()}`;
            document.getElementById('healthStatus').className = `metric-value ${health.health_status}`;
            document.getElementById('cpuPercent').textContent = health.cpu_percent.toFixed(1);
            document.getElementById('memoryPercent').textContent = health.memory_percent.toFixed(1);
            document.getElementById('diskPercent').textContent = health.disk_percent.toFixed(1);
            
            // Request stats
            document.getElementById('totalRequests').textContent = stats.total_requests.toLocaleString();
            document.getElementById('successRate').textContent = (stats.success_rate * 100).toFixed(1);
            document.getElementById('avgResponseTime').textContent = stats.avg_response_time_ms.toFixed(1);
            document.getElementById('failedRequests').textContent = stats.failed_requests.toLocaleString();
            
            // Security events
            document.getElementById('securityEvents').textContent = security.total_events.toLocaleString();
            document.getElementById('blockedIPs').textContent = data.rate_limit_stats.blocked_ips || 0;
            document.getElementById('rateLimited').textContent = security.by_severity?.warning || 0;
            document.getElementById('threats').textContent = security.by_severity?.critical || 0;
            
            // Performance
            const uptimeHours = Math.floor(health.uptime_seconds / 3600);
            const uptimeMinutes = Math.floor((health.uptime_seconds % 3600) / 60);
            document.getElementById('uptime').textContent = `${uptimeHours}h ${uptimeMinutes}m`;
            document.getElementById('activeConnections').textContent = health.active_connections;
            document.getElementById('memoryUsage').textContent = (health.memory_percent * 16).toFixed(0); // Assuming 16GB total
            document.getElementById('cacheHitRate').textContent = '85'; // Placeholder
            
            // Update charts
            updateHealthChart(data.health_history);
            updateResponseChart(data.health_history);
            updateEndpointsTable(data.top_endpoints);
        }

        function getHealthColor(status) {
            switch(status) {
                case 'healthy': return '#27ae60';
                case 'warning': return '#f39c12';
                case 'critical': return '#e74c3c';
                default: return '#95a5a6';
            }
        }

        function updateHealthChart(history) {
            const labels = history.map(h => new Date(h.timestamp).toLocaleTimeString());
            const cpuData = history.map(h => h.cpu_percent);
            const memoryData = history.map(h => h.memory_percent);
            
            healthChart.data.labels = labels.slice(-20); // Last 20 points
            healthChart.data.datasets[0].data = cpuData.slice(-20);
            healthChart.data.datasets[1].data = memoryData.slice(-20);
            healthChart.update('none');
        }

        function updateResponseChart(history) {
            const labels = history.map(h => new Date(h.timestamp).toLocaleTimeString());
            const responseData = history.map(h => h.response_time_ms);
            
            responseChart.data.labels = labels.slice(-20);
            responseChart.data.datasets[0].data = responseData.slice(-20);
            responseChart.update('none');
        }

        function updateEndpointsTable(endpoints) {
            const tbody = document.querySelector('#endpointsTable tbody');
            tbody.innerHTML = endpoints.map(ep => `
                <tr>
                    <td>${ep.endpoint}</td>
                    <td>${ep.requests.toLocaleString()}</td>
                    <td>${ep.avg_time_ms.toFixed(1)}</td>
                </tr>
            `).join('');
        }

        // Socket event handlers
        socket.on('dashboard_update', function(data) {
            document.getElementById('refreshIndicator').style.display = 'block';
            updateDashboard(data);
            setTimeout(() => {
                document.getElementById('refreshIndicator').style.display = 'none';
            }, 500);
        });

        // Initial load
        fetch('/api/dashboard')
            .then(response => response.json())
            .then(data => updateDashboard(data));
    </script>
</body>
</html>
"""


@app.route('/')
def dashboard():
    """Serve the monitoring dashboard."""
    return render_template_string(DASHBOARD_TEMPLATE)


@app.route('/api/dashboard')
def dashboard_api():
    """API endpoint for dashboard data."""
    return jsonify(monitoring_collector.get_dashboard_data())


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    health = monitoring_collector.collect_system_health()
    return jsonify({
        'status': health._get_health_status(),
        'timestamp': health.timestamp.isoformat(),
        'metrics': health.to_dict()
    })


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    # Send initial data
    data = monitoring_collector.get_dashboard_data()
    emit('dashboard_update', data)


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")


async def broadcast_updates():
    """Broadcast real-time updates to connected clients."""
    while True:
        try:
            data = monitoring_collector.get_dashboard_data()
            socketio.emit('dashboard_update', data)
            await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Error broadcasting updates: {e}")
            await asyncio.sleep(10)


def start_monitoring_server(host: str = '0.0.0.0', port: int = 5000):
    """Start the monitoring dashboard server."""
    logger.info(f"Starting monitoring dashboard on {host}:{port}")
    
    # Start background update task
    asyncio.create_task(broadcast_updates())
    
    # Start Flask-SocketIO server
    socketio.run(app, host=host, port=port, debug=False)


# Example usage and testing
if __name__ == "__main__":
    print("SUM Monitoring Dashboard")
    print("=" * 40)
    
    # Simulate some metrics
    for i in range(10):
        monitoring_collector.record_request(
            endpoint=f"/api/test{i % 3}",
            method="POST",
            response_time=0.1 + (i * 0.05),
            success=i % 5 != 0,  # 20% failure rate
            status_code=200 if i % 5 != 0 else 500
        )
    
    # Test dashboard data
    dashboard_data = monitoring_collector.get_dashboard_data()
    print("Dashboard data keys:", list(dashboard_data.keys()))
    print("Current health:", dashboard_data['current_health']['health_status'])
    print("Total requests:", dashboard_data['request_stats']['total_requests'])
    
    # Start server
    print("\\nStarting monitoring dashboard server...")
    print("Visit http://localhost:5000 to view the dashboard")
    
    start_monitoring_server()