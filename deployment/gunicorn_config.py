"""
Gunicorn configuration for production deployment
"""
import multiprocessing
import os

# Server socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '5001')}"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'  # Use 'gevent' for async
worker_connections = 1000
timeout = 300  # 5 minutes for long operations
keepalive = 2

# Threading
threads = 4
thread = threads

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Preload app for better memory usage (but disable for development)
preload_app = os.getenv('ENV', 'production') == 'production'

# Server mechanics
daemon = False
pidfile = '/tmp/sum-gunicorn.pid'
user = None
group = None
tmp_upload_dir = None

# Logging
accesslog = '-'  # Log to stdout
errorlog = '-'   # Log to stderr
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'sum-platform'

# Server hooks
def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting SUM Platform server...")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading SUM Platform server...")

def when_ready(server):
    """Called just after the server is started."""
    server.log.info("SUM Platform server is ready. Listening at: %s", server.address)

def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT."""
    worker.log.info("Worker interrupted: %s", worker.pid)

def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info("Forking worker: %s", worker.pid)

def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info("Worker spawned: %s", worker.pid)

def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Forking new master process...")

def pre_request(worker, req):
    """Called just before a worker processes the request."""
    worker.log.debug("%s %s", req.method, req.path)

def post_request(worker, req, environ, resp):
    """Called after a worker processes the request."""
    worker.log.debug("%s %s - %s", req.method, req.path, resp.status)

def child_exit(server, worker):
    """Called just after a worker has been exited."""
    server.log.info("Worker exited: %s", worker.pid)

def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    server.log.info("Worker cleanup: %s", worker.pid)

def nworkers_changed(server, new_value, old_value):
    """Called when number of workers changes."""
    server.log.info("Number of workers changed from %s to %s", old_value, new_value)

def on_exit(server):
    """Called just before exiting."""
    server.log.info("Shutting down SUM Platform server...")

# Environment-specific overrides
if os.getenv('ENV') == 'development':
    reload = True
    workers = 2
    loglevel = 'debug'