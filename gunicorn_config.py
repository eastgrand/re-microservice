"""Gunicorn configuration for Render deployment.
This file configures Gunicorn to better handle memory constraints and timeouts.
"""

import os
import multiprocessing

# Basic configuration
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"
workers = 1  # Single worker for Render free tier
threads = 2  # Multiple threads to handle slow responses
worker_class = 'gthread'
timeout = 120  # Increased timeout to allow for slow model loading

# Make sure we respond quickly to health checks
# This helps Render.com detect that our service is running
backlog = 100  # Increased connection queue
worker_connections = 100  # Maximum number of simultaneous client connections
keepalive = 65  # How long to keep connections open

# Render has memory constraints, so configure for that
max_requests = 10
max_requests_jitter = 5
preload_app = True  # Preload the app to avoid loading the model multiple times
worker_tmp_dir = "/tmp"  # Ensure we can create worker temp directories

# Log configuration
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
accesslog = '-'
errorlog = '-'
capture_output = True

# Worker timeouts and reliability
graceful_timeout = 120
keepalive = 5

# For Render-specific environment
is_render = 'RENDER' in os.environ
if is_render:
    # Force more aggressive memory settings on Render
    print("Running on Render: using specialized worker settings")
    # Force aggressive garbage collection
    import gc
    gc.set_threshold(100, 5, 5)  # More aggressive than the default (700, 10, 10)
    
    # Define worker initialization and cleanup functions
    def on_starting(server):
        """Called just before the master process is initialized."""
        print("Gunicorn starting...")
    
    def worker_abort(worker):
        """Called when a worker receives SIGABRT signal."""
        print(f"Worker {worker.pid} aborted")
        
    def worker_int(worker):
        """Called when a worker receives SIGINT signal."""
        print(f"Worker {worker.pid} interrupted")
        
    def worker_exit(server, worker):
        """Called just after a worker has been exited."""
        print(f"Worker {worker.pid} exited")
        # Run garbage collection on worker exit
        gc.collect()
        
    def pre_fork(server, worker):
        """Called just before a worker is forked."""
        print(f"Pre-forking worker {worker.pid}")
        
    def post_fork(server, worker):
        """Called just after a worker has been forked."""
        print(f"Worker {worker.pid} forked")
        # Clear memory caches after fork
        gc.collect()
