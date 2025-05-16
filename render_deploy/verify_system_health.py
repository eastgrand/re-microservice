#!/usr/bin/env python3
"""
Comprehensive System Verification for SHAP Microservice
- Verifies Redis connection and patches
- Checks memory optimization settings
- Validates worker configuration
- Tests the job queue functionality
"""

import os
import sys
import logging
import time
import json
import argparse
from datetime import datetime
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("system-verification")

# Set up argument parser
parser = argparse.ArgumentParser(description='Verify SHAP microservice system health')
parser.add_argument('--check-web', action='store_true', help='Run web service health checks')
parser.add_argument('--check-worker', action='store_true', help='Run worker health checks')
parser.add_argument('--check-redis', action='store_true', help='Run Redis connection checks')
parser.add_argument('--check-memory', action='store_true', help='Run memory optimization checks')
parser.add_argument('--check-job', action='store_true', help='Try enqueueing and processing a test job')
parser.add_argument('--all', action='store_true', help='Run all checks')
parser.add_argument('--service-url', help='URL for the web service', default=os.environ.get('SHAP_SERVICE_URL', 'http://localhost:10000'))
parser.add_argument('--api-key', help='API key for the service', default=os.environ.get('API_KEY', ''))

def check_system_info():
    """Get system information"""
    logger.info("=== System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    # Check memory
    try:
        import psutil
        vm = psutil.virtual_memory()
        logger.info(f"Total memory: {vm.total / (1024**2):.1f} MB")
        logger.info(f"Available memory: {vm.available / (1024**2):.1f} MB")
        logger.info(f"Memory used: {vm.percent}%")
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss / (1024**2)
        logger.info(f"Process memory usage: {process_memory:.1f} MB")
    except ImportError:
        logger.warning("psutil not available, skipping detailed memory info")
    
    # Check environment variables
    logger.info("=== Environment Variables ===")
    env_vars = [
        'AGGRESSIVE_MEMORY_MANAGEMENT', 'SHAP_BATCH_SIZE', 'SHAP_MAX_BATCH_SIZE',
        'REDIS_URL', 'REDIS_HEALTH_CHECK_INTERVAL', 'PORT'
    ]
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        # Mask Redis URL for security
        if var == 'REDIS_URL' and value != 'Not set':
            parts = value.split('@')
            if len(parts) > 1:
                masked = f"{parts[0].split(':')[0]}:***@{parts[1]}"
                logger.info(f"{var}: {masked}")
            else:
                logger.info(f"{var}: [URL present but masked for security]")
        else:
            logger.info(f"{var}: {value}")
    
    # Check for required packages
    logger.info("=== Required Packages ===")
    packages = ['redis', 'rq', 'flask', 'pandas', 'numpy', 'shap', 'xgboost']
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            logger.info(f"{package}: Installed (version {version})")
        except ImportError:
            logger.error(f"{package}: Not installed")

def verify_redis_connection():
    """Verify Redis connection settings and functionality"""
    logger.info("=== Redis Connection Verification ===")
    try:
        # Import redis
        import redis
        import redis_connection_patch
        
        # Get Redis URL from environment or use default
        redis_url = os.environ.get('REDIS_URL')
        if not redis_url:
            logger.error("REDIS_URL is not set in the environment")
            return False
            
        # Apply patches
        logger.info("Applying Redis connection patches...")
        redis_connection_patch.apply_all_patches()
        
        # Create Redis connection
        logger.info("Creating Redis connection...")
        start_time = time.time()
        redis_conn = redis.from_url(
            redis_url,
            socket_timeout=10,
            socket_connect_timeout=10,
            socket_keepalive=True,
            health_check_interval=30,
            retry_on_timeout=True
        )
        conn_time = time.time() - start_time
        logger.info(f"Connection established in {conn_time:.2f} seconds")
        
        # Verify connection with PING
        logger.info("Testing connection with PING...")
        start_time = time.time()
        ping_result = redis_conn.ping()
        ping_time = time.time() - start_time
        logger.info(f"PING successful: {ping_result} (took {ping_time:.2f}s)")
        
        # Check Redis memory usage
        try:
            info = redis_conn.info('memory')
            used_memory_mb = info['used_memory'] / (1024 * 1024)
            peak_memory_mb = info['used_memory_peak'] / (1024 * 1024)
            logger.info(f"Redis memory usage: {used_memory_mb:.2f} MB (peak: {peak_memory_mb:.2f} MB)")
            
            # Check if memory usage is concerning
            if used_memory_mb > 400:  # Assuming 512MB limit
                logger.warning(f"Redis memory usage is high: {used_memory_mb:.2f} MB")
            
        except Exception as e:
            logger.warning(f"Could not get Redis memory info: {str(e)}")
        
        # Check queue status
        try:
            from rq import Queue
            queue = Queue('shap-jobs', connection=redis_conn)
            queue_length = len(queue)
            logger.info(f"Job queue length: {queue_length}")
            
            # Check for started jobs that might be stuck
            started_registry = queue.started_job_registry
            started_jobs = started_registry.get_job_ids()
            logger.info(f"Started jobs: {len(started_jobs)}")
            if started_jobs:
                logger.warning(f"There are {len(started_jobs)} jobs in 'started' state that might be stuck")
                
            # Check for failed jobs
            failed_jobs = queue.failed_job_registry.get_job_ids()
            logger.info(f"Failed jobs: {len(failed_jobs)}")
            if failed_jobs:
                logger.warning(f"There are {len(failed_jobs)} failed jobs")
                
        except Exception as e:
            logger.warning(f"Could not check job queue status: {str(e)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Redis verification failed: {str(e)}", exc_info=True)
        return False

def verify_memory_settings():
    """Verify memory optimization settings"""
    logger.info("=== Memory Optimization Verification ===")
    try:
        # Import memory optimization module
        from optimize_memory import AGGRESSIVE_MEMORY, DEFAULT_MAX_MEMORY_MB, get_memory_usage
        
        # Check current settings
        logger.info(f"Aggressive memory management: {AGGRESSIVE_MEMORY}")
        logger.info(f"Max memory threshold: {DEFAULT_MAX_MEMORY_MB} MB")
        
        # Get current memory usage
        current_usage = get_memory_usage()
        logger.info(f"Current memory usage: {current_usage:.2f} MB")
        
        # Check if current usage is below threshold
        if current_usage > DEFAULT_MAX_MEMORY_MB:
            logger.warning(f"Current memory usage ({current_usage:.2f} MB) exceeds threshold ({DEFAULT_MAX_MEMORY_MB} MB)")
        else:
            logger.info(f"Memory usage is within threshold limits")
            
        # Test garbage collection
        gc.collect()
        after_gc = get_memory_usage()
        logger.info(f"Memory after garbage collection: {after_gc:.2f} MB (freed {current_usage - after_gc:.2f} MB)")
        
        return True
        
    except Exception as e:
        logger.error(f"Memory settings verification failed: {str(e)}", exc_info=True)
        return False

def verify_worker_setup():
    """Verify worker setup"""
    logger.info("=== Worker Setup Verification ===")
    try:
        # Check if worker process files exist
        files_to_check = ['simple_worker.py', 'setup_worker.py']
        for file in files_to_check:
            if os.path.exists(file):
                logger.info(f"Worker file found: {file}")
                # Check if it's executable
                if os.access(file, os.X_OK):
                    logger.info(f"{file} is executable")
                else:
                    logger.warning(f"{file} is not executable. Fix with: chmod +x {file}")
            else:
                logger.error(f"Worker file not found: {file}")
        
        # Check if worker patches exist
        try:
            import worker_process_fix
            logger.info("Worker process patches module exists")
        except ImportError:
            logger.warning("Worker process patches module not found")
        
        # Check if rendering config exists
        if os.path.exists('render.yaml'):
            logger.info("render.yaml exists")
            
            # Quick check of render.yaml worker configuration
            with open('render.yaml', 'r') as f:
                content = f.read()
                if 'simple_worker.py' in content:
                    logger.info("render.yaml appears to be using simple_worker.py")
                else:
                    logger.warning("render.yaml might not be using simple_worker.py")
                    
                if 'AGGRESSIVE_MEMORY_MANAGEMENT' in content and 'false' in content:
                    logger.info("Aggressive memory management appears to be disabled in render.yaml")
        else:
            logger.warning("render.yaml not found")
            
        return True
        
    except Exception as e:
        logger.error(f"Worker setup verification failed: {str(e)}", exc_info=True)
        return False

def test_job_queue():
    """Test the job queue by enqueueing and processing a test job"""
    logger.info("=== Job Queue Test ===")
    try:
        # Import required modules
        import redis
        from rq import Queue
        import json
        
        # Get Redis URL
        redis_url = os.environ.get('REDIS_URL')
        if not redis_url:
            logger.error("REDIS_URL is not set")
            return False
            
        # Connect to Redis
        redis_conn = redis.from_url(redis_url)
        
        # Create queue
        queue = Queue('shap-jobs', connection=redis_conn)
        
        # Define a simple test job function
        def test_job(job_id):
            time.sleep(1)  # Simulate some processing time
            return {
                'job_id': job_id,
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'result': 'Test job completed successfully'
            }
        
        # Enqueue the job
        job_id = f"test_job_{int(time.time())}"
        logger.info(f"Enqueueing test job with ID: {job_id}")
        job = queue.enqueue(test_job, job_id)
        
        if not job:
            logger.error("Failed to enqueue job")
            return False
            
        logger.info(f"Job enqueued with ID: {job.id}")
        
        # Wait for job to complete
        max_wait = 30  # seconds
        start_time = time.time()
        while not job.is_finished and time.time() - start_time < max_wait:
            time.sleep(1)
            job.refresh()
            logger.info(f"Job status: {job.get_status()}")
            
        if job.is_finished:
            result = job.result
            logger.info(f"Job completed with result: {result}")
            return True
        else:
            logger.error(f"Job did not complete within {max_wait} seconds")
            return False
            
    except Exception as e:
        logger.error(f"Job queue test failed: {str(e)}", exc_info=True)
        return False

def check_web_service(service_url, api_key=None):
    """Check the web service health endpoints"""
    logger.info("=== Web Service Health Check ===")
    try:
        import requests
        
        headers = {"X-API-KEY": api_key} if api_key else {}
        
        # Check basic ping endpoint
        logger.info(f"Checking ping endpoint at {service_url}/ping")
        try:
            response = requests.get(f"{service_url}/ping", timeout=10)
            if response.status_code == 200:
                logger.info("Ping endpoint: OK")
            else:
                logger.warning(f"Ping endpoint returned status code {response.status_code}")
        except Exception as e:
            logger.error(f"Error checking ping endpoint: {str(e)}")
            
        # Check health endpoint
        logger.info("Checking health endpoint")
        try:
            response = requests.get(f"{service_url}/health", headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Health endpoint returned: {json.dumps(data, indent=2)}")
                if data.get("status") == "healthy":
                    logger.info("Health status: healthy")
                else:
                    logger.warning(f"Health status: {data.get('status')}")
            else:
                logger.warning(f"Health endpoint returned status code {response.status_code}")
        except Exception as e:
            logger.error(f"Error checking health endpoint: {str(e)}")
            
        # Check Redis ping endpoint
        logger.info("Checking Redis ping endpoint")
        try:
            response = requests.get(f"{service_url}/admin/redis_ping", headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Redis ping response time: {data.get('response_time_ms')} ms")
            else:
                logger.warning(f"Redis ping endpoint returned status code {response.status_code}")
        except Exception as e:
            logger.error(f"Error checking Redis ping endpoint: {str(e)}")
            
        return True
            
    except Exception as e:
        logger.error(f"Web service health check failed: {str(e)}", exc_info=True)
        return False

def main():
    """Main function to run all verification checks"""
    args = parser.parse_args()
    
    # If no specific checks are requested, run all checks
    if not any([args.check_web, args.check_worker, args.check_redis, 
                args.check_memory, args.check_job]):
        args.all = True
    
    logger.info("=" * 60)
    logger.info(f"SHAP Microservice System Verification - {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    # Always check system info
    check_system_info()
    
    results = {}
    
    # Run requested checks
    if args.all or args.check_redis:
        results['redis'] = verify_redis_connection()
        
    if args.all or args.check_memory:
        results['memory'] = verify_memory_settings()
        
    if args.all or args.check_worker:
        results['worker'] = verify_worker_setup()
        
    if args.all or args.check_job:
        results['job_queue'] = test_job_queue()
        
    if args.all or args.check_web:
        results['web_service'] = check_web_service(args.service_url, args.api_key)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Verification Summary:")
    all_passed = True
    for check, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"  {check.upper()}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("✅ All checks passed! System appears to be healthy.")
        return 0
    else:
        logger.warning("⚠️ Some checks failed. Please review the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
