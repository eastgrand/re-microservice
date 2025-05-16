#!/usr/bin/env python3
# filepath: /Users/voldeck/code/shap-microservice/simple_worker.py
"""
Simple RQ Worker for SHAP microservice

This is a simplified worker script that runs an RQ worker for the SHAP microservice.
It doesn't rely on the Connection context manager which appears to be causing issues.
"""

import os
import sys
import gc
import logging
import time
import traceback
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple-worker")

def cleanup_stale_workers(conn):
    """Clean up stale worker registrations from Redis"""
    try:
        logger.info("Checking for stale worker registrations...")
        existing_workers_key = 'rq:workers'
        worker_keys = conn.smembers(existing_workers_key)
        
        if not worker_keys:
            logger.info("No existing workers found.")
            return
            
        logger.info(f"Found {len(worker_keys)} worker registrations.")
        cleaned = 0
        
        for worker_key in worker_keys:
            try:
                # Check if the worker is still alive using its heartbeat
                heartbeat_key = f"{worker_key.decode()}:heartbeat"
                last_heartbeat = conn.get(heartbeat_key)
                
                # If no heartbeat or heartbeat is old (>60 seconds), clean up
                if not last_heartbeat or (time.time() - float(last_heartbeat.decode())) > 60:
                    logger.info(f"Cleaning up stale worker: {worker_key.decode()}")
                    
                    # Delete the worker key and remove from workers set
                    conn.delete(worker_key)
                    conn.srem(existing_workers_key, worker_key)
                    cleaned += 1
            except Exception as e:
                logger.warning(f"Error checking worker {worker_key}: {str(e)}")
                
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} stale worker registrations")
        else:
            logger.info("No stale workers found")
    except Exception as e:
        logger.warning(f"Error during stale worker cleanup: {str(e)}")

def main():
    """Run a simple RQ worker"""
    # Enable garbage collection
    gc.enable()
    logger.info(f"Garbage collection enabled with thresholds: {gc.get_threshold()}")
    
    # Get Redis URL from environment
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        logger.error("REDIS_URL environment variable not set")
        return 1
    
    try:
        # Import needed modules
        import redis
        from rq import Queue, Worker
        from rq.job import Job
        
        logger.info("Connecting to Redis...")
        
        # Connect to Redis with improved parameters
        conn = redis.from_url(
            redis_url,
            socket_timeout=10,
            socket_connect_timeout=10,
            socket_keepalive=True,
            health_check_interval=30,
            retry_on_timeout=True
        )
        
        # Test Redis connection
        try:
            start_time = time.time()
            conn.ping()
            ping_time = time.time() - start_time
            logger.info(f"Redis connection successful. Ping time: {round(ping_time * 1000, 2)} ms")
        except Exception as e:
            logger.error(f"Redis connection test failed: {str(e)}")
            logger.error("Will try to proceed anyway")
        
        # Repair any stuck jobs
        try:
            queue_name = 'shap-jobs'
            queue = Queue(queue_name, connection=conn)
            
            # Check for stuck jobs in started state
            started_registry_key = f'rq:started_job_registry:{queue_name}'
            started_job_ids = conn.smembers(started_registry_key)
            
            if started_job_ids:
                logger.info(f"Found {len(started_job_ids)} jobs in started registry. Attempting repair...")
                
                for job_id_bytes in started_job_ids:
                    try:
                        job_id = job_id_bytes.decode('utf-8')
                        job = Job.fetch(job_id, connection=conn)
                        
                        # Requeue the job
                        job.requeue()
                        logger.info(f"Requeued job {job_id}")
                    except Exception as e:
                        logger.error(f"Error requeuing job {job_id_bytes}: {str(e)}")
            
            logger.info("Job repair completed")
        except Exception as e:
            logger.error(f"Error repairing jobs: {str(e)}")
            logger.error("Will try to proceed anyway")
        
        # Clean up any stale workers in Redis
        cleanup_stale_workers(conn)
        
        # Create and start worker with unique name using timestamp
        # Use hostname, PID, timestamp and random UUID to ensure uniqueness
        hostname = os.environ.get('HOSTNAME', 'unknown')
        unique_id = f"{hostname}-{os.getpid()}-{int(time.time())}-{str(uuid.uuid4())[:8]}"
        worker_name = f"shap-worker-{unique_id}"
        
        logger.info("Starting simple worker...")
        worker = Worker(['shap-jobs'], connection=conn, name=worker_name)
        logger.info(f"Worker created with ID: {worker.name}")
        
        # Apply memory optimization if available
        try:
            from shap_memory_fix import apply_memory_patches
            apply_memory_patches()
            logger.info("Applied memory optimization patches")
            
            # Get batch size
            max_rows = int(os.environ.get('SHAP_MAX_BATCH_SIZE', '300'))
            logger.info(f"Using max batch size: {max_rows} rows")
        except ImportError:
            logger.warning("Could not import shap_memory_fix - continuing without memory optimizations")
        except Exception as e:
            logger.warning(f"Error applying memory optimizations: {str(e)}")
        
        # Apply Redis patches if available
        try:
            from redis_connection_patch import apply_all_patches
            apply_all_patches()
            logger.info("Applied Redis connection patches")
        except ImportError:
            logger.warning("Could not import redis_connection_patch - continuing without Redis patches")
        except Exception as e:
            logger.warning(f"Error applying Redis patches: {str(e)}")
        
        # Start working (with exception handling and retry logic)
        logger.info("Starting to process jobs...")
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    logger.info(f"Retry attempt {attempt}/{max_retries}...")
                    
                    # Generate a new unique name for the retry
                    unique_id = f"{hostname}-{os.getpid()}-{int(time.time())}-{str(uuid.uuid4())[:8]}"
                    worker_name = f"shap-worker-retry-{unique_id}"
                    worker = Worker(['shap-jobs'], connection=conn, name=worker_name)
                    logger.info(f"Created new worker with ID: {worker.name}")
                
                worker.work(with_scheduler=True)
                break  # If successful, exit the retry loop
                
            except ValueError as ve:
                if "There exists an active worker" in str(ve) and attempt < max_retries:
                    logger.warning(f"Worker name collision detected: {str(ve)}")
                    logger.info(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Worker error: {str(ve)}")
                    if attempt >= max_retries:
                        logger.error("Maximum retry attempts reached")
                        logger.error(traceback.format_exc())
                        return 1
            except KeyboardInterrupt:
                logger.info("Worker stopped by user")
                return 0
            except Exception as e:
                logger.error(f"Worker error: {str(e)}")
                logger.error(traceback.format_exc())
                if attempt >= max_retries:
                    return 1
                logger.info(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
    
    except ImportError as e:
        logger.error(f"Missing required module: {str(e)}")
        logger.error("Please install the required packages: pip install redis rq psutil")
        return 1
    except Exception as e:
        logger.error(f"Error in worker: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
