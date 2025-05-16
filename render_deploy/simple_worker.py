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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple-worker")

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
        
        # Create and start worker
        logger.info("Starting simple worker...")
        worker = Worker(['shap-jobs'], connection=conn, name=f"simple-worker-{os.getpid()}")
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
        
        # Start working (with exception handling)
        logger.info("Starting to process jobs...")
        try:
            worker.work(with_scheduler=True)
        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
            return 0
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")
            logger.error(traceback.format_exc())
            return 1
    
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
