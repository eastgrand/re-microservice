#!/usr/bin/env python3
"""
Repair Stuck Jobs

This script detects and repairs stuck jobs in the SHAP microservice.
It will move jobs stuck in "started" state back to the queue.

Usage:
  python repair_stuck_jobs.py [--url REDIS_URL] [--force] [--timeout SECONDS]
"""

import os
import sys
import time
import argparse
import redis
import logging
from rq import Queue, Worker
from rq.job import Job
from rq.registry import StartedJobRegistry, FailedJobRegistry

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("repair-jobs")

def get_redis_connection(redis_url=None):
    """Get connection to Redis"""
    if not redis_url:
        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        
    logger.info(f"Connecting to Redis at: {redis_url[:20]}...")
    
    try:
        # Apply better connection parameters
        conn = redis.from_url(
            redis_url,
            socket_timeout=5,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30
        )
        conn.ping()  # Test connection
        logger.info("✅ Successfully connected to Redis")
        return conn
    except Exception as e:
        logger.error(f"❌ Could not connect to Redis: {str(e)}")
        return None

def repair_stuck_jobs(conn, force=False, timeout=300):
    """Repair jobs that are stuck in started state"""
    try:
        queue = Queue('shap-jobs', connection=conn)
        
        # Get started jobs
        started_registry = StartedJobRegistry(queue.name, queue.connection)
        started_job_ids = started_registry.get_job_ids()
        
        if not started_job_ids:
            logger.info("No jobs found in started registry.")
            return 0
            
        logger.info(f"Found {len(started_job_ids)} jobs in started registry.")
        
        # Get active workers
        workers = Worker.all(connection=conn)
        logger.info(f"Active workers: {len(workers)}")
        
        # Track active jobs
        active_jobs = set()
        for worker in workers:
            job = worker.get_current_job()
            if job:
                active_jobs.add(job.id)
                logger.info(f"Worker {worker.name} is processing job {job.id}")
        
        # Process each started job
        repaired_count = 0
        
        for job_id in started_job_ids:
            try:
                job = Job.fetch(job_id, connection=conn)
                
                # Skip if job is actively being processed by a worker
                if job_id in active_jobs and not force:
                    logger.info(f"Job {job_id} is currently being processed by a worker. Skipping.")
                    continue
                
                # Calculate how long the job has been in started state
                time_in_started = None
                
                if job.started_at:
                    current_time = time.time()
                    time_in_started = current_time - job.started_at
                    logger.info(f"Job {job_id} has been in started state for {time_in_started:.2f} seconds")
                
                # Check if job has been in started state too long or force is enabled
                if force or (time_in_started and time_in_started > timeout):
                    logger.warning(f"Repairing job {job_id}...")
                    
                    # Remove from started registry
                    started_registry.remove(job)
                    
                    # Requeue the job (this automatically adds to queue)
                    job.set_status('queued')
                    queue.enqueue_job(job)
                    
                    logger.info(f"✅ Job {job_id} moved from started back to queue")
                    repaired_count += 1
                else:
                    if time_in_started:
                        logger.info(f"Job {job_id} hasn't exceeded timeout ({timeout} seconds). Skipping.")
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {str(e)}")
        
        return repaired_count
    except Exception as e:
        logger.error(f"Error repairing jobs: {str(e)}")
        return 0

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Repair stuck jobs in SHAP microservice')
    parser.add_argument('--url', type=str, help='Redis URL')
    parser.add_argument('--force', action='store_true', help='Force repair all jobs in started state')
    parser.add_argument('--timeout', type=int, default=300, help='Time in seconds after which a started job is considered stuck')
    args = parser.parse_args()
    
    # Connect to Redis
    conn = get_redis_connection(args.url)
    if not conn:
        sys.exit(1)
    
    # Repair stuck jobs
    repaired = repair_stuck_jobs(conn, args.force, args.timeout)
    
    if repaired > 0:
        logger.info(f"✅ Successfully repaired {repaired} stuck jobs")
    else:
        logger.info("No jobs needed repair")

if __name__ == "__main__":
    main()
