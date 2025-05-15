#!/usr/bin/env python3
"""
Worker Diagnostics Script

This script diagnoses issues with the RQ worker process in the SHAP microservice.
It directly connects to Redis and examines the state of the worker and job queues.
"""

import os
import sys
import time
import redis
import json
import logging
from rq import Queue, Worker
from rq.job import Job

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("worker-diagnostics")

def get_redis_connection():
    """Get a connection to Redis"""
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

def get_queue_status(conn):
    """Get information about the job queue"""
    try:
        queue = Queue('shap-jobs', connection=conn)
        
        logger.info(f"Queue: {queue.name}")
        logger.info(f"Jobs in queue: {queue.count}")
        
        job_ids = queue.job_ids
        if job_ids:
            logger.info(f"Job IDs in queue: {job_ids}")
            
        logger.info(f"Started job registry count: {queue.started_job_registry.count}")
        started_job_ids = queue.started_job_registry.get_job_ids()
        logger.info(f"Started job IDs: {started_job_ids}")
        
        logger.info(f"Failed job registry count: {queue.failed_job_registry.count}")
        failed_job_ids = queue.failed_job_registry.get_job_ids()
        logger.info(f"Failed job IDs: {failed_job_ids}")
        
        logger.info(f"Finished job registry count: {queue.finished_job_registry.count}")
        finished_job_ids = queue.finished_job_registry.get_job_ids()[:5]  # Get only the first 5
        logger.info(f"Recent finished job IDs: {finished_job_ids}")
        
        # Check workers
        workers = Worker.all(connection=conn)
        logger.info(f"Active workers: {len(workers)}")
        for i, worker in enumerate(workers):
            logger.info(f"Worker {i+1}: {worker.name}")
            logger.info(f"  State: {worker.get_state()}")
            logger.info(f"  Last heartbeat: {worker.last_heartbeat}")
            current_job = worker.get_current_job()
            if current_job:
                logger.info(f"  Current job: {current_job.id}")
                logger.info(f"  Job status: {current_job.get_status()}")
                logger.info(f"  Job created: {current_job.created_at}")
                
        return {
            "queue_count": queue.count,
            "started_count": queue.started_job_registry.count,
            "failed_count": queue.failed_job_registry.count,
            "finished_count": queue.finished_job_registry.count,
            "started_jobs": started_job_ids,
            "failed_jobs": failed_job_ids,
            "worker_count": len(workers)
        }
    except Exception as e:
        logger.error(f"❌ Error getting queue status: {str(e)}")
        return None

def inspect_job(conn, job_id):
    """Inspect the status and details of a specific job"""
    try:
        job = Job.fetch(job_id, connection=conn)
        
        logger.info(f"Job ID: {job.id}")
        logger.info(f"Status: {job.get_status()}")
        logger.info(f"Created at: {job.created_at}")
        logger.info(f"Enqueued at: {job.enqueued_at}")
        
        if job.started_at:
            logger.info(f"Started at: {job.started_at}")
            
        if job.ended_at:
            logger.info(f"Ended at: {job.ended_at}")
            
        if job.result:
            logger.info(f"Has result: {'success' if job.result.get('success') else 'failure'}")
            
        if job.exc_info:
            logger.info(f"Exception info: {job.exc_info}")
            
        logger.info(f"TTL: {job.ttl}")
        logger.info(f"Result TTL: {job.result_ttl}")
        logger.info(f"Failure TTL: {job.failure_ttl}")
        
        # Check which queue this job is in
        queue = Queue('shap-jobs', connection=conn)
        
        if job_id in queue.job_ids:
            logger.info(f"Job is in main queue")
        elif job_id in queue.started_job_registry.get_job_ids():
            logger.info(f"Job is in started registry")
        elif job_id in queue.finished_job_registry.get_job_ids():
            logger.info(f"Job is in finished registry")
        elif job_id in queue.failed_job_registry.get_job_ids():
            logger.info(f"Job is in failed registry")
        else:
            logger.info(f"Job is not in any standard registry")
    except Exception as e:
        logger.error(f"❌ Error inspecting job {job_id}: {str(e)}")

def fix_stuck_jobs(conn):
    """Attempt to fix stuck jobs"""
    try:
        queue = Queue('shap-jobs', connection=conn)
        
        # Get started jobs
        started_job_ids = queue.started_job_registry.get_job_ids()
        
        if not started_job_ids:
            logger.info("No stuck jobs found in started registry.")
            return
            
        logger.info(f"Found {len(started_job_ids)} potentially stuck jobs in started registry.")
        
        for job_id in started_job_ids:
            try:
                job = Job.fetch(job_id, connection=conn)
                
                # Calculate how long the job has been in started state
                if job.started_at:
                    started_time = job.started_at
                    current_time = job.connection.hget(job.key, 'current_time') or time.time()
                    duration = current_time - started_time
                    
                    logger.info(f"Job {job_id} has been in started state for {duration:.2f} seconds")
                    
                    # If job has been started for too long (5 minutes), requeue it
                    if duration > 300:
                        logger.warning(f"Job {job_id} appears stuck. Attempting to requeue...")
                        
                        # Remove from started registry
                        queue.started_job_registry.remove(job)
                        
                        # Requeue the job
                        queue.enqueue_job(job)
                        logger.info(f"✅ Requeued job {job_id}")
            except Exception as e:
                logger.error(f"Error fixing job {job_id}: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Error fixing stuck jobs: {str(e)}")

def main():
    # Connect to Redis
    conn = get_redis_connection()
    if not conn:
        sys.exit(1)
        
    # Print queue status
    status = get_queue_status(conn)
    if not status:
        sys.exit(1)
        
    # Inspect any stuck jobs
    stuck_jobs = status.get("started_jobs", [])
    if stuck_jobs:
        logger.info("\n--- INSPECTING STUCK JOBS ---")
        for job_id in stuck_jobs:
            inspect_job(conn, job_id)
            
    # Attempt to fix stuck jobs
    fix_stuck_jobs(conn)
    
    # Provide recommendations
    logger.info("\n--- DIAGNOSTIC SUMMARY ---")
    
    if status["started_count"] > 0:
        logger.warning("⚠️ There are jobs stuck in 'started' state")
        logger.info("Possible causes:")
        logger.info("1. Worker process is not running")
        logger.info("2. Worker is stuck or crashed while processing a job")
        logger.info("3. Job execution is taking too long")
        logger.info("\nRecommended actions:")
        logger.info("1. Verify worker is running: 'rq info -u REDIS_URL'")
        logger.info("2. Restart the worker: 'rq worker shap-jobs --url REDIS_URL'")
        logger.info("3. Check for memory issues in worker process")
    elif status["queue_count"] > 0 and status["worker_count"] == 0:
        logger.warning("⚠️ There are jobs in queue but no active workers")
        logger.info("Recommended action: Start the worker process")
    else:
        logger.info("✅ Queue system appears to be in good state")

if __name__ == "__main__":
    main()
