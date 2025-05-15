#!/usr/bin/env python3
"""
Worker Process Monitoring and Cleanup

This script improves job handling in the SHAP microservice by:
1. Adding timeout handling to worker processes
2. Detecting and recovering stuck jobs
3. Automatically cleaning up stale job registries
"""

import os
import sys
import time
import logging
from redis_connection_patch import apply_all_patches

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("worker-fix")

def patch_job_handling():
    """Patch the RQ job handling to better handle timeouts and cleanup"""
    import rq
    from rq.worker import Worker
    from rq.job import Job
    from rq.queue import Queue
    from functools import wraps
    
    logger.info("Applying worker process patches...")
    
    # Store original perform_job method
    original_perform_job = Worker.perform_job
    
    @wraps(original_perform_job)
    def patched_perform_job(self, job, queue):
        """Enhanced perform_job with better error handling and status updates"""
        logger.info(f"Worker {self.name} starting job {job.id}")
        
        # Add start timestamp directly to Redis for tracking
        job.connection.hset(job.key, 'worker_start_time', time.time())
        job.connection.hset(job.key, 'worker_name', self.name)
        
        # Call original method with all its error handling
        result = original_perform_job(self, job, queue)
        
        # Add end timestamp
        job.connection.hset(job.key, 'worker_end_time', time.time())
        
        logger.info(f"Worker {self.name} completed job {job.id}")
        return result
    
    # Replace with enhanced version
    Worker.perform_job = patched_perform_job
    logger.info("✅ Worker.perform_job patched")
    
    # Store original get_current_job method
    original_get_current_job = Worker.get_current_job
    
    @wraps(original_get_current_job)
    def patched_get_current_job(self):
        """Enhanced get_current_job that handles edge cases better"""
        try:
            job = original_get_current_job(self)
            if job:
                # Update last activity time for monitoring
                job.connection.hset(job.key, 'last_activity', time.time())
            return job
        except Exception as e:
            logger.error(f"Error in get_current_job: {str(e)}")
            return None
    
    # Replace with enhanced version
    Worker.get_current_job = patched_get_current_job
    logger.info("✅ Worker.get_current_job patched")
    
    # Add custom method to Job class to check if a job is potentially stuck
    def is_stuck(self, threshold_seconds=300):
        """Check if a job appears to be stuck (in started state too long)"""
        if self.get_status() != 'started':
            return False
            
        # Check if job has been in started state too long
        if self.started_at:
            worker_start = self.connection.hget(self.key, 'worker_start_time')
            if worker_start:
                worker_start = float(worker_start)
                current_time = time.time()
                duration = current_time - worker_start
                return duration > threshold_seconds
            
            # Fall back to started_at if worker_start_time not available
            current_time = time.time()
            duration = current_time - self.started_at
            return duration > threshold_seconds
            
        return False
    
    # Add the method to the Job class
    Job.is_stuck = is_stuck
    logger.info("✅ Added Job.is_stuck method")
    
    # Patch Queue.enqueue method to add more metadata
    original_enqueue = Queue.enqueue
    
    @wraps(original_enqueue)
    def patched_enqueue(self, *args, **kwargs):
        """Enhanced enqueue with better metadata"""
        # Add default timeout if not specified
        if 'job_timeout' not in kwargs:
            kwargs['job_timeout'] = 600  # 10 minutes
            
        result = original_enqueue(self, *args, **kwargs)
        # Add submission metadata
        if result:
            result.connection.hset(result.key, 'enqueue_timestamp', time.time())
        return result
        
    # Replace with enhanced version
    Queue.enqueue = patched_enqueue
    logger.info("✅ Queue.enqueue patched")
    
    # Create cleanup function for queue
    def cleanup_stale_jobs(self, age_threshold=3600):
        """Clean up stale job registries"""
        try:
            # Check started jobs
            started_job_ids = self.started_job_registry.get_job_ids()
            cleaned_count = 0
            
            current_time = time.time()
            
            for job_id in started_job_ids:
                try:
                    job = Job.fetch(job_id, connection=self.connection)
                    
                    # Check if job is stale
                    if job.is_stuck(threshold_seconds=age_threshold):
                        logger.warning(f"Cleaning up stuck job {job_id}")
                        
                        # Remove from started registry
                        self.started_job_registry.remove(job)
                        
                        # Add to failed registry with explanation
                        job.set_status(rq.job.JobStatus.FAILED)
                        job.exc_info = "Job timed out or became stuck in worker process"
                        job.save()
                        self.failed_job_registry.add(job, current_time)
                        
                        cleaned_count += 1
                except Exception as e:
                    logger.error(f"Error cleaning up job {job_id}: {str(e)}")
            
            return cleaned_count
        except Exception as e:
            logger.error(f"Error cleaning up stale jobs: {str(e)}")
            return 0
    
    # Add the method to the Queue class
    Queue.cleanup_stale_jobs = cleanup_stale_jobs
    logger.info("✅ Added Queue.cleanup_stale_jobs method")
    
    logger.info("Worker process patches applied successfully")

def add_cleanup_endpoint(app):
    """Add an endpoint to manually trigger cleanup of stuck jobs"""
    try:
        from flask import jsonify
        import rq
        from rq.queue import Queue
        
        # Add cleanup endpoint
        @app.route('/admin/cleanup_jobs', methods=['POST'])
        def cleanup_jobs_endpoint():
            """Clean up stale jobs that appear to be stuck"""
            try:
                # Get Redis connection from app
                redis_conn = app.config.get('redis_conn')
                if not redis_conn:
                    return jsonify({
                        "success": False,
                        "error": "Redis connection not found in app config"
                    }), 500
                
                # Create queue
                queue = Queue('shap-jobs', connection=redis_conn)
                
                # Run cleanup
                cleaned_count = queue.cleanup_stale_jobs(age_threshold=600)  # 10 minutes
                
                return jsonify({
                    "success": True,
                    "cleaned_count": cleaned_count,
                    "message": f"Cleaned up {cleaned_count} stuck jobs"
                })
            except Exception as e:
                logger.error(f"Error in cleanup endpoint: {str(e)}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
                
        logger.info("✅ Added job cleanup endpoint: /admin/cleanup_jobs")
        return True
    except Exception as e:
        logger.error(f"Could not add job cleanup endpoint: {str(e)}")
        return False

def apply_all_worker_patches(app=None):
    """Apply all worker-related patches"""
    # First apply Redis connection patches
    apply_all_patches(app)
    
    # Then apply worker-specific patches
    patch_job_handling()
    
    # Add cleanup endpoint if app is provided
    if app is not None:
        add_cleanup_endpoint(app)
        
    logger.info("✅ All worker patches applied")

if __name__ == "__main__":
    print("This module applies fixes for worker processes in the SHAP microservice.")
    print("To use, import and call apply_all_worker_patches() in your app.py file.")
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Simple test mode - just apply patches
        apply_all_worker_patches()
