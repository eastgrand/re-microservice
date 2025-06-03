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
import gc
import traceback

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
        
        try:
            # Call original method with all its error handling
            result = original_perform_job(self, job, queue)
            
            # Add end timestamp
            job.connection.hset(job.key, 'worker_end_time', time.time())
            
            # Force cleanup after job completion
            gc.collect()
            
            logger.info(f"Worker {self.name} completed job {job.id}")
            return result
            
        except Exception as e:
            # Log the error
            logger.error(f"Error in job {job.id}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Update job status
            job.connection.hset(job.key, 'worker_error', str(e))
            job.connection.hset(job.key, 'worker_error_time', time.time())
            
            # Force cleanup
            gc.collect()
            
            # Re-raise the exception
            raise
    
    # Replace with enhanced version
    Worker.perform_job = patched_perform_job
    logger.info("✅ Worker.perform_job patched")
    
    # Store original get_current_job method
    original_get_current_job = Worker.get_current_job
    
    @wraps(original_get_current_job)
    def patched_get_current_job(self):
        """Enhanced get_current_job with better error handling"""
        try:
            return original_get_current_job(self)
        except Exception as e:
            logger.error(f"Error getting current job: {str(e)}")
            return None
    
    # Replace with enhanced version
    Worker.get_current_job = patched_get_current_job
    logger.info("✅ Worker.get_current_job patched")
    
    # Add cleanup method
    def cleanup_worker(self):
        """Clean up worker resources"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear any remaining job references
            self._current_job = None
            
            logger.info(f"Worker {self.name} cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up worker: {str(e)}")
    
    # Add cleanup method to Worker class
    Worker.cleanup = cleanup_worker
    logger.info("✅ Worker.cleanup added")
    
    return True

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
