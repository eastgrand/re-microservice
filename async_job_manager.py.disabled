import asyncio
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from enum import Enum
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import redis
import pickle

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobManager:
    """
    Manages asynchronous analysis jobs with Redis backend for persistence.
    Supports job queuing, progress tracking, and result storage.
    """
    
    def __init__(self, redis_url: str = None, max_workers: int = 4):
        self.redis_url = redis_url or "redis://localhost:6379"
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.job_handlers: Dict[str, Callable] = {}
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()  # Test connection
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage.")
            self.redis_client = None
    
    def register_handler(self, job_type: str, handler: Callable):
        """Register a job handler function."""
        self.job_handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")
    
    def submit_job(self, job_type: str, job_data: Dict[str, Any], 
                  priority: int = 1, timeout: int = 300) -> str:
        """
        Submit a new job for processing.
        Returns job ID.
        """
        job_id = str(uuid.uuid4())
        
        job_info = {
            'job_id': job_id,
            'job_type': job_type,
            'status': JobStatus.QUEUED.value,
            'priority': priority,
            'timeout': timeout,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'progress': 0,
            'message': 'Job queued for processing',
            'data': job_data,
            'result': None,
            'error': None,
            'estimated_completion': None
        }
        
        # Store job info
        self._store_job(job_id, job_info)
        
        # Submit to executor
        future = self.executor.submit(self._process_job, job_id)
        job_info['future'] = future
        
        logger.info(f"Submitted job {job_id} of type {job_type}")
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a job."""
        job_info = self._get_job(job_id)
        if not job_info:
            return None
        
        # Don't return the future object or raw data in status
        status_info = {k: v for k, v in job_info.items() 
                      if k not in ['future', 'data']}
        
        return status_info
    
    def get_job_result(self, job_id: str) -> Optional[Any]:
        """Get the result of a completed job."""
        job_info = self._get_job(job_id)
        if not job_info:
            return None
        
        if job_info['status'] == JobStatus.COMPLETED.value:
            return job_info.get('result')
        
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if it's not already completed."""
        job_info = self._get_job(job_id)
        if not job_info:
            return False
        
        if job_info['status'] in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
            return False
        
        # Cancel the future if it exists
        if 'future' in job_info:
            job_info['future'].cancel()
        
        # Update status
        job_info['status'] = JobStatus.CANCELLED.value
        job_info['updated_at'] = datetime.now().isoformat()
        job_info['message'] = 'Job cancelled by user'
        
        self._store_job(job_id, job_info)
        logger.info(f"Cancelled job {job_id}")
        return True
    
    def _process_job(self, job_id: str):
        """Process a job in the background."""
        try:
            job_info = self._get_job(job_id)
            if not job_info:
                logger.error(f"Job {job_id} not found")
                return
            
            job_type = job_info['job_type']
            handler = self.job_handlers.get(job_type)
            
            if not handler:
                raise ValueError(f"No handler registered for job type: {job_type}")
            
            # Update status to processing
            self._update_job_status(job_id, JobStatus.PROCESSING, "Processing job...", 10)
            
            # Create progress callback
            def progress_callback(progress: int, message: str = None):
                self._update_job_status(job_id, JobStatus.PROCESSING, message, progress)
            
            # Execute the job
            start_time = time.time()
            result = handler(job_info['data'], progress_callback)
            processing_time = time.time() - start_time
            
            # Update with result
            job_info = self._get_job(job_id)
            job_info['status'] = JobStatus.COMPLETED.value
            job_info['progress'] = 100
            job_info['message'] = f'Job completed successfully in {processing_time:.2f}s'
            job_info['result'] = result
            job_info['processing_time'] = processing_time
            job_info['updated_at'] = datetime.now().isoformat()
            
            self._store_job(job_id, job_info)
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            
            # Update with error
            job_info = self._get_job(job_id)
            if job_info:
                job_info['status'] = JobStatus.FAILED.value
                job_info['error'] = str(e)
                job_info['message'] = f'Job failed: {str(e)}'
                job_info['updated_at'] = datetime.now().isoformat()
                
                self._store_job(job_id, job_info)
    
    def _update_job_status(self, job_id: str, status: JobStatus, message: str = None, progress: int = None):
        """Update job status, message, and progress."""
        job_info = self._get_job(job_id)
        if not job_info:
            return
        
        job_info['status'] = status.value
        job_info['updated_at'] = datetime.now().isoformat()
        
        if message:
            job_info['message'] = message
        
        if progress is not None:
            job_info['progress'] = progress
            
            # Estimate completion time based on progress
            if progress > 0 and status == JobStatus.PROCESSING:
                created_time = datetime.fromisoformat(job_info['created_at'])
                elapsed = datetime.now() - created_time
                estimated_total = elapsed.total_seconds() * (100 / progress)
                estimated_completion = created_time + timedelta(seconds=estimated_total)
                job_info['estimated_completion'] = estimated_completion.isoformat()
        
        self._store_job(job_id, job_info)
    
    def _store_job(self, job_id: str, job_info: Dict[str, Any]):
        """Store job information."""
        # Store in memory
        self.jobs[job_id] = job_info.copy()
        
        # Store in Redis if available
        if self.redis_client:
            try:
                # Don't store the future object in Redis
                redis_job_info = {k: v for k, v in job_info.items() if k != 'future'}
                self.redis_client.setex(
                    f"job:{job_id}",
                    timedelta(hours=24),  # Jobs expire after 24 hours
                    json.dumps(redis_job_info, default=str)
                )
            except Exception as e:
                logger.warning(f"Failed to store job in Redis: {e}")
    
    def _get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job information."""
        # Try memory first
        if job_id in self.jobs:
            return self.jobs[job_id]
        
        # Try Redis if available
        if self.redis_client:
            try:
                job_data = self.redis_client.get(f"job:{job_id}")
                if job_data:
                    job_info = json.loads(job_data)
                    self.jobs[job_id] = job_info  # Cache in memory
                    return job_info
            except Exception as e:
                logger.warning(f"Failed to retrieve job from Redis: {e}")
        
        return None
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old jobs from memory and Redis."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean up memory
        old_job_ids = []
        for job_id, job_info in self.jobs.items():
            created_time = datetime.fromisoformat(job_info['created_at'])
            if created_time < cutoff_time:
                old_job_ids.append(job_id)
        
        for job_id in old_job_ids:
            del self.jobs[job_id]
        
        logger.info(f"Cleaned up {len(old_job_ids)} old jobs from memory")
        
        # Redis cleanup is handled by expiration
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        status_counts = {}
        for job_info in self.jobs.values():
            status = job_info['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_jobs': len(self.jobs),
            'active_workers': self.max_workers,
            'status_counts': status_counts,
            'redis_connected': self.redis_client is not None
        }
    
    def shutdown(self):
        """Shutdown the job manager."""
        logger.info("Shutting down job manager...")
        self.executor.shutdown(wait=True)
        if self.redis_client:
            self.redis_client.close()

# Global job manager instance
job_manager = JobManager()

# Analysis job handler
def handle_analysis_job(job_data: Dict[str, Any], progress_callback: Callable) -> Dict[str, Any]:
    """
    Handle SHAP analysis jobs.
    This function will be called by the job manager for analysis jobs.
    """
    try:
        progress_callback(20, "Loading data and model...")
        
        # Import analysis worker here to avoid circular imports
        from enhanced_analysis_worker import enhanced_analysis_worker
        
        progress_callback(40, "Performing SHAP analysis...")
        
        # Call the enhanced analysis worker
        result = enhanced_analysis_worker(job_data)
        
        progress_callback(90, "Finalizing results...")
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis job failed: {e}")
        raise

# Register the analysis job handler
job_manager.register_handler('analysis', handle_analysis_job) 