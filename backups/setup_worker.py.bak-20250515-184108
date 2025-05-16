#!/usr/bin/env python3
# filepath: /Users/voldeck/code/shap-microservice/setup_worker.py
"""
Worker setup script for SHAP microservice

This script configures the worker properly with memory optimizations
and starts the Redis Queue worker process.
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
logger = logging.getLogger("worker-setup")

# Ensure necessary environment variables
REDIS_URL = os.environ.get("REDIS_URL")
if not REDIS_URL:
    logger.error("ERROR: REDIS_URL environment variable not set")
    sys.exit(1)

# Get batch size from environment or use default
MAX_ROWS_TO_PROCESS = int(os.environ.get('SHAP_MAX_BATCH_SIZE', '300'))

def setup_worker():
    """Set up the worker with memory optimizations"""
    logger.info("Setting up memory-optimized SHAP worker")
    
    # Enable aggressive garbage collection
    gc.enable()
    logger.info(f"Garbage collection enabled with thresholds: {gc.get_threshold()}")

    # Try to apply memory patches if possible
    try:
        # First create the memory optimization module if it doesn't exist
        if not os.path.exists('shap_memory_fix.py'):
            logger.info("Creating shap_memory_fix.py module...")
            create_memory_fix_module()
            
        # Now import and apply the memory patches
        from shap_memory_fix import apply_memory_patches
        apply_memory_patches()
        logger.info("Applied SHAP memory optimizations")
    except ImportError:
        logger.warning("Could not import shap_memory_fix - continuing without memory optimizations")
        traceback.print_exc()
    except Exception as e:
        logger.warning(f"Error applying memory optimizations: {str(e)}")
        traceback.print_exc()

    # Optional: Apply Redis connection patches if available
    try:
        import redis
        from redis_connection_patch import apply_all_patches
        apply_all_patches()
        logger.info("Applied Redis connection patches")
    except ImportError:
        logger.warning("Could not import redis_connection_patch - continuing without Redis patches")
    except Exception as e:
        logger.warning(f"Error applying Redis patches: {str(e)}")
        traceback.print_exc()

    # Optional: Repair stuck jobs if possible
    try:
        import redis
        from repair_stuck_jobs import repair_stuck_jobs
        conn = redis.from_url(REDIS_URL)
        repaired = repair_stuck_jobs(conn, force=True)
        logger.info(f"Repaired {repaired} stuck jobs")
    except ImportError:
        logger.warning("Could not import repair_stuck_jobs")
    except Exception as e:
        logger.warning(f"Error repairing stuck jobs: {str(e)}")
        traceback.print_exc()
    
    logger.info(f"Worker setup complete. Using max batch size of {MAX_ROWS_TO_PROCESS} rows")
    return True

def create_memory_fix_module():
    """Create the memory optimization module if it doesn't exist"""
    content = """#!/usr/bin/env python3
\"\"\"
SHAP Memory Optimization

This module provides memory optimization for SHAP analysis,
allowing for the processing of larger datasets without running out of memory.
\"\"\"

import os
import gc
import logging
import numpy as np
from typing import Union, Optional, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shap-memory-fix")

# Configuration for batch processing
MAX_ROWS_TO_PROCESS = int(os.environ.get('SHAP_MAX_BATCH_SIZE', '300'))

class ShapValuesWrapper:
    \"\"\"Wrapper class to maintain compatibility with SHAP values API\"\"\"
    
    def __init__(self, values, base_values=None, data=None, feature_names=None):
        \"\"\"Initialize with raw SHAP values\"\"\"
        self.values = values
        self.base_values = base_values
        self.data = data
        self.feature_names = feature_names
        
    def __len__(self):
        \"\"\"Return the number of rows in the SHAP values\"\"\"
        if hasattr(self.values, '__len__'):
            return len(self.values)
        return 0

def apply_memory_patches(app=None):
    \"\"\"
    Apply memory optimization patches to the Flask app
    
    Args:
        app: Flask application instance
    \"\"\"
    import gc
    
    # Enable garbage collection to run more aggressively
    gc.enable()
    logger.info("Garbage collection enabled with thresholds: %s", gc.get_threshold())
    
    # Function to create memory-optimized explainer
    def create_memory_optimized_explainer(model, X, feature_names=None, 
                                         max_rows=MAX_ROWS_TO_PROCESS):
        \"\"\"
        Creates a memory-optimized explainer by processing data in batches
        
        Args:
            model: The trained model to explain
            X: Input features to explain
            feature_names: Optional list of feature names
            max_rows: Maximum number of rows to process in one batch
            
        Returns:
            ShapValuesWrapper containing the computed SHAP values
        \"\"\"
        import shap
        
        # Start with garbage collection to ensure we have maximum memory available
        gc.collect()
        
        logger.info(f"Creating memory-optimized explainer for {len(X)} rows")
        logger.info(f"Using max batch size of {max_rows} rows")
        
        # Check if the dataset is small enough to process in one go
        if len(X) <= max_rows:
            logger.info("Dataset small enough for direct processing")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X)
            
            # Return ShapValuesWrapper if needed
            if not hasattr(shap_values, 'values'):
                return ShapValuesWrapper(shap_values)
            return shap_values
        
        # Process in batches for large datasets
        logger.info(f"Processing large dataset in batches")
        all_shap_values = []
        total_rows = len(X)
        chunks = (total_rows + max_rows - 1) // max_rows  # Ceiling division
        
        for i in range(chunks):
            start_idx = i * max_rows
            end_idx = min((i + 1) * max_rows, total_rows)
            
            logger.info(f"Processing batch {i+1}/{chunks} (rows {start_idx}-{end_idx})")
            
            # Extract this chunk of data
            X_chunk = X.iloc[start_idx:end_idx]
            
            # Create explainer and get SHAP values for this chunk
            explainer = shap.TreeExplainer(model)
            chunk_shap_values = explainer(X_chunk)
            
            # Extract values (handle different return types from different SHAP versions)
            if hasattr(chunk_shap_values, 'values'):
                all_shap_values.append(chunk_shap_values.values)
            else:
                all_shap_values.append(chunk_shap_values)
                
            # Force cleanup to free memory
            del explainer
            del chunk_shap_values
            del X_chunk
            gc.collect()
            
        # Combine all chunks
        logger.info("Combining SHAP values from all batches")
        try:
            combined_values = np.vstack(all_shap_values)
            return ShapValuesWrapper(combined_values)
        except:
            logger.warning("Could not combine values using np.vstack, returning list")
            return ShapValuesWrapper(all_shap_values)
    
    # Patch the calculation function
    global calculate_shap_values
    
    # Store original function if it exists
    if 'calculate_shap_values' in globals():
        original_calculate_shap_values = calculate_shap_values
        
        # Define patched function with same signature
        def memory_optimized_calculate_shap_values(model, X, feature_names=None, **kwargs):
            \"\"\"Memory optimized version of calculate_shap_values\"\"\"
            logger.info("Using memory-optimized SHAP calculation")
            return create_memory_optimized_explainer(model, X, feature_names, MAX_ROWS_TO_PROCESS)
        
        # Replace the global function
        calculate_shap_values = memory_optimized_calculate_shap_values
    else:
        # If function doesn't exist yet, create it
        def calculate_shap_values(model, X, feature_names=None, **kwargs):
            \"\"\"Memory optimized SHAP calculation function\"\"\"
            logger.info("Using memory-optimized SHAP calculation")
            return create_memory_optimized_explainer(model, X, feature_names, MAX_ROWS_TO_PROCESS)
    
    # If app is provided, add memory monitoring endpoint
    if app is not None:
        logger.info("Adding memory monitoring endpoint to Flask app")
        try:
            from flask import jsonify
            import psutil
            
            @app.route('/admin/memory', methods=['GET'])
            def memory_status():
                \"\"\"Return current memory usage and status\"\"\"
                try:
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    return jsonify({
                        "success": True,
                        "memory_usage_mb": memory_mb,
                        "optimized_worker_applied": True,
                        "gc_enabled": gc.isenabled(),
                        "gc_counts": gc.get_count(),
                        "gc_threshold": gc.get_threshold(),
                        "max_rows_per_batch": MAX_ROWS_TO_PROCESS
                    })
                except Exception as e:
                    logger.error(f"Error in memory endpoint: {str(e)}")
                    return jsonify({
                        "success": False,
                        "error": str(e)
                    }), 500
        except ImportError:
            logger.warning("Could not add memory endpoint: psutil not installed")
    
    logger.info("Memory optimization patches applied successfully")
    return True

# Export the memory optimization function
__all__ = ['apply_memory_patches', 'MAX_ROWS_TO_PROCESS', 'ShapValuesWrapper']
"""
    
    with open('shap_memory_fix.py', 'w') as f:
        f.write(content)
    
    logger.info("Created shap_memory_fix.py module")
    return True

def start_worker():
    """Start the RQ worker process"""
    logger.info("Starting RQ worker process")
    
    try:
        import redis
        from rq import Queue, Connection, Worker
        from rq.job import Job
        
        # Connect to Redis with improved parameters
        conn = redis.from_url(
            REDIS_URL,
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
            logger.error(f"Redis ping failed: {str(e)}")
            logger.error("Will try to proceed anyway")
        
        # First, try to repair any stuck jobs
        try:
            queue_name = 'shap-jobs'
            queue = Queue(queue_name, connection=conn)
            
            # Check for any jobs in started state
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
        
        # Start worker with improved settings
        logger.info("Starting worker...")
        with Connection(conn):
            worker = Worker(['shap-jobs'], name=f"memory-optimized-worker-{os.getpid()}")
            logger.info(f"Worker started with ID {worker.name}")
            logger.info("Listening for jobs on queue: shap-jobs")
            logger.info(f"Using max batch size: {MAX_ROWS_TO_PROCESS} rows")
            
            # Start working (with exception handling)
            try:
                worker.work(with_scheduler=True)
            except KeyboardInterrupt:
                logger.info("Worker stopped by user")
                sys.exit(0)
            except Exception as e:
                logger.critical(f"Worker crashed with error: {str(e)}")
                logger.critical(traceback.format_exc())
                sys.exit(1)
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting worker: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    logger.info("Worker setup script started")
    
    # Install required packages if they don't exist
    try:
        import redis
        import rq
        import psutil
    except ImportError:
        logger.info("Installing missing dependencies...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "redis", "rq", "psutil"])
        logger.info("Dependencies installed")
    
    setup_worker()
    start_worker()
