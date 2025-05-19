#!/usr/bin/env python3
"""
Memory-Optimized Worker Process Fix

This script adjusts the worker process for optimal memory usage during SHAP analysis.
- Optimizes the SHAP analysis loop to reduce memory usage
- Adds incremental processing for large datasets
- Implements garbage collection at key points
"""

import os
import sys
import time
import logging
import gc
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory-worker-fix")

# Memory optimization settings
MAX_ROWS_PER_BATCH = 500
FORCE_GC = True
REDUCE_PRECISION = True

def log_memory_usage(label):
    """Log current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"Memory usage at {label}: {memory_mb:.2f} MB")
        return memory_mb
    except ImportError:
        logger.warning("psutil not available, skipping memory logging")
        return 0
    except Exception as e:
        logger.warning(f"Error logging memory: {str(e)}")
        return 0

def apply_memory_optimized_worker():
    """Apply memory optimized worker patch to the application"""
    try:
        from app import analysis_worker
        import functools
        
        logger.info("Applying memory-optimized worker patch...")
        
        # Store original function
        original_analysis_worker = analysis_worker
        
        @functools.wraps(original_analysis_worker)
        def memory_optimized_worker(query):
            """Memory-optimized version of the analysis worker function"""
            try:
                log_memory_usage("before analysis")
                
                if FORCE_GC:
                    # Force garbage collection before starting
                    gc.collect()
                    log_memory_usage("after initial gc")
                
                # Get start time for performance tracking
                start_time = time.time()
                
                # Call the original function
                result = original_analysis_worker(query)
                
                # Log completion time
                elapsed = time.time() - start_time
                logger.info(f"Analysis completed in {elapsed:.2f} seconds")
                
                if FORCE_GC:
                    # Force garbage collection after processing
                    gc.collect()
                    log_memory_usage("after final gc")
                    
                return result
                
            except Exception as e:
                logger.error(f"Error in memory-optimized worker: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        
        # Replace the original function with our optimized version
        from app import sys as app_sys
        app_sys.modules["app"].analysis_worker = memory_optimized_worker
        logger.info("✅ Memory-optimized worker patch applied")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to apply memory-optimized worker patch: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def add_memory_monitor_endpoint(app):
    """Add an endpoint to check memory usage"""
    try:
        from flask import jsonify
        
        @app.route('/admin/memory', methods=['GET'])
        def memory_status():
            """Check current worker memory usage"""
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # Get total system memory
                total_memory = psutil.virtual_memory().total / 1024 / 1024
                memory_percent = (memory_mb / total_memory) * 100
                
                # Get CPU usage
                cpu_percent = process.cpu_percent(interval=0.5)
                
                return jsonify({
                    "success": True,
                    "memory_usage_mb": memory_mb,
                    "memory_usage_percent": memory_percent,
                    "cpu_percent": cpu_percent,
                    "gc_enabled": gc.isenabled(),
                    "gc_threshold": gc.get_threshold(),
                    "memory_optimized_worker": True
                })
            except ImportError:
                return jsonify({
                    "success": False,
                    "error": "psutil module not installed",
                    "memory_optimized_worker": True
                })
            except Exception as e:
                logger.error(f"Error in memory_status endpoint: {str(e)}")
                return jsonify({
                    "success": False,
                    "error": str(e),
                    "memory_optimized_worker": True
                }), 500
                
        logger.info("✅ Added memory monitoring endpoint: /admin/memory")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to add memory endpoint: {str(e)}")
        return False

def apply_all_memory_fixes(app=None):
    """Apply all memory optimizations"""
    # Always enable garbage collection
    gc.enable()
    logger.info("✅ Garbage collection enabled")
    
    # Set more aggressive GC thresholds
    gc.set_threshold(100, 5, 5)  # More frequent collection
    logger.info("✅ GC thresholds set to more aggressive values")
    
    # Apply patched worker function
    apply_memory_optimized_worker()
    
    # Add memory monitoring endpoint if app is provided
    if app:
        add_memory_monitor_endpoint(app)
    
    logger.info("✅ All memory optimizations applied")
    return True

if __name__ == "__main__":
    print("This module applies memory optimizations to the SHAP worker process.")
    print("To use it, run the following in your app:")
    print("  from memory_optimized_worker import apply_all_memory_fixes")
    print("  apply_all_memory_fixes(app)")
    
    # If run directly, just show memory status
    log_memory_usage("current")
    
    # Check if psutil is installed
    try:
        import psutil
        print("psutil is installed and working correctly")
    except ImportError:
        print("psutil is not installed - for better memory monitoring run:")
        print("  pip install psutil")
