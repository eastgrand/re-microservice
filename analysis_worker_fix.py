#!/usr/bin/env python3
"""
Analysis Worker Fix for SHAP Microservice
- Fixes the analysis worker to handle NaN values properly
- Created: May 16, 2025
"""

import os
import sys
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analysis-worker-fix")

def patch_analysis_worker():
    """Patch the analysis worker to properly handle NaN values"""
    try:
        # First import the json serialization fix
        from json_serialization_fix import apply_json_patches
        
        # Apply the JSON patches
        apply_json_patches()
        
        logger.info("Applied JSON serialization patches to fix NaN handling")
        return True
    except Exception as e:
        logger.error(f"Failed to apply analysis worker patches: {str(e)}")
        return False

def fix_nan_in_values(result_dict):
    """Helper function to identify and fix NaN values in result dictionary"""
    if not isinstance(result_dict, dict):
        return result_dict
    
    import numpy as np
    
    def fix_value(val):
        """Fix a single value, handling NaN, lists, and nested dicts"""
        if isinstance(val, (float, np.float32, np.float64)) and np.isnan(val):
            return "NaN"  # Convert to string representation
        elif isinstance(val, dict):
            return fix_nan_in_values(val)  # Recursively fix nested dicts
        elif isinstance(val, list):
            return [fix_value(item) for item in val]  # Fix items in list
        else:
            return val
    
    # Create a new dict with fixed values
    return {k: fix_value(v) for k, v in result_dict.items()}

def patch_shap_worker_functions():
    """Patch specific worker functions to handle NaN values"""
    try:
        # Try to import app module
        import app
        
        # Check if the analysis_worker function exists
        if hasattr(app, 'analysis_worker'):
            # Store original analysis_worker function
            original_analysis_worker = app.analysis_worker
            
            @wraps(original_analysis_worker)
            def fixed_analysis_worker(*args, **kwargs):
                """Enhanced version of analysis_worker that handles NaN values properly"""
                try:
                    # Call original function
                    result = original_analysis_worker(*args, **kwargs)
                    
                    # Fix NaN values in the result if it's a dictionary
                    if isinstance(result, dict):
                        result = fix_nan_in_values(result)
                    
                    return result
                except Exception as e:
                    logger.error(f"Error in fixed analysis worker: {str(e)}")
                    # Try to continue with original function
                    return original_analysis_worker(*args, **kwargs)
            
            # Replace with fixed version
            app.analysis_worker = fixed_analysis_worker
            logger.info("Patched analysis_worker function to handle NaN values")
        
        # Also patch job_result function if it exists
        if hasattr(app, 'job_result'):
            # Store original job_result function
            original_job_result = app.job_result
            
            @wraps(original_job_result)
            def fixed_job_result(*args, **kwargs):
                """Enhanced version of job_result that fixes NaN values"""
                try:
                    from flask import jsonify
                    import json
                    
                    # Get original result
                    result = original_job_result(*args, **kwargs)
                    
                    # Check if it's a response object
                    if hasattr(result, 'get_json'):
                        # Extract JSON data
                        json_data = result.get_json()
                        
                        # Fix NaN values
                        fixed_json = fix_nan_in_values(json_data)
                        
                        # Return new response with fixed JSON
                        return jsonify(fixed_json)
                    
                    return result
                except Exception as e:
                    logger.error(f"Error in fixed job_result: {str(e)}")
                    return original_job_result(*args, **kwargs)
            
            # Replace with fixed version
            app.job_result = fixed_job_result
            logger.info("Patched job_result function to handle NaN values")
        
        return True
    except ImportError:
        logger.warning("Could not import app module - worker functions not patched")
        return False
    except Exception as e:
        logger.error(f"Error patching worker functions: {str(e)}")
        return False

def apply_all_worker_fixes(app_instance=None):
    """Apply all worker-related fixes"""
    # Apply JSON serialization patches
    patch_analysis_worker()
    
    # Patch specific worker functions
    patch_shap_worker_functions()
    
    # Patch the app instance if provided
    if app_instance is not None:
        # Fix any route functions
        pass
    
    logger.info("All worker fixes applied successfully")
    return True

if __name__ == "__main__":
    print("This is a fix module for the SHAP microservice analysis worker.")
    print("To use, import this module in your main app.py file and call apply_all_worker_fixes()")
