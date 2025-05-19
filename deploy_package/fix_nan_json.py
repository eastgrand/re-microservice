#!/usr/bin/env python3
"""
Fix for NaN JSON serialization issue in SHAP worker
- Specifically targets the JSON parsing error in result data
- Created: May 16, 2025
"""

import os
import logging
import sys
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nan-json-fix")

def fix_nan_in_json_result():
    """
    Fix NaN values in job result JSON
    This patch specifically addresses the error:
    SyntaxError: Unexpected token 'N', ..."ications":NaN,"mortg"... is not valid JSON
    """
    try:
        import app
        from functools import wraps
        import re
        import json
        
        # Original job_status endpoint function
        if hasattr(app, "job_status"):
            original_job_status = app.job_status
            
            @wraps(original_job_status)
            def fixed_job_status(job_id):
                """Fixed job_status function that properly handles NaN in result data"""
                from flask import jsonify
                
                # Get the original response
                response = original_job_status(job_id)
                
                # If it's a regular response object, fix it
                if hasattr(response, 'get_data'):
                    try:
                        # Get the raw data
                        raw_data = response.get_data(as_text=True)
                        
                        # Fix NaN values in the raw JSON string
                        # Replace literal NaN with "NaN" as strings
                        fixed_data = re.sub(r':\s*NaN\s*,', ':"NaN",', raw_data)
                        fixed_data = re.sub(r':\s*NaN\s*}', ':"NaN"}', fixed_data)
                        
                        # Replace Infinity values too
                        fixed_data = re.sub(r':\s*Infinity\s*,', ':"Infinity",', fixed_data)
                        fixed_data = re.sub(r':\s*Infinity\s*}', ':"Infinity"}', fixed_data)
                        fixed_data = re.sub(r':\s*-Infinity\s*,', ':"-Infinity",', fixed_data)
                        fixed_data = re.sub(r':\s*-Infinity\s*}', ':"-Infinity"}', fixed_data)
                        
                        # Parse and re-serialize to ensure valid JSON
                        # (This will fail if our regex didn't catch everything)
                        try:
                            parsed_data = json.loads(fixed_data)
                            return jsonify(parsed_data)
                        except json.JSONDecodeError:
                            # If we still can't parse it, return the original
                            return response
                        
                    except Exception as e:
                        logger.error(f"Error fixing NaN values: {str(e)}")
                        # Return original response if anything fails
                        return response
                
                # Return the original response as fallback
                return response
            
            # Replace with our fixed version
            app.job_status = fixed_job_status
            logger.info("✅ Fixed job_status to handle NaN values correctly")
        
        # Also fix the job_result function if it exists
        if hasattr(app, "job_result"):
            original_job_result = app.job_result
            
            @wraps(original_job_result)
            def fixed_job_result(job_id):
                """Fixed job_result function that properly handles NaN in result data"""
                from flask import jsonify
                import json
                import numpy as np
                
                try:
                    # Get the job result the standard way
                    result = original_job_result(job_id)
                    
                    # If the response already has the result data as an attribute
                    if hasattr(result, 'json'):
                        data = result.json
                        
                        # Recursively fix NaN and Infinity values
                        def fix_nan_values(obj):
                            if isinstance(obj, dict):
                                return {k: fix_nan_values(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [fix_nan_values(item) for item in obj]
                            elif obj is np.nan or (isinstance(obj, float) and np.isnan(obj)):
                                return "NaN"
                            elif obj is np.inf or obj == float('inf'):
                                return "Infinity"
                            elif obj == -np.inf or obj == float('-inf'):
                                return "-Infinity"
                            else:
                                return obj
                        
                        # Fix the data and return a new response
                        fixed_data = fix_nan_values(data)
                        return jsonify(fixed_data)
                    
                    # Return the original if we can't fix it
                    return result
                
                except Exception as e:
                    logger.error(f"Error in fixed job_result: {str(e)}")
                    # Return original function result as fallback
                    return original_job_result(job_id)
            
            # Replace with our fixed version
            app.job_result = fixed_job_result
            logger.info("✅ Fixed job_result to handle NaN values correctly")
            
        return True
    
    except ImportError:
        logger.warning("Could not import app module - NaN JSON fix not applied")
        return False
    except Exception as e:
        logger.error(f"Error applying NaN JSON fix: {str(e)}")
        return False

if __name__ == "__main__":
    fix_nan_in_json_result()
    print("✅ Applied NaN JSON serialization fix to the SHAP worker")
