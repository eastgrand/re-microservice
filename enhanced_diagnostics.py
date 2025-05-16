#!/usr/bin/env python3
"""
Enhanced Job Processing Diagnostics
- Adds detailed logging to trace data processing and JSON serialization
- Created: May 16, 2025
"""

import os
import sys
import json
import logging
import time
import traceback
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diagnostic")

def install_diagnostic_hooks():
    """Install diagnostic hooks to trace job processing and data values"""
    try:
        # Import the app to modify it
        import app
        import numpy as np
        import json
        
        # Add diagnostic dump function
        def diagnostic_dump(data, prefix="DIAG"):
            """Dump diagnostic info about object"""
            try:
                if isinstance(data, dict):
                    # Log dict keys and some sample values
                    keys = list(data.keys())
                    sample_values = {}
                    for k in keys[:3]:  # Sample first 3 keys
                        v = data[k]
                        if isinstance(v, (dict, list)):
                            sample_values[k] = f"{type(v).__name__} with {len(v)} items"
                        else:
                            sample_values[k] = f"{type(v).__name__}: {str(v)[:50]}..."
                    
                    logger.info(f"{prefix}: Dict with {len(keys)} keys. Keys: {', '.join(keys[:10])} {'...' if len(keys) > 10 else ''}")
                    logger.info(f"{prefix}: Sample values: {sample_values}")
                    
                    # Check for NaN/Infinity values
                    has_nan = False
                    nan_keys = []
                    for k, v in data.items():
                        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                            has_nan = True
                            nan_keys.append(k)
                        elif isinstance(v, dict):
                            for k2, v2 in v.items():
                                if isinstance(v2, float) and (np.isnan(v2) or np.isinf(v2)):
                                    has_nan = True
                                    nan_keys.append(f"{k}.{k2}")
                    
                    if has_nan:
                        logger.warning(f"{prefix}: ⚠️ Contains NaN/Infinity values in keys: {nan_keys}")
                
                elif isinstance(data, list):
                    logger.info(f"{prefix}: List with {len(data)} items")
                    if len(data) > 0:
                        logger.info(f"{prefix}: First item type: {type(data[0]).__name__}")
                        
                        # Check if list contains dicts with potential NaN values
                        if isinstance(data[0], dict):
                            for i, item in enumerate(data[:5]):  # Check first 5 items
                                has_nan = False
                                nan_keys = []
                                for k, v in item.items():
                                    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                                        has_nan = True
                                        nan_keys.append(k)
                                if has_nan:
                                    logger.warning(f"{prefix}: ⚠️ Item {i} contains NaN in keys: {nan_keys}")
                else:
                    logger.info(f"{prefix}: {type(data).__name__}: {str(data)[:100]}")
                
                # Try JSON serialization
                try:
                    json_str = json.dumps(data)
                    logger.info(f"{prefix}: ✅ Successfully serialized to JSON ({len(json_str)} chars)")
                except Exception as e:
                    logger.error(f"{prefix}: ❌ JSON serialization failed: {str(e)}")
                    
            except Exception as e:
                logger.error(f"{prefix}: Error during diagnostic: {str(e)}")
        
        # Add to app module
        app.diagnostic_dump = diagnostic_dump
        
        # Patch analysis_worker to add diagnostics
        if hasattr(app, 'analysis_worker'):
            original_analysis_worker = app.analysis_worker
            
            @wraps(original_analysis_worker)
            def diagnostic_analysis_worker(*args, **kwargs):
                """Wrapped analysis_worker with diagnostics"""
                try:
                    logger.info(f"DIAG: Starting analysis_worker with args: {args}")
                    if len(args) > 0:
                        diagnostic_dump(args[0], prefix="INPUT")
                    
                    # Call original function
                    start_time = time.time()
                    result = original_analysis_worker(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Log result
                    logger.info(f"DIAG: analysis_worker completed in {duration:.2f}s")
                    diagnostic_dump(result, prefix="OUTPUT")
                    
                    return result
                except Exception as e:
                    logger.error(f"DIAG: analysis_worker failed: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Re-raise the exception
                    raise
            
            # Replace with diagnostic version
            app.analysis_worker = diagnostic_analysis_worker
            logger.info("✅ Installed diagnostic hooks for analysis_worker")
        
        # Patch job_status endpoint
        if hasattr(app, 'job_status'):
            original_job_status = app.job_status
            
            @wraps(original_job_status)
            def diagnostic_job_status(job_id):
                """Wrapped job_status with diagnostics"""
                try:
                    logger.info(f"DIAG: Getting status for job {job_id}")
                    
                    # Call original function
                    result = original_job_status(job_id)
                    
                    # Log response data
                    if hasattr(result, 'get_json'):
                        try:
                            data = result.get_json()
                            logger.info(f"DIAG: Job status response data:")
                            diagnostic_dump(data, prefix="JOB_STATUS")
                        except Exception as e:
                            logger.error(f"DIAG: Error getting JSON from response: {str(e)}")
                    
                    # Log raw response
                    if hasattr(result, 'get_data'):
                        try:
                            raw_data = result.get_data(as_text=True)
                            logger.info(f"DIAG: Raw response (first 200 chars): {raw_data[:200]}...")
                            
                            # Check for "NaN" without quotes
                            if 'NaN' in raw_data and '"NaN"' not in raw_data:
                                logger.error(f"DIAG: ⚠️ Found unquoted NaN in response!")
                                
                                # Fix it on the fly
                                import re
                                fixed_data = re.sub(r':\s*NaN\s*,', ':"NaN",', raw_data)
                                fixed_data = re.sub(r':\s*NaN\s*}', ':"NaN"}', fixed_data)
                                logger.info(f"DIAG: Fixed response (first 200 chars): {fixed_data[:200]}...")
                                
                                # Try parsing the fixed data
                                try:
                                    json.loads(fixed_data)
                                    logger.info("DIAG: ✅ Fixed data is valid JSON")
                                except Exception as e:
                                    logger.error(f"DIAG: ❌ Fixed data is still invalid JSON: {str(e)}")
                            
                        except Exception as e:
                            logger.error(f"DIAG: Error getting raw data from response: {str(e)}")
                    
                    return result
                except Exception as e:
                    logger.error(f"DIAG: job_status failed: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Re-raise the exception
                    raise
            
            # Replace with diagnostic version
            app.job_status = diagnostic_job_status
            logger.info("✅ Installed diagnostic hooks for job_status")
        
        # Enhance JSON serialization
        import json
        original_dumps = json.dumps
        
        @wraps(original_dumps)
        def diagnostic_dumps(*args, **kwargs):
            """Wrapped json.dumps with diagnostics"""
            try:
                result = original_dumps(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"DIAG: json.dumps failed: {str(e)}")
                if len(args) > 0:
                    logger.info("DIAG: Examining data that failed to serialize:")
                    
                    # Check for NaN values
                    try:
                        import numpy as np
                        data = args[0]
                        
                        if isinstance(data, dict):
                            nan_entries = [(k, v) for k, v in data.items() 
                                          if isinstance(v, float) and (np.isnan(v) or np.isinf(v))]
                            if nan_entries:
                                logger.warning(f"DIAG: Found NaN/Infinity entries: {nan_entries}")
                        
                        # Try saving invalid data to file for inspection
                        with open('failed_json_data.txt', 'w') as f:
                            f.write(str(data))
                        logger.info("DIAG: Saved problematic data to failed_json_data.txt")
                    except:
                        pass
                
                # Re-raise the exception
                raise
        
        # Replace the json.dumps function
        json.dumps = diagnostic_dumps
        logger.info("✅ Installed diagnostic hooks for json.dumps")
        
        return True
    except Exception as e:
        logger.error(f"Failed to install diagnostic hooks: {str(e)}")
        return False

if __name__ == "__main__":
    print("Installing diagnostic hooks...")
    install_diagnostic_hooks()
    print("Diagnostic hooks installed. Check logs for detailed processing information.")
