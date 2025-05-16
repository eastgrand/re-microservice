#!/usr/bin/env python3
"""
RQ (Redis Queue) Serialization Fix
- Directly patches RQ's serialization to handle NaN values properly
- Created: May 16, 2025
"""

import os
import sys
import logging
import traceback
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rq-fix")

def patch_rq_serialization():
    """Patch RQ's serialization to handle NaN values properly"""
    try:
        import rq
        import json
        import pickle
        import numpy as np
        
        logger.info("Patching RQ serialization...")
        
        # Define our custom JSON encoder for RQ
        class NanSafeJSONEncoder(json.JSONEncoder):
            """JSON encoder that safely handles NaN and Infinity values"""
            def default(self, obj):
                if isinstance(obj, (float, np.float32, np.float64)):
                    if np.isnan(obj):
                        return "NaN"  # Use string representation for NaN
                    elif np.isinf(obj) and obj > 0:
                        return "Infinity"  # Use string representation for +Infinity
                    elif np.isinf(obj) and obj < 0:
                        return "-Infinity"  # Use string representation for -Infinity
                elif isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return super().default(obj)
        
        # Patch RQ's dumps function
        if hasattr(rq.job, 'dumps'):
            original_dumps = rq.job.dumps
            
            @wraps(original_dumps)
            def safe_dumps(obj):
                """Safe version of dumps that handles NaN values"""
                try:
                    return original_dumps(obj)
                except Exception as e:
                    logger.warning(f"RQ dumps failed with standard encoder: {str(e)}")
                    try:
                        # Try with our custom JSON encoder
                        if isinstance(obj, dict):
                            logger.info("Using custom JSON encoder for job data")
                            return json.dumps(obj, cls=NanSafeJSONEncoder)
                        else:
                            # Fall back to pickle for complex objects
                            logger.info("Using pickle for complex job data")
                            return pickle.dumps(obj)
                    except Exception as e2:
                        logger.error(f"Custom encoder failed too: {str(e2)}")
                        # Re-raise the original exception
                        raise e
            
            # Replace the original dumps function
            rq.job.dumps = safe_dumps
            logger.info("✅ Patched rq.job.dumps with NaN-safe version")
        else:
            logger.warning("⚠️ rq.job.dumps not found - serialization not patched")
        
        # Patch RQ's loads function
        if hasattr(rq.job, 'loads'):
            original_loads = rq.job.loads
            
            @wraps(original_loads)
            def safe_loads(data):
                """Safe version of loads that handles special string values"""
                result = original_loads(data)
                
                # Convert string representations back to special values
                if isinstance(result, dict):
                    for k, v in result.items():
                        if v == "NaN":
                            result[k] = float('nan')
                        elif v == "Infinity":
                            result[k] = float('inf')
                        elif v == "-Infinity":
                            result[k] = float('-inf')
                
                return result
            
            # Replace the original loads function
            rq.job.loads = safe_loads
            logger.info("✅ Patched rq.job.loads with special value handler")
        else:
            logger.warning("⚠️ rq.job.loads not found - deserialization not patched")
        
        # Patch Status class if it exists (to handle result serialization)
        if hasattr(rq, 'job') and hasattr(rq.job, 'Job'):
            original_status = rq.job.Job.status
            
            @property
            def safe_status(self):
                """Safe version of status property that handles serialization errors"""
                try:
                    return original_status.fget(self)
                except Exception as e:
                    logger.error(f"Error getting job status: {str(e)}")
                    # Try to fix or return a safe value
                    return 'failed'
            
            # Replace the property
            rq.job.Job.status = property(safe_status)
            logger.info("✅ Patched rq.job.Job.status with error handling")
            
            # Also patch get_status method if it exists
            if hasattr(rq.job.Job, 'get_status'):
                original_get_status = rq.job.Job.get_status
                
                @wraps(original_get_status)
                def safe_get_status(self):
                    """Safe version of get_status that handles serialization errors"""
                    try:
                        return original_get_status(self)
                    except Exception as e:
                        logger.error(f"Error in get_status: {str(e)}")
                        # Try to fix or return a safe value
                        return 'failed'
                
                # Replace the method
                rq.job.Job.get_status = safe_get_status
                logger.info("✅ Patched rq.job.Job.get_status with error handling")
        
        # Globally replace json.dumps and json.loads in the rq package
        for module_name in sys.modules:
            if module_name.startswith('rq.'):
                module = sys.modules[module_name]
                if hasattr(module, 'json'):
                    # Replace the json module's dumps function
                    def create_safe_dumps(original_fn):
                        @wraps(original_fn)
                        def safe_fn(obj, *args, **kwargs):
                            try:
                                # First try with cls=NanSafeJSONEncoder
                                if 'cls' not in kwargs:
                                    kwargs['cls'] = NanSafeJSONEncoder
                                return original_fn(obj, *args, **kwargs)
                            except:
                                # Fall back to standard encoder
                                kwargs.pop('cls', None)
                                return original_fn(obj, *args, **kwargs)
                        return safe_fn
                    
                    # Replace dumps function
                    module.json.dumps = create_safe_dumps(module.json.dumps)
                    logger.info(f"✅ Patched json.dumps in {module_name}")
        
        logger.info("✅ RQ serialization patching complete")
        return True
    except Exception as e:
        logger.error(f"Failed to patch RQ serialization: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Demo parser for debugging
def test_nan_serialization():
    """Test NaN serialization fix"""
    import json
    import numpy as np
    
    # Test data with NaN values
    test_data = {
        "value1": 1.0,
        "value2": float('nan'),
        "value3": float('inf'),
        "nested": {
            "nestedValue1": 2.0,
            "nestedValue2": float('nan')
        }
    }
    
    print("Original data:", test_data)
    print("value2 is NaN:", np.isnan(test_data["value2"]))
    print("value3 is Infinity:", np.isinf(test_data["value3"]))
    
    # Try standard JSON encoder
    try:
        json_str = json.dumps(test_data)
        print("Standard JSON encoder succeeded?", True)
        print("JSON string:", json_str)
    except Exception as e:
        print("Standard JSON encoder failed:", str(e))
    
    # Try with NanSafeJSONEncoder
    try:
        class NanSafeJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, float) and np.isnan(obj):
                    return "NaN"
                elif isinstance(obj, float) and np.isinf(obj):
                    return "Infinity" if obj > 0 else "-Infinity"
                return super().default(obj)
        
        json_str = json.dumps(test_data, cls=NanSafeJSONEncoder)
        print("Custom JSON encoder succeeded:", True)
        print("JSON string:", json_str)
        
        # Try parsing it back
        parsed = json.loads(json_str)
        print("Parsed back:", parsed)
        print("NaN is now:", parsed["value2"])
        
        # Verify fix
        if parsed["value2"] == "NaN" and parsed["nested"]["nestedValue2"] == "NaN":
            print("✅ NaN serialization fix works!")
        else:
            print("❌ NaN serialization fix fails!")
    except Exception as e:
        print("Custom JSON encoder failed:", str(e))

if __name__ == "__main__":
    # Test serialization
    print("Testing NaN serialization fix...")
    test_nan_serialization()
    
    # Apply patches
    print("\nPatching RQ serialization...")
    patch_rq_serialization()
    print("RQ serialization patched")
