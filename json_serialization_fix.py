#!/usr/bin/env python3
"""
JSON Serialization Fix for SHAP Microservice
- Handles NaN, Infinity, and -Infinity values in JSON serialization
- Created: May 16, 2025
"""

import json
import numpy as np
import math
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("json-fix")

class NumpyJSONEncoder(json.JSONEncoder):
    """Enhanced JSON encoder that properly handles numpy types and special values like NaN"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            if np.isnan(obj):
                return "NaN"  # Use string representation for NaN
            elif np.isinf(obj) and obj > 0:
                return "Infinity"  # Use string representation for +Infinity
            elif np.isinf(obj) and obj < 0:
                return "-Infinity"  # Use string representation for -Infinity
            else:
                return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif obj is None:
            return None
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        return super().default(obj)

def patch_json_serialization():
    """Patch JSON serialization to handle NaN values"""
    # Store original dumps method
    original_dumps = json.dumps
    
    @wraps(original_dumps)
    def safe_dumps(*args, **kwargs):
        """Enhanced version of json.dumps that handles NaN values"""
        # Use our custom encoder by default
        if 'cls' not in kwargs:
            kwargs['cls'] = NumpyJSONEncoder
        return original_dumps(*args, **kwargs)
    
    # Replace the original dumps with our patched version
    json.dumps = safe_dumps
    
    # Patch dumps
    logger.info("Patched json.dumps with NaN-safe serializer")
    
    # Also patch Flask's jsonify if it's available
    try:
        from flask import jsonify as flask_jsonify
        import flask.json
        
        # Store original jsonify
        original_jsonify = flask_jsonify
        
        @wraps(original_jsonify)
        def safe_jsonify(*args, **kwargs):
            """Enhanced version of flask.jsonify that handles NaN values"""
            # Ensure the default JSON encoder is our NaN-safe encoder
            flask.json.jsonify = original_jsonify
            flask.json.JSONEncoder = NumpyJSONEncoder
            return original_jsonify(*args, **kwargs)
        
        # Replace flask's jsonify
        flask.jsonify = safe_jsonify
        flask.json.JSONEncoder = NumpyJSONEncoder
        
        logger.info("Patched Flask's jsonify with NaN-safe serializer")
    except ImportError:
        logger.info("Flask not found, skipping Flask jsonify patch")
    
    return True

def decode_special_json_values(obj):
    """Decode special JSON values like 'NaN', 'Infinity', '-Infinity'"""
    if isinstance(obj, dict):
        return {k: decode_special_json_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_special_json_values(item) for item in obj]
    elif obj == "NaN":
        return float('nan')
    elif obj == "Infinity":
        return float('inf')
    elif obj == "-Infinity":
        return float('-inf')
    else:
        return obj

def patch_json_deserialization():
    """Patch JSON deserialization to handle special values"""
    # Store original loads method
    original_loads = json.loads
    
    @wraps(original_loads)
    def safe_loads(*args, **kwargs):
        """Enhanced version of json.loads that handles special values"""
        # First use the original loads
        result = original_loads(*args, **kwargs)
        # Then decode special values
        return decode_special_json_values(result)
    
    # Replace the original loads with our patched version
    json.loads = safe_loads
    logger.info("Patched json.loads with special value decoder")
    
    return True

def apply_json_patches():
    """Apply all JSON-related patches"""
    patch_json_serialization()
    patch_json_deserialization()
    logger.info("All JSON serialization patches applied successfully")
    return True

if __name__ == "__main__":
    print("This is a patch module for the SHAP microservice JSON serialization.")
    print("To use, import this module in your main app.py file and call apply_json_patches()")
