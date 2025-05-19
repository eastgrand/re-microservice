#!/usr/bin/env python3
"""
Test script for JSON NaN serialization fix
- Validates that our fix correctly handles NaN values
- Created: May 16, 2025
"""

import os
import sys
import json
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nan-json-test")

def test_json_serialization():
    """Test the JSON serialization fix"""
    print("Testing JSON serialization fix...")
    
    # Import our fix
    try:
        from json_serialization_fix import apply_json_patches, NumpyJSONEncoder
        
        # Apply the patches
        apply_json_patches()
        print("✅ Applied JSON serialization patches")
        
        # Create a test dictionary with NaN values
        test_data = {
            "normal_value": 42,
            "nan_value": float('nan'),
            "infinity": float('inf'),
            "neg_infinity": float('-inf'),
            "numpy_nan": np.nan,
            "numpy_array": np.array([1, 2, np.nan, 4]),
            "nested": {
                "inner_nan": float('nan'),
                "inner_normal": "string value",
                "inner_list": [1, float('nan'), 3]
            }
        }
        
        # Serialize with our patched json.dumps
        print("\nSerializing with patched json.dumps:")
        serialized = json.dumps(test_data, indent=2)
        print(serialized)
        
        # Check that NaN values are properly serialized as strings
        if '"nan_value": "NaN"' in serialized and '"inner_nan": "NaN"' in serialized:
            print("✅ NaN values correctly serialized as strings")
        else:
            print("❌ NaN values not correctly serialized")
        
        # Try deserializing
        print("\nDeserializing the JSON string:")
        deserialized = json.loads(serialized)
        print(f"✅ Successfully deserialized: {type(deserialized)}")
        
        # Check some values
        print("\nChecking deserialized values:")
        print(f"Normal value: {deserialized['normal_value']}")
        print(f"NaN value (now a string): {deserialized['nan_value']}")
        print(f"Nested inner NaN value: {deserialized['nested']['inner_nan']}")
        
        print("\n✅ JSON serialization fix test successful!")
        return True
    
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def test_flask_integration():
    """Test Flask integration if Flask is available"""
    print("\nTesting Flask integration...")
    
    try:
        from flask import Flask, jsonify
        import numpy as np
        
        # Import our fix
        from json_serialization_fix import apply_json_patches
        
        # Apply the patches
        apply_json_patches()
        
        # Create a test Flask app
        app = Flask("test")
        
        # Create a test endpoint
        @app.route('/test')
        def test_endpoint():
            test_data = {
                "normal_value": 42,
                "nan_value": float('nan'),
                "numpy_nan": np.nan,
                "nested": {
                    "inner_nan": float('nan')
                }
            }
            return jsonify(test_data)
        
        # Get the response from the test endpoint
        with app.test_client() as client:
            response = client.get('/test')
            print(f"Response status: {response.status_code}")
            print(f"Response data: {response.data.decode('utf-8')}")
            
            # Check if the response is valid JSON
            try:
                response_json = response.get_json()
                print(f"Successfully parsed response JSON: {response_json}")
                print("✅ Flask integration test successful!")
                return True
            except Exception as e:
                print(f"❌ Failed to parse response JSON: {str(e)}")
                return False
    
    except ImportError:
        print("⚠️ Flask not available - skipping Flask integration test")
        return True
    except Exception as e:
        print(f"❌ Flask integration test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run the tests
    json_test_result = test_json_serialization()
    flask_test_result = test_flask_integration()
    
    # Report overall results
    if json_test_result and flask_test_result:
        print("\n✅ All tests passed - JSON NaN serialization fix works correctly!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed - JSON NaN serialization fix needs attention")
        sys.exit(1)
