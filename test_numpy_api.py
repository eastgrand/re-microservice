#!/usr/bin/env python3
"""
Test script to verify NumPy API changes
"""
import numpy as np

def test_bool_types():
    """Test bool8 vs bool_ usage"""
    print("Testing NumPy boolean type APIs...")
    
    try:
        # Try to use bool8 (deprecated)
        bool_type = np.bool8
        print("np.bool8 is available")
    except Exception as e:
        print(f"np.bool8 error: {e}")
    
    try:
        # Use bool_ (recommended)
        bool_type = np.bool_
        print("np.bool_ is available")
        test_value = np.array([True, False], dtype=bool_type)
        print(f"Created array with bool_: {test_value}")
    except Exception as e:
        print(f"np.bool_ error: {e}")

def test_obj2sctype():
    """Test obj2sctype vs dtype().type usage"""
    print("\nTesting NumPy obj2sctype API...")
    
    try:
        # Try to use obj2sctype (deprecated)
        float_type = np.obj2sctype(float)
        print("np.obj2sctype is available")
        print(f"float type: {float_type}")
    except Exception as e:
        print(f"np.obj2sctype error: {e}")
    
    try:
        # Use dtype().type (recommended)
        float_type = np.dtype(float).type
        print("np.dtype().type is available")
        print(f"float type: {float_type}")
        
        # Test with an array
        test_array = np.array([1.0, 2.0], dtype=float_type)
        print(f"Created array with dtype().type: {test_array}")
    except Exception as e:
        print(f"np.dtype().type error: {e}")

if __name__ == "__main__":
    test_bool_types()
    test_obj2sctype()
    print("\nTest completed.")
