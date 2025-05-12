#!/usr/bin/env python3
"""
Fix NumPy obj2sctype deprecation in SHAP
"""

import os
import sys
import re
import glob

def find_shap_dir():
    """Find the SHAP package directory"""
    for path in sys.path:
        if 'site-packages' in path or 'dist-packages' in path:
            shap_dir = os.path.join(path, 'shap')
            if os.path.exists(shap_dir):
                return shap_dir
    return None

def main():
    shap_dir = find_shap_dir()
    if not shap_dir:
        print("Error: Could not find SHAP installation")
        return 1
    
    print(f"Found SHAP at: {shap_dir}")
    
    colorconv_path = os.path.join(shap_dir, 'plots', 'colors', '_colorconv.py')
    if not os.path.exists(colorconv_path):
        print(f"Error: Could not find _colorconv.py at {colorconv_path}")
        return 1
    
    print(f"Patching {colorconv_path}")
    
    try:
        # Read the file
        with open(colorconv_path, 'r') as f:
            content = f.read()
        
        # Replace np.obj2sctype with np.dtype().type
        if 'np.obj2sctype' in content:
            content = content.replace('np.obj2sctype(dtype)', 'np.dtype(dtype).type')
            content = content.replace('np.issubdtype(dtype_in, np.obj2sctype(dtype))', 
                                      'np.issubdtype(dtype_in, np.dtype(dtype).type)')
            
            # Write back the file
            with open(colorconv_path, 'w') as f:
                f.write(content)
            
            print("Successfully updated _colorconv.py to fix NumPy 2.0 compatibility")
            return 0
        else:
            print("No np.obj2sctype references found in _colorconv.py")
            return 1
    except Exception as e:
        print(f"Error patching file: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
