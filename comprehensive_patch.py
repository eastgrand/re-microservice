#!/usr/bin/env python3
"""
Comprehensive patch for SHAP library to fix various NumPy compatibility issues
"""

import os
import sys
import re

def find_site_packages_dir():
    """Find the site-packages directory where SHAP is installed"""
    for path in sys.path:
        if 'site-packages' in path or 'dist-packages' in path:
            shap_init = os.path.join(path, 'shap', '__init__.py')
            if os.path.exists(shap_init):
                return os.path.join(path, 'shap')
    return None

def patch_file(filepath, replacements):
    """Apply multiple text replacements to a single file"""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return False
    
    try:
        with open(filepath, 'r') as file:
            content = file.read()
        
        original_content = content
        for old_text, new_text in replacements:
            content = content.replace(old_text, new_text)
        
        if content != original_content:
            with open(filepath, 'w') as file:
                file.write(content)
            print(f"Successfully patched: {filepath}")
            return True
        else:
            print(f"No changes needed for: {filepath}")
            return False
    except Exception as e:
        print(f"Error patching {filepath}: {str(e)}")
        return False

def patch_numpy_obj2sctype(filepath):
    """Replace np.obj2sctype with np.dtype(obj).type"""
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return False
    
    try:
        with open(filepath, 'r') as file:
            content = file.read()
        
        # Look for np.obj2sctype usage and replace with np.dtype().type
        pattern = r'np\.obj2sctype\(([^)]+)\)'
        replacement = r'np.dtype(\1).type'
        
        new_content = re.sub(pattern, replacement, content)
        
        if new_content != content:
            with open(filepath, 'w') as file:
                file.write(new_content)
            print(f"Successfully patched np.obj2sctype in: {filepath}")
            return True
        else:
            print(f"No np.obj2sctype found in: {filepath}")
            return False
    except Exception as e:
        print(f"Error patching {filepath}: {str(e)}")
        return False

def main():
    shap_dir = find_site_packages_dir()
    if not shap_dir:
        print("Error: Could not find SHAP installation directory.")
        return 1
    
    print(f"Found SHAP installation at: {shap_dir}")
    
    # Define all the patches we want to apply
    patches = [
        {
            'file': os.path.join(shap_dir, 'plots', 'colors', '_colorconv.py'),
            'replacements': [
                ('np.bool8: (False, True),', 'np.bool_: (False, True),  # Changed from bool8 to bool_')
            ]
        },
        {
            'file': os.path.join(shap_dir, 'explainers', '_tree.py'),
            'replacements': [
                (
                    "try:\n    import pyspark\nexcept ImportError as e:\n    record_import_error(\"pyspark\", \"PySpark could not be imported!\", e)",
                    "try:\n    import pyspark\nexcept ImportError as e:\n    record_import_error(\"pyspark\", \"PySpark could not be imported!\", e)\n    pyspark = None"
                ),
                (
                    "import catboost\n                if type(X) != catboost.Pool:",
                    "try:\n                    import catboost\n                    catboost_imported = True\n                except ImportError:\n                    catboost_imported = False\n                    \n                if catboost_imported and type(X) != catboost.Pool:"
                )
            ]
        }
    ]
    
    # Apply all patches
    patched_files = 0
    for patch in patches:
        if patch_file(patch['file'], patch['replacements']):
            patched_files += 1
    
    # Fix the obj2sctype issue
    colorconv_path = os.path.join(shap_dir, 'plots', 'colors', '_colorconv.py')
    if patch_numpy_obj2sctype(colorconv_path):
        patched_files += 1
    
    print(f"\nPatched {patched_files} files.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
