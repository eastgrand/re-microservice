#!/usr/bin/env python3
"""
Direct patch for SHAP library to fix numpy.bool8 and import issues.
This script doesn't import SHAP to avoid triggering the errors we're trying to fix.
"""

import os
import sys

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
    
    print(f"\nPatched {patched_files} files out of {len(patches)} files checked.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
