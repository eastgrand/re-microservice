#!/usr/bin/env python3
"""
This script patches the SHAP library to handle missing imports and API changes gracefully.
It ensures that even when PySpark and CatBoost are not installed, and handles numpy API changes.
"""

import os
import re
import sys

def direct_file_edit(file_path, search_text, replace_text):
    """
    Directly edit a file by searching for text and replacing it.
    Avoids importing problematic modules.
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return False
        
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        if search_text in content:
            new_content = content.replace(search_text, replace_text)
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"Successfully patched: {file_path}")
            return True
        else:
            print(f"Search text not found in {file_path}")
            return False
    except Exception as e:
        print(f"Error patching {file_path}: {e}")
        return False

def find_shap_base_dir():
    """Find the base directory of the SHAP package"""
    try:
        import shap
        return os.path.dirname(os.path.abspath(shap.__file__))
    except ImportError:
        print("Error: Could not import shap. Is it installed?")
        return None

def main():
    # Find the SHAP package directory
    shap_dir = find_shap_base_dir()
    if not shap_dir:
        return 1
    
    patches_applied = 0
    
    # 1. Fix the numpy.bool8 issue in _colorconv.py
    colorconv_path = os.path.join(shap_dir, 'plots', 'colors', '_colorconv.py')
    if direct_file_edit(colorconv_path, 
                        'np.bool8: (False, True),',
                        'np.bool_: (False, True),  # Changed from bool8 to bool_'):
        patches_applied += 1
    
    # 2. Fix the pyspark import in _tree.py
    tree_path = os.path.join(shap_dir, 'explainers', '_tree.py')
    if direct_file_edit(tree_path,
                       "try:\n    import pyspark\nexcept ImportError as e:\n    record_import_error(\"pyspark\", \"PySpark could not be imported!\", e)",
                       "try:\n    import pyspark\nexcept ImportError as e:\n    record_import_error(\"pyspark\", \"PySpark could not be imported!\", e)\n    pyspark = None"):
        patches_applied += 1
    
    # 3. Fix the catboost import in _tree.py
    if direct_file_edit(tree_path,
                       "import catboost\n                if type(X) != catboost.Pool:",
                       "try:\n                    import catboost\n                    catboost_imported = True\n                except ImportError:\n                    catboost_imported = False\n                    \n                if catboost_imported and type(X) != catboost.Pool:"):
        patches_applied += 1
    
    print(f"\nTotal patches applied: {patches_applied}")
    if patches_applied > 0:
        print("SHAP patching completed successfully.")
    else:
        print("No patches were applied - files may already be patched or issues not found.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
