#!/usr/bin/env python
"""
SHAP Compatibility Patcher

This script patches the SHAP library to work with newer versions of NumPy
by replacing deprecated np.int references with np.int32.
"""

import os
import re
import sys
import glob
import shutil
import tempfile
from pathlib import Path


def find_shap_package_dir():
    """Find the SHAP package directory."""
    try:
        import shap
        return os.path.dirname(shap.__file__)
    except ImportError:
        print("SHAP package not found. Please install it first.")
        sys.exit(1)


def patch_file(file_path):
    """Replace np.int with np.int32 in a file and fix import issues."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Replace np.int with np.int32 but only when it's used as a dtype
    patched_content = re.sub(r'dtype=np\.int\)', r'dtype=np.int32)', content)
    patched_content = re.sub(r'dtype=np\.int,', r'dtype=np.int32,', patched_content)
    patched_content = re.sub(r'dtype=np\.int$', r'dtype=np.int32', patched_content, flags=re.MULTILINE)
    
    # Fix missing imports in _tree.py
    if os.path.basename(file_path) == '_tree.py':
        # Fix pyspark import
        patched_content = re.sub(
            r'try:\s*import pyspark\s*except ImportError as e:',
            r'try:\n    import pyspark\nexcept ImportError as e:\n    pyspark = None',
            patched_content
        )
        
        # Fix catboost import (anywhere in the file)
        patched_content = patched_content.replace(
            'import catboost',
            'try:\n    import catboost\nexcept ImportError:\n    catboost = None'
        )
    
    # Only write back if changes were made
    if content != patched_content:
        # Create backup
        backup_path = file_path + '.bak'
        shutil.copy2(file_path, backup_path)
        
        # Write patched content
        with open(file_path, 'w') as f:
            f.write(patched_content)
        
        return True
    
    return False


def patch_shap():
    """Patch all SHAP Python files."""
    shap_dir = find_shap_package_dir()
    print(f"Found SHAP package at: {shap_dir}")
    
    # Find all Python files in SHAP package
    py_files = glob.glob(os.path.join(shap_dir, '**', '*.py'), recursive=True)
    
    patched_files = 0
    for py_file in py_files:
        if patch_file(py_file):
            print(f"Patched: {py_file}")
            patched_files += 1
    
    print(f"\nDone! Patched {patched_files} files.")
    print(f"Backups created with .bak extension.")


if __name__ == "__main__":
    patch_shap()
