#!/usr/bin/env python3
"""
Comprehensive patch script for SHAP library to fix:
1. Missing pyspark fallback in _tree.py
2. Improved catboost import handling in _tree.py
3. NumPy bool8 to bool_ conversion
4. NumPy obj2sctype to dtype().type conversion
"""
import os
import re
import sys
import shutil
from pathlib import Path
from datetime import datetime

# Path to SHAP library in virtual environment
SHAP_PATH = Path("/Users/voldeck/code/shap-microservice/venv/lib/python3.13/site-packages/shap")

def create_backup(file_path):
    """Create a backup of the file before modifying it"""
    backup_path = f"{file_path}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    return backup_path

def fix_pyspark_import(tree_py_path):
    """Fix the pyspark import to handle ImportError properly"""
    with open(tree_py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if the fix is already applied
    if "pyspark = None" in content:
        print("Pyspark import fix already applied.")
        return False

    # Replace the pyspark import section
    pattern = r'try:\s+import pyspark\s+except ImportError as e:\s+record_import_error\("pyspark", "PySpark could not be imported!", e\)'
    replacement = 'try:\n    import pyspark\nexcept ImportError as e:\n    record_import_error("pyspark", "PySpark could not be imported!", e)\n    pyspark = None  # Added to prevent NameError'
    
    new_content = re.sub(pattern, replacement, content)
    if new_content != content:
        create_backup(tree_py_path)
        with open(tree_py_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Fixed pyspark import in _tree.py")
        return True
    else:
        print("Couldn't find the pyspark import pattern or it's already fixed.")
        return False

def fix_catboost_import(tree_py_path):
    """Improve the catboost import to use try/except with a flag"""
    with open(tree_py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if the fix is already applied
    if "catboost_imported = " in content:
        print("Catboost import fix already applied.")
        return False

    # Find and replace the catboost section
    pattern = r'import catboost\s+if type\(X\) != catboost.Pool:'
    replacement = 'try:\n                    import catboost\n                    catboost_imported = True\nexcept ImportError:\n                    catboost_imported = False\n                    \n                if catboost_imported and type(X) != catboost.Pool:'
    
    new_content = re.sub(pattern, replacement, content)
    if new_content != content:
        if not os.path.exists(f"{tree_py_path}.bak"):  # Don't create duplicate backup
            create_backup(tree_py_path)
        with open(tree_py_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print("Fixed catboost import in _tree.py")
        return True
    else:
        print("Couldn't find the catboost import pattern or it's already fixed.")
        return False

def fix_numpy_bool8(file_path):
    """Replace np.bool8 with np.bool_ in a file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    patterns = [
        ('np.bool8', 'np.bool_'),
        ('numpy.bool8', 'numpy.bool_')
    ]
    
    modified = False
    new_content = content
    for old, new in patterns:
        if old in new_content:
            new_content = new_content.replace(old, new)
            modified = True
    
    if modified:
        create_backup(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed numpy.bool8 references in {file_path}")
        return True
    else:
        print(f"No numpy.bool8 references found in {file_path}")
        return False

def fix_numpy_obj2sctype(file_path):
    """Replace np.obj2sctype with np.dtype().type in a file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'np\.obj2sctype\(([^)]+)\)'
    replacement = r'np.dtype(\1).type'
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:
        create_backup(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed np.obj2sctype references in {file_path}")
        return True
    else:
        print(f"No np.obj2sctype references found in {file_path}")
        return False

def find_files_with_pattern(directory, patterns):
    """Find all files containing any of the patterns"""
    matching_files = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern in patterns:
                            if pattern in content:
                                matching_files.add(file_path)
                                break
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return matching_files

def main():
    # Paths to specific files
    tree_py = SHAP_PATH / "explainers" / "_tree.py"
    colorconv_py = SHAP_PATH / "plots" / "colors" / "_colorconv.py"
    
    if not tree_py.exists():
        print(f"Error: {tree_py} does not exist!")
        return
    
    print("Starting comprehensive SHAP patch...")
    
    # Fix pyspark import in _tree.py
    fix_pyspark_import(tree_py)
    
    # Fix catboost import in _tree.py
    fix_catboost_import(tree_py)
    
    # Find files with numpy.bool8 references
    print("\nSearching for files with numpy.bool8 references...")
    bool8_files = find_files_with_pattern(SHAP_PATH, ['np.bool8', 'numpy.bool8'])
    for file_path in bool8_files:
        fix_numpy_bool8(file_path)
    
    # Find files with numpy.obj2sctype references
    print("\nSearching for files with numpy.obj2sctype references...")
    obj2sctype_files = find_files_with_pattern(SHAP_PATH, ['np.obj2sctype', 'numpy.obj2sctype'])
    for file_path in obj2sctype_files:
        fix_numpy_obj2sctype(file_path)
    
    print("\nSHAP library patching complete!")

if __name__ == "__main__":
    main()
