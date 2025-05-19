#!/usr/bin/env python3
"""
Patch app.py with NaN JSON serialization fixes
This script will be run on Render during deployment
Created: May 16, 2025
"""

import os
import sys
import inspect
import importlib.util
from pathlib import Path

# Flag file to indicate patch has been applied
PATCH_FLAG = ".json_nan_fix_applied"

def load_module_from_file(file_path, module_name):
    """Load a module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    """Apply all fixes to running Python process"""
    print(f"Applying JSON NaN serialization fix...")
    
    # Check if fix has already been applied
    if os.path.exists(PATCH_FLAG):
        print(f"Fix already applied (flag file {PATCH_FLAG} exists)")
        return 0
        
    # Load and apply json serialization fix
    try:
        json_fix = load_module_from_file("json_serialization_fix.py", "json_serialization_fix")
        json_fix.apply_json_patches()
        print("✅ Applied JSON serialization patches")
        
        # Load and apply NaN JSON fix
        nan_fix = load_module_from_file("fix_nan_json.py", "fix_nan_json")
        nan_fix.fix_nan_in_json_result()
        print("✅ Applied NaN JSON fix")
        
        # Create flag file
        with open(PATCH_FLAG, "w") as f:
            f.write(f"JSON NaN serialization fix applied on {os.environ.get('RENDER_TIMESTAMP', 'unknown')}")
        
        return 0
    except Exception as e:
        print(f"Error applying fixes: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
