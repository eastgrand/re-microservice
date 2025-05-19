#!/usr/bin/env python3
# update_app_py.py
# Update app.py to include our fixes
# Created: May 16, 2025

import os
import sys
import re

def main():
    """Update app.py to include our fixes"""
    app_path = "app.py"
    
    try:
        # Read the current app.py
        with open(app_path, "r") as f:
            content = f.read()
        
        # Check if our fixes have already been applied
        if "import json_serialization_fix" in content:
            print("✅ app.py already includes our fixes")
            return 0
        
        # Find the imports section
        import_section_end = content.find("# Configure logging")
        if import_section_end == -1:
            import_section_end = content.find("app = Flask")
        
        if import_section_end == -1:
            print("❌ Could not find a good place to insert our imports")
            return 1
        
        # Create the patch to add
        patch = """
# Import JSON NaN serialization fix - Added May 16, 2025
import json_serialization_fix
json_serialization_fix.apply_json_patches()

# Import NaN JSON fix - Added May 16, 2025
import fix_nan_json
fix_nan_json.fix_nan_in_json_result()

"""
        
        # Insert the patch
        new_content = content[:import_section_end] + patch + content[import_section_end:]
        
        # Write back the updated file
        with open(app_path, "w") as f:
            f.write(new_content)
        
        print(f"✅ Successfully updated {app_path} with JSON NaN fixes")
        return 0
    except Exception as e:
        print(f"❌ Error updating {app_path}: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
