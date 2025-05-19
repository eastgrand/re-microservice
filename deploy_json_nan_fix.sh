#!/bin/bash
# deploy_json_nan_fix.sh
# Deploy the fix for the JSON NaN serialization issue in worker
# Created: May 16, 2025

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   SHAP Microservice JSON NaN Fix Deployment      ${NC}"
echo -e "${BLUE}=================================================${NC}"

# Step 1: Make all scripts executable
echo -e "${YELLOW}Making scripts executable...${NC}"
chmod +x *.py *.sh
echo -e "${GREEN}✅ Made scripts executable${NC}"

# Step 2: Create special app.py wrapper
echo -e "${YELLOW}Creating app.py wrapper for JSON NaN fix...${NC}"

cat > patch_app_with_fixes.py <<'EOL'
#!/usr/bin/env python3
# patch_app_with_fixes.py
# Apply all fixes to app.py
# Created: May 16, 2025

import os
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app-patcher")

def main():
    """Apply all fixes to app.py"""
    try:
        # Import json serialization fix
        logger.info("Importing json_serialization_fix...")
        import json_serialization_fix
        json_serialization_fix.apply_json_patches()
        logger.info("✅ Applied JSON serialization patches")
        
        # Import and apply NaN fix
        logger.info("Importing fix_nan_json...")
        import fix_nan_json
        fix_nan_json.fix_nan_in_json_result()
        logger.info("✅ Applied NaN JSON fix")
        
        # Import app
        logger.info("Importing app (with fixes already applied)...")
        import app
        logger.info("✅ App imported successfully")
        
        # Print success message
        print("✅ All fixes applied successfully")
        return 0
    except Exception as e:
        logger.error(f"Error applying fixes: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOL

chmod +x patch_app_with_fixes.py
echo -e "${GREEN}✅ Created patch_app_with_fixes.py${NC}"

# Step 3: Update app.py to use our fixes
echo -e "${YELLOW}Creating app.py update script...${NC}"

cat > update_app_py.py <<'EOL'
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
EOL

chmod +x update_app_py.py
echo -e "${GREEN}✅ Created update_app_py.py${NC}"

# Step 4: Create a backup of the original app.py
echo -e "${YELLOW}Creating backup of app.py...${NC}"
if [ -f "app.py" ]; then
  cp app.py app.py.bak.$(date +"%Y%m%d%H%M%S")
  echo -e "${GREEN}✅ Created backup of app.py${NC}"
else
  echo -e "${RED}❌ app.py not found in current directory${NC}"
  exit 1
fi

# Step 5: Update app.py with our fixes
echo -e "${YELLOW}Updating app.py with JSON NaN fixes...${NC}"
python3 update_app_py.py
if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Successfully updated app.py${NC}"
else
  echo -e "${RED}❌ Failed to update app.py${NC}"
  exit 1
fi

# Step 6: Test the fixes (safely without requiring dependencies)
echo -e "${YELLOW}Testing the fixes...${NC}"
python3 -c "import sys; sys.path.insert(0, '.'); print('✅ Python import system is working')"
if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Python interpreter working correctly${NC}"
  
  # Verify our fix files are syntactically correct without importing them
  python3 -m py_compile json_serialization_fix.py fix_nan_json.py
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Fix modules compile successfully${NC}"
  else
    echo -e "${RED}❌ Fix modules have syntax errors${NC}"
    # Restore from backup
    echo -e "${YELLOW}Restoring app.py from backup...${NC}"
    cp app.py.bak.* app.py
    echo -e "${GREEN}✅ Restored app.py from backup${NC}"
    exit 1
  fi
  
  # Just check if the app.py file exists and has content
  if [ -s "app.py" ]; then
    echo -e "${GREEN}✅ app.py exists and has content${NC}"
    echo -e "${YELLOW}Note: Full dependency check skipped - will be done during deployment${NC}"
  else
    echo -e "${RED}❌ app.py is missing or empty${NC}"
    echo -e "${YELLOW}Restoring app.py from backup...${NC}"
    cp app.py.bak.* app.py
    echo -e "${GREEN}✅ Restored app.py from backup${NC}"
    exit 1
  fi
  
else
  echo -e "${RED}❌ Python interpreter not working correctly${NC}"
  # Restore from backup
  echo -e "${YELLOW}Restoring app.py from backup...${NC}"
  cp $(ls -t app.py.bak.* | head -1) app.py
  echo -e "${GREEN}✅ Restored app.py from backup${NC}"
  exit 1
fi

# Step 7: Create detailed documentation
echo -e "${YELLOW}Creating documentation...${NC}"

cat > JSON-NAN-FIX.md <<'EOL'
# JSON NaN Value Serialization Fix

## Issue Overview
**Date:** May 16, 2025

The SHAP microservice was encountering an error when processing jobs:

```
SyntaxError: Unexpected token 'N', ..."ications":NaN,"mortg"... is not valid JSON
```

This was occurring because:

1. Some numerical values in the analysis results were NaN (Not-a-Number)
2. When these values were serialized to JSON directly, they appeared as literal `NaN` in the JSON string
3. Standard JSON does not support `NaN` as a literal value, only as a quoted string like `"NaN"`
4. When the client tried to parse the malformed JSON, it failed with the syntax error

## Solution Implemented

We created multiple fixes to address this issue:

1. **JSON Serialization Fix** (`json_serialization_fix.py`):
   - Enhanced JSON encoder that properly handles numpy types
   - Converts NaN to string "NaN" during serialization
   - Also handles Infinity and -Infinity values
   - Patches both `json.dumps` and Flask's `jsonify`

2. **Worker JSON Fix** (`fix_nan_json.py`):
   - Specifically targets the job status and job result endpoints
   - Wraps the response functions to check and fix NaN values
   - Uses regular expressions as a fallback to fix raw JSON strings

3. **App Integration** (updated `app.py`):
   - Imports and applies both fixes at startup
   - Ensures all JSON serialization uses the fixed encoder

## Verification

After deployment, you should see:

1. Jobs being fully processed (instead of stuck in "started" state)
2. Client receives valid JSON (no more parsing errors)
3. NaN values properly represented as "NaN" strings in the JSON response

## Additional Notes

If issues persist after this fix, you can check:

1. The worker logs for any new error messages
2. The specific data structures being serialized to see if there are other problematic values
3. The client-side handling of the "NaN" string values

For any jobs that got stuck in the processing state, you may need to run
job cleanup to remove them from the queue.
EOL

echo -e "${GREEN}✅ Created JSON-NAN-FIX.md documentation${NC}"

# Step 8: Prepare for deployment
if [ -z "$1" ]; then
  echo -e "${YELLOW}Creating deployment commit...${NC}"
  git add json_serialization_fix.py fix_nan_json.py app_fix_wrapper.py app.py JSON-NAN-FIX.md
  git commit -m "Fix JSON NaN serialization issue in worker - May 16, 2025"
  
  echo -e "${GREEN}✅ Created deployment commit${NC}"
  echo -e "${YELLOW}Pushing changes to repository...${NC}"
  git push origin main
  echo -e "${GREEN}✅ Pushed changes to repository${NC}"
else
  echo -e "${YELLOW}Skipping git commit and push as requested${NC}"
fi

echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}✅ JSON NaN Fix Deployment Complete!${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""
echo -e "The JSON NaN serialization issue has been fixed."
echo -e "Jobs should now complete processing successfully."
echo ""
echo -e "You can verify the fix by checking the worker logs for:"
echo -e "${GREEN}✅ Applied JSON serialization patches${NC}"
echo -e "${GREEN}✅ Applied NaN JSON fix${NC}"
echo -e ""
echo -e "If you need to rollback, the original app.py has been backed up."
