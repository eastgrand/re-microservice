#!/bin/bash
# deploy_json_nan_fix_direct.sh
# Deploy the fix for JSON NaN serialization - Direct deployment version
# Created: May 16, 2025

set -e  # Exit on any error

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   SHAP Microservice JSON NaN Fix - DIRECT DEPLOY  ${NC}"
echo -e "${BLUE}=================================================${NC}"

# Step 1: Make all scripts executable
echo -e "${YELLOW}Making scripts executable...${NC}"
chmod +x *.py *.sh 2>/dev/null || true
echo -e "${GREEN}✅ Made scripts executable${NC}"

# Step 2: Just create the fix files without trying to modify app.py
echo -e "${YELLOW}Creating fix files directly...${NC}"

# Step 3: Update app.py on Render, not locally
echo -e "${YELLOW}Creating app.py modifier...${NC}"

cat > patch_app_with_fixes.py <<'EOL'
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
EOL

chmod +x patch_app_with_fixes.py
echo -e "${GREEN}✅ Created patch_app_with_fixes.py${NC}"

# Create Render startup wrapper
echo -e "${YELLOW}Creating Render startup wrapper...${NC}"

cat > render_startup_wrapper.sh <<'EOL'
#!/bin/bash
# Wrapper script to apply JSON NaN serialization fix before starting the app
# This will be used in render.yaml

# Apply the JSON NaN serialization fix
echo "Applying JSON NaN serialization fix..."
python3 patch_app_with_fixes.py

# Start the app with the original command
echo "Starting app with original command..."
exec "$@"
EOL

chmod +x render_startup_wrapper.sh
echo -e "${GREEN}✅ Created render_startup_wrapper.sh${NC}"

# Create updated render.yaml file
echo -e "${YELLOW}Creating updated render.yaml with fix...${NC}"

# Create a backup of the original render.yaml
if [ -f "render.yaml" ]; then
  cp render.yaml render.yaml.bak.$(date +"%Y%m%d%H%M%S")
  echo -e "${GREEN}✅ Created backup of render.yaml${NC}"
  
  # Update the startCommand to use our wrapper
  sed -i.bak 's/startCommand: >-/startCommand: >-\n      .\/render_startup_wrapper.sh /' render.yaml
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Updated render.yaml${NC}"
  else
    echo -e "${RED}❌ Failed to update render.yaml${NC}"
    # Continue anyway as we'll provide manual instructions
  fi
else
  echo -e "${YELLOW}⚠️ render.yaml not found - will provide manual instructions${NC}"
fi

# Create documentation
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

We've created a runtime patch that fixes JSON serialization:

1. **JSON Serialization Fix** (`json_serialization_fix.py`):
   - Enhanced JSON encoder that properly handles numpy types
   - Converts NaN to string "NaN" during serialization
   - Also handles Infinity and -Infinity values
   - Patches both `json.dumps` and Flask's `jsonify`

2. **Worker JSON Fix** (`fix_nan_json.py`):
   - Specifically targets the job status and job result endpoints
   - Wraps the response functions to check and fix NaN values
   - Uses regular expressions as a fallback to fix raw JSON strings

3. **Deployment Integration**:
   - `patch_app_with_fixes.py`: Applies fixes at startup
   - `render_startup_wrapper.sh`: Ensures fixes are applied before app starts

## Manual Deployment Instructions

If the automatic deployment didn't work, you can apply these changes manually:

1. Upload these files to your Render service:
   - `json_serialization_fix.py`
   - `fix_nan_json.py`
   - `patch_app_with_fixes.py`
   - `render_startup_wrapper.sh`

2. Modify your `render.yaml` to use the wrapper script:
   ```yaml
   startCommand: >-
     ./render_startup_wrapper.sh echo "Starting web service" &&
     python3 -c "import gc; gc.enable()" &&
     # ...rest of your original command...
   ```

3. Redeploy your service on Render

## Verification

After deployment, check the logs for:
```
✅ Applied JSON serialization patches
✅ Applied NaN JSON fix
```

Jobs should now complete successfully without JSON parsing errors.
EOL

echo -e "${GREEN}✅ Created JSON-NAN-FIX.md documentation${NC}"

# Create deployment package
echo -e "${YELLOW}Creating deployment package...${NC}"

mkdir -p deploy_package
cp json_serialization_fix.py fix_nan_json.py patch_app_with_fixes.py render_startup_wrapper.sh JSON-NAN-FIX.md deploy_package/

echo -e "${GREEN}✅ Created deployment package in deploy_package/ directory${NC}"

echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}✅ JSON NaN Fix Direct Deployment Ready!${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""
echo -e "To deploy this fix, push these files to your repository:"
echo -e "${YELLOW}  - json_serialization_fix.py${NC}"
echo -e "${YELLOW}  - fix_nan_json.py${NC}"
echo -e "${YELLOW}  - patch_app_with_fixes.py${NC}"
echo -e "${YELLOW}  - render_startup_wrapper.sh${NC}"
echo -e "${YELLOW}  - JSON-NAN-FIX.md${NC}"
echo ""
echo -e "Then modify the startCommand in render.yaml to use the wrapper:"
echo -e "${YELLOW}  startCommand: >-${NC}"
echo -e "${YELLOW}    ./render_startup_wrapper.sh echo \"Starting web service\" &&${NC}"
echo -e "${YELLOW}    # ...rest of your original command...${NC}"
echo ""
echo -e "This approach doesn't require local dependencies because"
echo -e "the fix will be applied at runtime on Render."
