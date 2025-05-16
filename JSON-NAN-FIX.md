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
