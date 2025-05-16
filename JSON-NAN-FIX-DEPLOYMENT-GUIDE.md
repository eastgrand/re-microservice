# JSON NaN Fix Deployment Guide

## Issue Overview

The SHAP microservice is currently experiencing an issue where jobs get stuck in the "started" state and eventually fail with:

```
SyntaxError: Unexpected token 'N', ..."ications":NaN,"mortg"... is not valid JSON
```

This occurs because some values in the analysis results are `NaN`, which are not valid in standard JSON.

## Deployment Options

You have two deployment options to fix this issue:

### Option 1: Direct Deployment (Recommended)

This option doesn't require pandas or any local dependencies, as it applies the fix at runtime on Render:

```bash
./deploy_json_nan_fix_direct.sh
```

This script:
1. Creates the necessary fix files
2. Creates a startup wrapper script
3. Creates a deployment package
4. Provides instructions for modifying render.yaml

It's ideal if you don't have all the dependencies installed locally.

### Option 2: Integrated Deployment 

This option attempts to modify app.py locally and test the changes:

```bash
./deploy_json_nan_fix.sh
```

However, this requires having pandas and other dependencies installed locally, which you may not have.

## Manual Deployment Steps

If both scripts fail, you can manually deploy the fix:

1. Copy these files to your repository:
   - `json_serialization_fix.py`
   - `fix_nan_json.py`
   - `patch_app_with_fixes.py`
   - `render_startup_wrapper.sh`

2. Modify your `render.yaml` to use the wrapper script:
   ```yaml
   # For the web service
   startCommand: >-
     ./render_startup_wrapper.sh echo "Starting web service" &&
     python3 -c "import gc; gc.enable()" &&
     # ...rest of your original command...
     
   # For the worker service  
   startCommand: >-
     ./render_startup_wrapper.sh echo "Starting memory-optimized SHAP worker" &&
     # ...rest of your original command...
   ```

3. Push the changes to your repository
4. Deploy on Render

## Verification

After deployment, check the logs for:
```
✅ Applied JSON serialization patches
✅ Applied NaN JSON fix
```

Jobs should now complete successfully without JSON parsing errors.
