# DEPENDENCY ORDER FIX FOR RENDER DEPLOYMENT

## Issue Overview
**Date:** May 16, 2025

When deploying the SHAP microservice to Render.com, the build process was failing with the error:
```
ModuleNotFoundError: No module named 'pandas'
```

This occurred when `create_minimal_model.py` was executed as part of the deployment process, but before all required dependencies were installed.

## Root Cause Analysis

The original deployment flow had these issues:

1. The `render.yaml` file specified a `buildCommand` sequence where `render_pre_deploy.sh` was executed before dependencies were fully installed
2. Inside `render_pre_deploy.sh`, the script tried to create a minimal model using `create_minimal_model.py`
3. The minimal model script requires pandas, numpy, and other ML libraries
4. Only a limited set of dependencies were installed at that point (`psutil`, `python-dotenv`, `rq`, `redis`)

## Solution Implemented

We have implemented the following fixes:

1. **Updated `render.yaml`** to install core ML dependencies first:
   - Now installs numpy, pandas, xgboost, and scikit-learn at the start of the build process

2. **Enhanced `render_pre_deploy.sh`**:
   - Added explicit dependency installation before minimal model creation

3. **Created helper scripts**:
   - `install_dependencies.sh`: A dedicated script to install dependencies in the correct order
   - `deploy_fixed_dependencies.sh`: A new deployment script that ensures proper dependency handling

4. **Documentation**:
   - This document explaining the issue and solution

## How to Deploy

To deploy with the fixed dependency handling:

```bash
# Make the scripts executable
chmod +x *.sh

# Run the deployment script
./deploy_fixed_dependencies.sh
```

This will:
1. Update the repository with the fixed configuration
2. Push changes to your Git repository
3. Trigger a new deployment on Render with the corrected dependency order

## Verification

After deployment, you should see successful build logs without the "No module named 'pandas'" error. The minimal model should be created successfully, allowing both the web service and worker to start properly.

## Additional Notes

If issues persist after this fix, consider:

1. Verifying all ML dependencies are correctly specified in `requirements.txt`
2. Checking if any other scripts have similar dependency issues
3. Using the `install_dependencies.sh` script early in any custom deployment processes
