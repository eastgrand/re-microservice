# Model Training and Deployment Process

This document explains how to ensure the model is properly trained before and during deployment to Render.

## Understanding the Error

The error you encountered:
```
FileNotFoundError: Model not found, please train the model first
```

This happens because the model file (`models/xgboost_model.pkl`) was not properly created during the build process on Render. This could be due to:

1. The training process failing silently
2. The model file not being found in the expected location
3. The model file not being persisted between build and start phases

## Solution

We've updated several components to fix this issue:

1. **Updated `render.yaml`** - Now explicitly includes `python train_model.py` in the build command
2. **Enhanced `train_model.py`** - Added a `train_and_save_model()` function that can be called from other modules
3. **Improved `app.py`** - Added better error handling and fallback mechanisms to train models on demand
4. **Created `train_before_deploy.sh`** - A script to verify everything is ready before deployment

## Pre-Deployment Steps

Before deploying to Render, follow these steps:

1. Run the preparation script:
   ```bash
   ./train_before_deploy.sh
   ```

2. This script will:
   - Set up the environment
   - Train the model
   - Verify all files exist
   - Prepare for deployment

3. If the script completes successfully, you can commit the model files to your repository:
   ```bash
   git add models/xgboost_model.pkl models/feature_names.txt data/cleaned_data.csv
   git commit -m "Add trained model and feature files for deployment"
   git push
   ```

4. If the repository has size limitations that prevent including the model files, the updated `render.yaml` will ensure they're generated during deployment.

## Deployment Process

When deploying to Render, the following occurs:

1. Render clones your repository
2. It runs the build command defined in `render.yaml`:
   ```yaml
   buildCommand: >-
     pip install -r requirements.txt && 
     python fix_flask_werkzeug.py && 
     python patch_shap.py && 
     python setup_for_render.py && 
     python train_model.py
   ```

3. The script `setup_for_render.py`:
   - Creates necessary directories
   - Checks for/creates data files
   - Maps data fields
   - Runs the model training script
   - Verifies model file exists

4. The explicit `python train_model.py` command ensures the model is trained even if the previous steps failed to do so

5. When the application starts, the enhanced error handling in `app.py` provides multiple fallback options:
   - It first tries to load the existing model
   - If that fails, it runs `setup_for_render.py`
   - If that still fails, it runs `train_model.py`
   - In the worst case, it creates an emergency minimal model

## Troubleshooting

If deployment still fails:

1. Check the Render logs for specific error messages
2. Verify that your repository includes the mapping files and scripts
3. Ensure your Render instance has sufficient memory for model training
4. Consider creating a smaller, optimized model for production use

## Full Dataset Usage

These changes ensure that the full dataset is used for training and inference when available, falling back to sample data only if absolutely necessary. The mapping of ALL fields from `nesto_merge_0.csv` is properly maintained throughout this process.
