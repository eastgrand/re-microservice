# SHAP Microservice Model Training Optimization

**Date:** May 15, 2025  
**Status:** Implemented  

## Overview

This update introduces a feature to make deployments to Render faster by allowing you to skip the model training step. This is useful when your dataset hasn't changed, but you want to deploy code updates more quickly.

## The Problem

Each time the SHAP microservice is deployed to Render, the model training process runs as part of the deployment, which:

1. Takes significant time (5-10 minutes)
2. Uses substantial memory resources
3. Is unnecessary when the underlying data hasn't changed

## The Solution

We've added a feature that allows you to toggle model training during deployment:

1. A flag file (`.skip_training`) to indicate model training should be skipped
2. A utility script (`skip_training.py`) to manage this flag
3. A modified deployment script that checks for this flag
4. Documentation on how to use this feature

## How to Use

### Skip Model Training for Faster Deployments

```bash
# Train the model locally first (only needed once)
python train_model.py

# Enable skipping model training 
python skip_training.py enable

# Commit the changes including the flag and model files
git add .skip_training models/xgboost_model.pkl models/feature_names.txt
git commit -m "Enable skipping model training for faster deployments"
git push
```

### Re-enable Model Training

```bash
# Re-enable model training when data has changed
python skip_training.py disable

# Commit the change
git add .skip_training
git commit -m "Re-enable model training due to data changes"
git push
```

### Check Current Status

```bash
python skip_training.py status
```

## Important Notes

1. When skipping training, the model files must be included in your repository
2. The `.gitignore` file has been updated to NOT ignore the `.skip_training` flag
3. If model files are not found but skipping is enabled, a minimal model will be created

## Documentation

For more details, see [DEPLOYMENT-OPTIMIZATION.md](./DEPLOYMENT-OPTIMIZATION.md)
