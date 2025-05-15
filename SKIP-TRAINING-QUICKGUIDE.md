## Quick Guide: Skip Model Training on Deployment

**Updated:** May 15, 2025

### Current Issue

When deploying to Render, the model is being retrained every time, even when the data hasn't changed. This:
- Makes deployments slow (adds ~5-10 minutes)
- Increases memory usage during deployment
- Is unnecessary when only code changes are being made

### The Solution

We've added a feature to skip model training during deployment. Here's how to use it:

### Step 1: Enable Skip Training

```bash
# Make sure you have the model trained locally first
python train_model.py

# Enable skip training
python skip_training.py enable

# Run the preparation script
./prepare_skip_training_deployment.sh
```

### Step 2: Commit and Push

```bash
# Force add the model files (normally ignored by .gitignore)
git add .skip_training models/xgboost_model.pkl models/feature_names.txt -f

# Commit and push
git commit -m "Enable skip training for faster deployments"
git push
```

### Step 3: Deploy

Deploy as usual to Render. The deployment will now be much faster since it skips the model training step.

### When to Re-enable Training

Re-enable model training when:
- Your dataset has changed
- Model parameters have changed
- You want to force a fresh model

```bash
python skip_training.py disable
git add .skip_training
git commit -m "Re-enable model training"
git push
```

### Checking Status

To check whether training will be skipped:

```bash
python skip_training.py status
```

For more details, see [DEPLOYMENT-OPTIMIZATION.md](./DEPLOYMENT-OPTIMIZATION.md)
