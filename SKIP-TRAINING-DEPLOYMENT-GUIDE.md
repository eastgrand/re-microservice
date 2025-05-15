# Instructions for Deploying Without Model Training

## Issue Fixed

The model training skip feature was not working correctly during deployment, causing unnecessary time to be spent training the model even when no changes to the model were needed.

## What We Fixed

1. **Enhanced Skip Training Detection**:
   - Now checks for both the `.skip_training` file AND a `SKIP_MODEL_TRAINING` environment variable
   - Added better logging during the skip detection process

2. **Render Configuration**:
   - Added `SKIP_MODEL_TRAINING=true` environment variable to `render.yaml`
   - Updated build command to check for the flag file early in the process

3. **Deployment Scripts**:
   - Created a dedicated script `deploy_skip_trained_model.sh` that ensures proper flag setup
   - Made the skip training feature more robust in `deploy_to_render.sh`

4. **Documentation**:
   - Added comprehensive documentation in `README.md`
   - Created a dedicated guide in `SKIP-TRAINING-GUIDE.md`

## How to Deploy Without Model Training

### Method 1: Use the New Deployment Script (Recommended)

```bash
cd /Users/voldeck/code/shap-microservice
./deploy_skip_trained_model.sh
```

This script will:
1. Create/verify the `.skip_training` flag
2. Commit it to git if needed
3. Push to trigger deployment
4. Deployment will automatically skip model training

### Method 2: Manual Process

```bash
cd /Users/voldeck/code/shap-microservice

# Enable skip training
python3 skip_training.py enable

# Verify the flag exists
ls -la .skip_training

# Add to git if not already tracked
git add .skip_training
git commit -m "Enable model training skip"

# Push to deploy
git push origin main
```

### Method 3: Environment Variable Only

If you prefer not to use the flag file, you can:
1. Go to the Render dashboard
2. Edit the service configuration
3. Add environment variable: `SKIP_MODEL_TRAINING=true`
4. Deploy manually

## Verifying It Works

When deploying, look for these lines in the Render logs:

```
🔄 Skip training flag detected (.skip_training file exists)
🔄 Skipping model training step...
✅ Using existing model files from repository
```

If you see these messages, the model training is being skipped and your deployment will be much faster.

## Troubleshooting

If you're still seeing model training happening:

1. Make sure you've applied our fixes by running:
   ```bash
   ./apply_model_skip_fix.sh
   ```

2. Verify the `.skip_training` file is in your repository:
   ```bash
   git ls-files .skip_training
   ```

3. Check that the environment variable is set in Render dashboard

4. Review the Render logs to see if the skip training detection messages appear
