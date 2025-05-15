# Model Training Skip Feature Guide

## Purpose

The model training step during deployment can take several minutes, which significantly slows down the deployment process. This feature allows you to skip model training when:

1. You're making changes unrelated to the model (UI changes, API fixes, etc.)
2. You're deploying with the same dataset as before
3. You're testing deployment changes quickly

## How to Use the Feature

### Quick Deployment with Skip Training

Use our dedicated script:

```bash
./deploy_skip_trained_model.sh
```

This single command handles everything needed to skip model training during deployment.

### Manual Control

You have full control over when to skip training:

```bash
# Enable skipping model training
python3 skip_training.py enable

# Disable skipping model training
python3 skip_training.py disable

# Check current status
python3 skip_training.py status
```

### Verify It's Working

When the skip training feature is active, you should see this message in the Render deployment logs:

```
🔄 Skip training flag detected (.skip_training file exists)
🔄 Skipping model training step...
✅ Using existing model files from repository
```

## How It Works

The feature uses a simple flag file (`.skip_training`) that's checked during deployment:

1. If the flag file exists, model training is skipped
2. If not, normal model training proceeds

We also support using the `SKIP_MODEL_TRAINING=true` environment variable as an alternative.

## Troubleshooting

If you're still seeing model training occur even with the skip feature enabled:

1. **Check the flag file exists**: Run `ls -la .skip_training` to verify the file exists
2. **Verify it's committed to git**: Run `git ls-files .skip_training` to check
3. **Check Render logs**: Look for "Skip training flag detected" in the build logs
4. **Ensure model files exist**: The `models/xgboost_model.pkl` and `models/feature_names.txt` files must exist

## Best Practices

1. **Enable for rapid testing**: Use during development for faster feedback
2. **Disable for production releases**: When making significant changes, disable skip to ensure a fresh model
3. **Re-enable after verification**: After confirming your model works correctly, re-enable to speed up future deployments

## Deployment Time Comparison

| Configuration | Typical Deployment Time |
|--------------|------------------------|
| Training Enabled | 8-10 minutes |
| Training Skipped | 2-3 minutes |

This represents a 70-75% reduction in deployment time!
