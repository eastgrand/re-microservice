# Deployment Optimization Guide

## Skip Model Training During Deployment

The SHAP microservice includes a feature to skip model training during deployment to Render. This is useful when:
- You've already trained the model locally
- The training data hasn't changed
- You want faster deployments
- You want to avoid the memory usage of model training on Render

### How It Works

The deployment process checks for a flag file (`.skip_training`) to determine whether to skip the model training step. When this file is present, the deployment script will use the existing model files instead of retraining.

### Usage

To control model training behavior, use the `skip_training.py` script:

1. **Enable skipping model training** (faster deployments):
   ```bash
   python skip_training.py enable
   ```
   This creates a `.skip_training` file that tells the deployment process to skip training.

2. **Disable skipping model training** (full training on deploy):
   ```bash
   python skip_training.py disable
   ```
   This removes the `.skip_training` file, ensuring models are trained during deployment.

3. **Check current status**:
   ```bash
   python skip_training.py status
   ```
   This shows whether model training will be skipped or performed.

### Recommended Workflow

For the most efficient workflow:

1. Train the model locally:
   ```bash
   python train_model.py
   ```

2. Verify the model files were created:
   - Check that `models/xgboost_model.pkl` exists
   - Check that `models/feature_names.txt` exists

3. Enable skipping for faster deployments:
   ```bash
   python skip_training.py enable
   ```

4. Commit the model files to your repository:
   ```bash
   git add models/xgboost_model.pkl models/feature_names.txt
   git commit -m "Add trained model files"
   ```

5. Deploy to Render (training will be skipped)

### When to Re-Enable Training

You should disable skipping and allow training when:

1. Your dataset has changed
2. You've modified model parameters or features
3. You're experiencing model-related issues

To re-enable training:
```bash
python skip_training.py disable
```

### Fallback Mechanism

If the `.skip_training` flag is set but the model files aren't found, the deployment script will fall back to creating a minimal model to prevent application failures.

### Important Notes

- The `.skip_training` file should be committed to your repository to ensure it's present during deployment.
- Model files (`models/xgboost_model.pkl` and `models/feature_names.txt`) must be committed to your repository when using this feature.
- If your repository has size limits that prevent committing model files, you can't use this feature and should remove the `.skip_training` file.
