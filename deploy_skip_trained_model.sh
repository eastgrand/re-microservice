#!/bin/bash
# deploy_skip_trained_model.sh - Deploy with model training skipped
# This script ensures model training is skipped during deployment for faster testing

set -e  # Exit on any error

echo "========================================"
echo "    DEPLOYING WITH MODEL TRAINING SKIP  "
echo "========================================"

# Step 1: Ensure the .skip_training flag exists
echo "Creating/verifying .skip_training flag..."
python3 skip_training.py enable

# Step 2: Make sure it's tracked in git
echo "Ensuring .skip_training flag is tracked in git..."
git add .skip_training
git commit -m "Enable model training skip flag for faster deployment" || echo "Skip flag already committed"

# Step 3: Push changes to trigger deployment
echo "Pushing changes to Render..."
git push origin main

echo ""
echo "========================================"
echo "    DEPLOYMENT INITIATED                "
echo "========================================"
echo ""
echo "The deployment has been started with the model training skip flag enabled."
echo "This should make your deployment complete faster by using the existing model."
echo ""
echo "Monitor the deployment progress at: https://dashboard.render.com"
echo ""
