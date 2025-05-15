#!/bin/bash
# finalize_model_skip_implementation.sh - Apply all changes and deploy with model training skipped

set -e  # Exit on any error

cd "$(dirname "$0")"  # Move to script directory

echo "============================================================="
echo "    FINALIZING MODEL TRAINING SKIP FEATURE IMPLEMENTATION    "
echo "============================================================="

# 1. Apply all our changes
echo "Step 1: Applying all fixes to repository..."
git add .skip_training deploy_to_render.sh render.yaml deploy_skip_trained_model.sh \
    SKIP-TRAINING-GUIDE.md SKIP-TRAINING-DEPLOYMENT-GUIDE.md README.md

git commit -m "Fix model training skip feature - May 15, 2025"

# 2. Verify skip training is enabled
echo "Step 2: Verifying skip training is enabled..."
python3 skip_training.py enable

# 3. Push changes to repository
echo "Step 3: Pushing changes to trigger deployment..."
git push origin main

echo "============================================================="
echo "    DEPLOYMENT STARTED - MODEL TRAINING WILL BE SKIPPED      "
echo "============================================================="

echo "The changes have been pushed and deployment has started."
echo "Check the Render logs for the following messages:"
echo ""
echo "  🔄 Skip training flag detected (.skip_training file exists)"
echo "  🔄 Skipping model training step..."
echo "  ✅ Using existing model files from repository"
echo ""
echo "Your deployment should complete much faster now (~2-3 minutes)."

# Monitor Render deployment
if command -v curl &> /dev/null; then
    echo ""
    echo "============================================================="
    echo "    WAITING 90 SECONDS TO CHECK DEPLOYMENT STATUS...         "
    echo "============================================================="
    echo ""
    sleep 90
    
    echo "Checking service health..."
    curl -s https://nesto-mortgage-analytics.onrender.com/ping | grep -q "status.*OK" && \
        echo "✅ Service appears to be healthy!" || \
        echo "⚠️ Service may still be deploying. Check Render dashboard."
fi

echo ""
echo "For future deployments, simply use:"
echo "./deploy_skip_trained_model.sh"
echo ""
