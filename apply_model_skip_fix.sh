#!/bin/bash
# apply_model_skip_fix.sh - Apply the model training skip feature fix

cd "$(dirname "$0")"  # Move to script directory

echo "===========================================" 
echo "   APPLYING MODEL TRAINING SKIP FIX        "
echo "===========================================" 

# Stage all our changes
git add .skip_training deploy_to_render.sh render.yaml deploy_skip_trained_model.sh SKIP-TRAINING-GUIDE.md README.md

# Commit the changes
git commit -m "Fix model training skip feature for faster deployments"

# Push to the repository
git push origin main

# Show confirmation
echo ""
echo "===========================================" 
echo "   FIX APPLIED AND PUSHED TO REPOSITORY    "
echo "===========================================" 
echo ""
echo "To deploy with model training skipped, run:"
echo "./deploy_skip_trained_model.sh"
echo ""
