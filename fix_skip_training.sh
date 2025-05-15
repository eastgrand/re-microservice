#!/bin/bash
# fix_skip_training.sh - Definitive fix for the model training skip feature
# Created: May 15, 2025

set -e  # Exit on any error

echo "=========================================================="
echo "    APPLYING DEFINITIVE FIX FOR MODEL TRAINING SKIP       "
echo "=========================================================="

# 1. Ensure the flag file exists and is properly set up
echo "Creating skip training flag file..."
cat > .skip_training << 'EOF'
# This file indicates model training should be skipped during deployment
# Created: May 15, 2025
SKIP_MODEL_TRAINING=true
EOF

# 2. Modify the setup_for_render.py script to respect the flag
echo "Patching setup_for_render.py to recognize skip training flag..."

# Backup original file
cp setup_for_render.py setup_for_render.py.bak

# Add direct flag check at the top of the file
sed -i.bak '1i\
# Check for skip training flag\
import os\
SKIP_TRAINING = os.path.exists(".skip_training") or os.environ.get("SKIP_MODEL_TRAINING") == "true"\
if SKIP_TRAINING:\
    print("SKIP TRAINING FLAG DETECTED - WILL NOT TRAIN MODEL")\
' setup_for_render.py

# Replace any training logic with conditional check
sed -i.bak 's/logger.info("Training model...")/logger.info("Checking if model training should be skipped...")\
    if SKIP_TRAINING:\
        logger.info("SKIPPING MODEL TRAINING due to skip_training flag")\
    else:\
        logger.info("Training model...")/g' setup_for_render.py

# 3. Update the deploy_to_render.sh script to set environment variable explicitly
echo "Updating deploy_to_render.sh to set environment variable..."
sed -i.bak '1i\
# Set environment variable to skip training\
export SKIP_MODEL_TRAINING=true\
' deploy_to_render.sh

# 4. Add skip training check to every script that might train models
echo "Adding skip training check to train_model.py..."
if [ -f "train_model.py" ]; then
    cp train_model.py train_model.py.bak
    sed -i.bak '1i\
# Check for skip training flag\
import os\
if os.path.exists(".skip_training") or os.environ.get("SKIP_MODEL_TRAINING") == "true":\
    print("SKIP TRAINING FLAG DETECTED - EXITING WITHOUT TRAINING")\
    import sys\
    sys.exit(0)\
' train_model.py
fi

# 5. Commit all changes to the repository
echo "Committing changes to repository..."
git add .skip_training setup_for_render.py deploy_to_render.sh
if [ -f "train_model.py" ]; then
    git add train_model.py
fi

git commit -m "Apply definitive fix for skip training feature"

# 6. Push changes to repository
echo "Pushing changes to trigger deployment..."
git push origin main

echo "=========================================================="
echo "    FIX APPLIED - DEPLOYMENT STARTED                     "
echo "=========================================================="
echo ""
echo "The fix has been applied and a new deployment has been triggered."
echo ""
echo "This deployment should DEFINITELY skip model training."
echo ""
echo "Check the Render logs for 'SKIPPING MODEL TRAINING' messages."
echo ""
echo "You can monitor the deployment at: https://dashboard.render.com"

# Wait a bit and check the service status
sleep 90
echo "Checking service status..."
curl -s https://nesto-mortgage-analytics.onrender.com/ping || echo "Service may still be deploying"
