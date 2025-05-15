#!/bin/bash
# This script helps prepare for a deployment with skip_training enabled

echo "=========================================================="
echo "  PREPARING DEPLOYMENT WITH SKIP TRAINING ENABLED"
echo "=========================================================="

# Check if skip_training is enabled
if [ ! -f ".skip_training" ]; then
    echo "❌ Skip training is not enabled yet. Running skip_training.py..."
    python3 skip_training.py enable
fi

# Check model files
MODEL_FILE="models/xgboost_model.pkl"
FEATURE_FILE="models/feature_names.txt"

if [ ! -f "$MODEL_FILE" ] || [ ! -f "$FEATURE_FILE" ]; then
    echo "❌ Model files missing. Training model now..."
    python3 train_model.py
    
    if [ ! -f "$MODEL_FILE" ] || [ ! -f "$FEATURE_FILE" ]; then
        echo "❌ Model training failed. Cannot proceed."
        exit 1
    fi
fi

echo "✅ Skip training is enabled and model files exist."
echo 
echo "To commit these changes for deployment, run:"
echo 
echo "  git add .skip_training $MODEL_FILE $FEATURE_FILE -f"
echo "  git commit -m \"Enable skip training for faster deployments\""
echo "  git push"
echo 
echo "NOTE: The -f flag is required to force adding the model files"
echo "      that are normally ignored by .gitignore"
echo 
echo "=========================================================="
