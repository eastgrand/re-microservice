#!/bin/bash
# verify_skip_training.sh - Verify that model training is skipped
# Created May 15, 2025

echo "==============================================" 
echo "    VERIFYING SKIP TRAINING FUNCTIONALITY     "
echo "==============================================" 
echo ""

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# 1. Check if .skip_training file exists
if [ -f ".skip_training" ]; then
    echo "✅ Skip training flag exists: .skip_training"
else
    echo "❌ Skip training flag does NOT exist"
    echo "Creating it now..."
    echo "This file indicates model training should be skipped during deployment" > .skip_training
    echo "✅ Created .skip_training flag file"
fi

# 2. Look for environment variable
if [ -n "$SKIP_MODEL_TRAINING" ] && [ "$SKIP_MODEL_TRAINING" = "true" ]; then
    echo "✅ Environment variable is set: SKIP_MODEL_TRAINING=$SKIP_MODEL_TRAINING"
else
    echo "ℹ️ Environment variable not set, setting it now"
    export SKIP_MODEL_TRAINING=true
    echo "✅ Set SKIP_MODEL_TRAINING=true"
fi

# 3. Test with setup_for_render.py
echo ""
echo "Testing setup_for_render.py behavior with flags set..."
echo ""

# Capture output to check for "SKIPPING MODEL TRAINING" message
OUTPUT=$(python3 setup_for_render.py 2>&1)
echo "$OUTPUT" | grep -i "SKIPPING MODEL TRAINING" || echo "❌ setup_for_render.py did not detect skip flag"

if echo "$OUTPUT" | grep -i "SKIPPING MODEL TRAINING" > /dev/null; then
    echo "✅ setup_for_render.py correctly detected skip flag"
else
    echo "❌ setup_for_render.py did NOT detect skip flag"
    echo "   Check implementation of skip detection"
fi

# 4. Report status of model files
echo ""
echo "Checking model files..."

if [ -f "models/xgboost_model.pkl" ] && [ -f "models/feature_names.txt" ]; then
    echo "✅ Model files exist and can be used when skipping training"
else
    echo "⚠️ Model files don't exist, will need to create minimal model"
fi

echo ""
echo "==============================================" 
echo "    VERIFICATION COMPLETE                     "
echo "==============================================" 
echo ""
echo "Based on this verification:"
echo ""
if echo "$OUTPUT" | grep -i "SKIPPING MODEL TRAINING" > /dev/null; then
    echo "✅ Skip training feature IS WORKING CORRECTLY"
    echo "   Model training will be skipped during deployments"
else
    echo "❌ Skip training feature IS NOT WORKING CORRECTLY"
    echo "   Check implementation in setup_for_render.py"
fi
echo ""
