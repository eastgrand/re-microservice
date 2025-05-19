#!/bin/bash
# fix_dependencies.sh - Ensure dependencies are installed before model creation
# This script fixes the issue where create_minimal_model.py fails due to missing pandas dependency

set -e  # Exit immediately if a command exits with a non-zero status

echo "============================================="
echo "    FIXING DEPENDENCIES FOR MODEL CREATION   "
echo "============================================="

# Step 1: Install core dependencies first
echo "Installing core dependencies..."
pip install numpy pandas xgboost scikit-learn

# Step 2: Create minimal model with dependencies ensured
echo "Creating minimal model..."
python3 create_minimal_model.py

# Step 3: Verify model was created
if [ -f "models/xgboost_minimal.pkl" ] && [ -f "models/minimal_feature_names.txt" ]; then
    echo "✅ Minimal model created successfully!"
    # Copy to standard locations as a backup
    cp models/xgboost_minimal.pkl models/xgboost_model.pkl
    cp models/minimal_feature_names.txt models/feature_names.txt
    echo "✅ Copied minimal model to standard locations"
else
    echo "❌ Failed to create minimal model!"
    exit 1
fi

echo "✅ Dependencies and model creation fix completed successfully."
