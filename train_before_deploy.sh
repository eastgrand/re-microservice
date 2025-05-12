#!/bin/bash
# This script prepares the environment for deployment by training the model
# before pushing to the repository that will be deployed to Render.

echo "========== PREPARING FOR DEPLOYMENT =========="
echo "1. Setting up environment..."
python setup_for_render.py

echo ""
echo "2. Training model..."
python train_model.py

echo ""
echo "3. Verifying model and feature files..."
if [ -f "models/xgboost_model.pkl" ]; then
  MODEL_SIZE=$(du -h models/xgboost_model.pkl | cut -f1)
  echo "✅ Model file exists (Size: $MODEL_SIZE)"
else
  echo "❌ ERROR: Model file not found!"
  exit 1
fi

if [ -f "models/feature_names.txt" ]; then
  FEATURE_COUNT=$(wc -l < models/feature_names.txt)
  echo "✅ Feature names file exists ($FEATURE_COUNT features)"
else
  echo "❌ ERROR: Feature names file not found!"
  exit 1
fi

echo ""
echo "4. Verifying dataset..."
if [ -f "data/cleaned_data.csv" ]; then
  DATA_SIZE=$(du -h data/cleaned_data.csv | cut -f1)
  echo "✅ Cleaned dataset exists (Size: $DATA_SIZE)"
else
  echo "❌ WARNING: Cleaned dataset not found! Sample data will be used in deployment."
fi

echo ""
echo "5. Deployment preparation complete!"
echo "You can now commit these files to your repository and deploy to Render:"
echo "   - models/xgboost_model.pkl"
echo "   - models/feature_names.txt" 
echo "   - data/cleaned_data.csv (if available)"
echo ""
echo "Remember to use the render.yaml buildCommand as defined, which includes model training:"
echo "buildCommand: >"
echo "  pip install -r requirements.txt && "
echo "  python fix_flask_werkzeug.py && "
echo "  python patch_shap.py && "
echo "  python setup_for_render.py && "
echo "  python train_model.py"
echo ""
echo "This ensures that even if the model files are not included in your repository,"
echo "they will be generated during the deployment process on Render."
