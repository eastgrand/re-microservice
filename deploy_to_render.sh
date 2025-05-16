echo "Preparing for deployment to Render..."
#!/bin/bash
set -e

echo "Preparing for deployment to Render..."

# Install dependencies
python3 -m pip install --force-reinstall --no-cache-dir -r requirements.txt memory-profiler

# Enable memory optimization environment variables
export MEMORY_OPTIMIZATION=true
export MAX_MEMORY_MB=400
export AGGRESSIVE_MEMORY_MANAGEMENT=true

# Apply compatibility patches
echo "Applying compatibility patches..."
python3 fix_flask_werkzeug.py
python3 patch_shap.py

# Run setup and data preparation scripts
echo "Running setup script with memory optimization..."
python3 setup_for_render.py
python3 fix_categorical_types.py
python3 fix_categorical_data.py

# Use both direct flag check and environment variable for flexibility
SKIP_TRAINING=false

# Check if skip flag file exists
if [ -f ".skip_training" ]; then
    echo "🔄 Skip training flag detected (.skip_training file exists)"
    SKIP_TRAINING=true
fi

# Also check for environment variable override
if [ "$SKIP_MODEL_TRAINING" = "true" ]; then
    echo "🔄 Skip training environment variable detected (SKIP_MODEL_TRAINING=true)"
    SKIP_TRAINING=true
fi

# Apply the skipping logic
if [ "$SKIP_TRAINING" = "true" ]; then
    echo "🔄 Skipping model training step..."
    echo "🔄 This will make deployment much faster!"
    
    # Create models directory if it doesn't exist
    if [ ! -d "models" ]; then
        echo "📁 Creating models directory..."
        mkdir -p models
    fi
    
    # Check if model file exists
    if [ -f "models/xgboost_model.pkl" ] && [ -f "models/feature_names.txt" ]; then
        echo "✅ Using existing model files from repository"
        # Create a backup copy just to be safe
        echo "📋 Creating backup of existing model files..."
        cp models/xgboost_model.pkl models/xgboost_model.pkl.bak
        cp models/feature_names.txt models/feature_names.txt.bak
    else
        echo "⚠️  Warning: Model files not found but skip_training is enabled"
        echo "⚠️  Will create minimal model as fallback"
        python3 create_minimal_model.py
        # Copy minimal model files to standard locations
        cp models/xgboost_minimal.pkl models/xgboost_model.pkl
        cp models/minimal_feature_names.txt models/feature_names.txt
        echo "✅ Copied minimal model to standard location"
    fi
else
    echo "🔄 Training model (set SKIP_MODEL_TRAINING=true or create .skip_training to skip)..."
    # Create minimal model as backup first
    python3 create_minimal_model.py
    # Train full model
    python3 train_model.py
fi

# Check if model was created successfully
if [ -f "models/xgboost_model.pkl" ]; then
    echo "Model created successfully!"
    ls -la models/
else
    echo "Error: Model creation failed!"
    exit 1
fi

# Test application startup and port binding
echo "Testing application startup and port binding..."
export PORT=10000
export FLASK_ENV=development
timeout 10 gunicorn app:app --bind 0.0.0.0:$PORT --timeout 10 &
GUNICORN_PID=$!
sleep 5

# Check if gunicorn is actually running and binding to the port
if ps -p $GUNICORN_PID > /dev/null; then
    echo "✅ Application started successfully and is binding to port $PORT"
    kill $GUNICORN_PID
    wait $GUNICORN_PID 2>/dev/null
else
    echo "❌ ERROR: Application failed to start or bind to port"
    echo "This is likely why Render deployment is failing with 'no open ports detected'"
    exit 1
fi

# Deploy to Render
echo "Everything looks good! To deploy to Render:"
echo "1. Commit your changes: git add . && git commit -m 'Enhanced memory optimization for Render deployment'"
echo "2. Push to your repository: git push origin main"
echo "3. Deploy from Render dashboard or via Render CLI"
echo ""
echo "Remember that the render.yaml file is already configured with memory optimization settings."
