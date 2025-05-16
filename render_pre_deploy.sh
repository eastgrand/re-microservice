#!/bin/bash
# filepath: /Users/voldeck/code/shap-microservice/render_pre_deploy.sh
# Pre-deployment check script specifically for Render.com

set -e  # Exit immediately if a command exits with a non-zero status

# Create .skip_training flag file
echo "Creating .skip_training flag file..."
echo "Skip model training during deployment - $(date)" > .skip_training
echo "✅ Created .skip_training flag file"

# Set environment variable for skipping model training
export SKIP_MODEL_TRAINING=true
echo "✅ Set SKIP_MODEL_TRAINING=true"

# Create directories if needed
mkdir -p models
mkdir -p data
echo "✅ Ensured models and data directories exist"

# Check for model files
echo "Checking for model files..."
if [ -f "models/xgboost_model.pkl" ] && [ -f "models/feature_names.txt" ]; then
    echo "✅ Found existing model files!"
    
    # Create backup just in case
    echo "Creating backup of model files..."
    cp models/xgboost_model.pkl models/xgboost_model.pkl.bak
    cp models/feature_names.txt models/feature_names.txt.bak
    echo "✅ Backup created"
else
    echo "⚠️ Model files not found"
    
    # Check if minimal model can be created
    if [ -f "create_minimal_model.py" ]; then
        echo "Creating minimal model..."
        python3 create_minimal_model.py
        
        if [ -f "models/xgboost_minimal.pkl" ] && [ -f "models/minimal_feature_names.txt" ]; then
            echo "Copying minimal model files to standard locations..."
            cp models/xgboost_minimal.pkl models/xgboost_model.pkl
            cp models/minimal_feature_names.txt models/feature_names.txt
            echo "✅ Created and installed minimal model"
        else
            echo "❌ ERROR: Failed to create minimal model files"
            exit 1
        fi
    else
        echo "❌ ERROR: Cannot create model files (create_minimal_model.py not found)!"
        exit 1
    fi
fi

# Set optimized environment variables for deployment
echo "Setting optimized environment variables..."
export MEMORY_OPTIMIZATION=true
export MAX_MEMORY_MB=475
export AGGRESSIVE_MEMORY_MANAGEMENT=false
export SHAP_MAX_BATCH_SIZE=500
export REDIS_HEALTH_CHECK_INTERVAL=30
export REDIS_SOCKET_KEEPALIVE=true
export REDIS_TIMEOUT=10
echo "✅ Set optimized environment variables"

# Verify worker script files
echo "Checking worker scripts..."
if [ -f "simple_worker.py" ]; then
    echo "✅ simple_worker.py found"
    chmod +x simple_worker.py
else
    echo "❌ ERROR: simple_worker.py not found!"
    exit 1
fi

if [ -f "setup_worker.py" ]; then
    echo "✅ setup_worker.py found"
    chmod +x setup_worker.py
    
    # Check and fix Connection import
    if grep -q "from rq import Queue, Connection" setup_worker.py; then
        echo "Fixing setup_worker.py Connection import..."
        sed -i.bak 's/from rq import Queue, Connection, Worker/from rq import Queue, Worker  # Removed Connection import/g' setup_worker.py
        echo "✅ Fixed Connection import"
    fi
fi

echo "Pre-deployment checks completed successfully!"
echo "The system is ready for deployment to Render.com"
echo ""
echo "Current model files:"
ls -la models/
echo ""
echo "Deployment can proceed safely."
exit 0
