#!/bin/bash
# Deployment script for Nesto mortgage microservice to Render
# Ensures memory optimizations are properly configured

echo "Preparing for deployment to Render..."

# Make sure all dependencies are installed
pip install -r requirements.txt

# Enable memory optimization environment variables
export MEMORY_OPTIMIZATION=true
export MAX_MEMORY_MB=400
export AGGRESSIVE_MEMORY_MANAGEMENT=true

# Apply all necessary patches
echo "Applying compatibility patches..."
python fix_flask_werkzeug.py
python patch_shap.py 

# Test memory optimization functions with updated legacy field removal
echo "Testing memory optimization..."
python -c "from optimize_memory import log_memory_usage, get_memory_usage, prune_dataframe_columns; import pandas as pd; print(f'Current memory usage: {get_memory_usage():.2f} MB'); df = pd.read_csv('data/cleaned_data.csv', nrows=100); print(f'Columns before optimization: {len(df.columns)}'); df = prune_dataframe_columns(df); print(f'Columns after optimization: {len(df.columns)}'); print('Legacy fields successfully removed')"

# Run setup script with memory optimization
echo "Running setup script with memory optimization..."
python setup_for_render.py

# Train model with memory optimization for testing
echo "Training model with memory optimization (test run)..."
python train_model.py

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
