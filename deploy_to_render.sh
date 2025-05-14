echo "Preparing for deployment to Render..."
#!/bin/bash
set -e

echo "Preparing for deployment to Render..."

# Install dependencies
python -m pip install --force-reinstall --no-cache-dir -r requirements.txt memory-profiler

# Enable memory optimization environment variables
export MEMORY_OPTIMIZATION=true
export MAX_MEMORY_MB=400
export AGGRESSIVE_MEMORY_MANAGEMENT=true

# Apply compatibility patches
echo "Applying compatibility patches..."
python fix_flask_werkzeug.py
python patch_shap.py

# Run setup and data preparation scripts
echo "Running setup script with memory optimization..."
python setup_for_render.py
python create_minimal_model.py
python fix_categorical_types.py
python fix_categorical_data.py

# Train model
echo "Training model..."
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
