#!/bin/bash

# Redis Connection Fix Deployment Script for SHAP Microservice
# This script applies Redis connection fixes and restarts the service

echo "===== SHAP Microservice Redis Fix Deployment ====="
echo "This script will apply Redis connection improvements and restart the service"
echo 

# Check if we're in the SHAP microservice directory
if [ ! -f "app.py" ]; then
  echo "Error: app.py not found in current directory"
  echo "Please run this script from the SHAP microservice directory"
  exit 1
fi

# Install any missing packages
echo "Installing required packages..."
pip install redis rq flask gunicorn python-dotenv requests

# Make sure the Redis connection patch is applied
if [ ! -f "redis_connection_patch.py" ]; then
  echo "Redis connection patch not found. Running redis_fix.py..."
  python3 redis_fix.py
else
  echo "Redis connection patch already exists."
fi

# Make the verification script executable
chmod +x verify_redis_connection.py

echo
echo "===== Restarting SHAP Microservice ====="
echo "This will restart the service to apply Redis connection improvements"

# Find the process ID of the current service (if any)
PID=$(ps -ef | grep "gunicorn" | grep -v grep | awk '{print $2}')
if [ ! -z "$PID" ]; then
  echo "Found running SHAP microservice (PID: $PID), stopping..."
  kill $PID
  sleep 2
  
  # Check if it's still running and force kill if necessary
  if ps -p $PID > /dev/null; then
    echo "Process still running, force killing..."
    kill -9 $PID
    sleep 2
  fi
fi

# Start the service
echo "Starting SHAP microservice with Redis improvements..."
nohup gunicorn -c gunicorn_config.py app:app > gunicorn.log 2>&1 &
NEW_PID=$!
echo "Started SHAP microservice with PID: $NEW_PID"

# Wait for startup
echo "Waiting for service to initialize..."
sleep 5

# Run the verification test
echo 
echo "===== Verifying Redis Connection ====="
python3 verify_redis_connection.py

# Show instructions for Render deployment
echo
echo "===== Deployment Instructions ====="
echo "To deploy these Redis connection improvements to Render:"
echo "1. Commit all changes to your repository"
echo "2. Push changes to your remote repository"
echo "3. Deploy to Render (automatic or manual)"
echo "4. Add the following environment variables in the Render dashboard:"
echo "   REDIS_TIMEOUT=5"
echo "   REDIS_SOCKET_KEEPALIVE=true"
echo "   REDIS_CONNECTION_POOL_SIZE=10"
echo "   AGGRESSIVE_MEMORY_MANAGEMENT=true"
echo
echo "After deployment, verify the Redis connection is working"
echo "by running: python3 verify_redis_connection.py"
echo
