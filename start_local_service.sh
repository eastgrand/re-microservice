#!/bin/bash

# Start SHAP Microservice Locally for Verification
# This script starts the SHAP microservice locally so you can verify the fix

echo "====================================="
echo "Starting SHAP Microservice for Local Verification"
echo "====================================="

# Check if the service is already running
SERVICE_PID=$(ps -ef | grep "python3 app.py\|gunicorn.*app:app" | grep -v grep | awk '{print $2}')
if [ ! -z "$SERVICE_PID" ]; then
  echo "⚠️ SHAP microservice is already running (PID: $SERVICE_PID)"
  echo "Stopping existing service before starting a new one..."
  kill $SERVICE_PID
  sleep 2
fi

# Determine how to run the service
echo "Checking available launch methods..."

if [ -f "gunicorn_config.py" ]; then
  echo "✅ Found gunicorn_config.py, will use gunicorn to start service"
  LAUNCH_CMD="gunicorn app:app --config gunicorn_config.py"
  # Extract port from gunicorn config
  PORT=$(grep -o "bind\s*=.*:[0-9]\+" gunicorn_config.py | grep -o "[0-9]\+")
  if [ -z "$PORT" ]; then
    PORT=8000  # gunicorn default
  fi
else
  echo "Using Flask development server"
  LAUNCH_CMD="python3 app.py"
  PORT=5000  # Flask default
fi

echo "Service will be available at http://localhost:$PORT"

# Set required environment variables
export FLASK_ENV=development
export FLASK_DEBUG=1

echo "====================================="
echo "Starting SHAP microservice with memory optimizations..."

# Run the service in the background
$LAUNCH_CMD &
SERVICE_PID=$!

echo "Service started with PID: $SERVICE_PID"
echo "Waiting for service to initialize..."

# Wait for service to start
sleep 3

# Check if service is running
if ps -p $SERVICE_PID > /dev/null; then
  echo "✅ Service is running"
  echo ""
  echo "You can now run the verification script:"
  echo "python3 verify_shap_fix.py http://localhost:$PORT"
  echo ""
  echo "Press Ctrl+C to stop the service when done"
  
  # Keep script running until user presses Ctrl+C
  trap "kill $SERVICE_PID; echo 'Service stopped'; exit 0" INT
  wait $SERVICE_PID
else
  echo "❌ Service failed to start. Check logs for errors."
  exit 1
fi
