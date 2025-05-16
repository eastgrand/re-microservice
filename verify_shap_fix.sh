#!/bin/bash

# SHAP Microservice Fix Verification Script
# This script helps verify that the SHAP fix is working correctly

echo "====================================="
echo "SHAP Fix Verification Helper"
echo "====================================="

# Check if we should use local or remote service
if [ "$1" == "--local" ] || [ "$1" == "-l" ]; then
  echo "Testing against local service..."
  
  # Start local service if it's not already running
  SERVICE_PID=$(ps -ef | grep "python3 app.py\|gunicorn.*app:app" | grep -v grep | awk '{print $2}')
  if [ -z "$SERVICE_PID" ]; then
    echo "Local service not detected. Starting it now..."
    
    # Start service in background
    ./start_local_service.sh &
    STARTER_PID=$!
    
    # Wait for service to start
    echo "Waiting for service to initialize..."
    sleep 5
    
    # Check if service started successfully
    SERVICE_PID=$(ps -ef | grep "python3 app.py\|gunicorn.*app:app" | grep -v grep | awk '{print $2}')
    if [ -z "$SERVICE_PID" ]; then
      echo "❌ Failed to start local service"
      kill $STARTER_PID 2>/dev/null
      exit 1
    fi
    
    echo "✅ Local service started"
    STOP_SERVICE=true
  else
    echo "✅ Local service already running (PID: $SERVICE_PID)"
    STOP_SERVICE=false
  fi
  
  # Determine service URL
  if [ -f "gunicorn_config.py" ]; then
    PORT=$(grep -o "bind\s*=.*:[0-9]\+" gunicorn_config.py | grep -o "[0-9]\+")
    if [ -z "$PORT" ]; then
      PORT=8000  # gunicorn default
    fi
  else
    PORT=5000  # Flask default
  fi
  
  SERVICE_URL="http://localhost:$PORT"
  
else
  # Remote service
  if [ -z "$SHAP_SERVICE_URL" ]; then
    echo "⚠️ No SHAP_SERVICE_URL environment variable found."
    
    # Try to extract URL from render.yaml
    if [ -f "render.yaml" ]; then
      SERVICE_NAME=$(grep -o "name:.*web" render.yaml | head -1 | awk '{print $2}')
      if [ ! -z "$SERVICE_NAME" ]; then
        SERVICE_URL="https://$SERVICE_NAME.onrender.com"
        echo "Found service name in render.yaml: $SERVICE_NAME"
        echo "Using URL: $SERVICE_URL"
      else
        echo "❌ Could not determine service URL from render.yaml"
        echo "Please provide the URL using SHAP_SERVICE_URL environment variable"
        echo "Example: SHAP_SERVICE_URL=https://your-service.onrender.com ./verify_shap_fix.sh"
        exit 1
      fi
    else
      echo "❌ Cannot determine service URL"
      echo "Please provide the URL using SHAP_SERVICE_URL environment variable"
      echo "Example: SHAP_SERVICE_URL=https://your-service.onrender.com ./verify_shap_fix.sh"
      exit 1
    fi
  else
    SERVICE_URL="$SHAP_SERVICE_URL"
  fi
  
  echo "Testing against deployed service at: $SERVICE_URL"
fi

# Check if API key is needed and available
if [ -z "$API_KEY" ]; then
  # Try to find API key in app.py or .env file
  if [ -f "app.py" ]; then
    API_KEY_LINE=$(grep -o "API_KEY.*=.*['\"]\w*['\"]" app.py)
    if [ ! -z "$API_KEY_LINE" ]; then
      API_KEY=$(echo $API_KEY_LINE | grep -o "['\"]\w*['\"]" | tr -d "'\"")
      echo "Found API key in app.py"
    fi
  fi
  
  # Still no API key, check .env file
  if [ -z "$API_KEY" ] && [ -f ".env" ]; then
    API_KEY_LINE=$(grep "API_KEY" .env)
    if [ ! -z "$API_KEY_LINE" ]; then
      API_KEY=$(echo $API_KEY_LINE | cut -d'=' -f2 | tr -d "'\" ")
      echo "Found API key in .env file"
    fi
  fi
  
  # If still no API key, export an empty one
  if [ -z "$API_KEY" ]; then
    echo "⚠️ No API key found. If the service requires authentication, this may fail."
    API_KEY=""
  fi
fi

export API_KEY
export SHAP_SERVICE_URL="$SERVICE_URL"

echo "====================================="
echo "Running verification script..."

# Run the Python verification script
python3 verify_shap_fix.py

# Save the verification result
VERIFY_RESULT=$?

# If we started the service, stop it
if [ "$STOP_SERVICE" = true ]; then
  echo "Stopping local service..."
  kill $SERVICE_PID 2>/dev/null
  kill $STARTER_PID 2>/dev/null
fi

echo "====================================="
if [ $VERIFY_RESULT -eq 0 ]; then
  echo "✅ Verification completed successfully!"
else
  echo "❌ Verification failed"
fi

exit $VERIFY_RESULT
