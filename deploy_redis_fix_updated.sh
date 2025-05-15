#!/bin/bash
# SHAP Microservice Redis Connection Fix Deployment Script
# Date: May 15, 2025

set -e  # Exit immediately if a command exits with a non-zero status

echo "=================================================="
echo "  SHAP Microservice Redis Connection Fix Deployment"
echo "=================================================="

# 1. Backup the current redis connection patch file
if [ -f "redis_connection_patch.py" ]; then
  echo "📂 Backing up existing redis_connection_patch.py..."
  cp redis_connection_patch.py redis_connection_patch.py.bak.$(date +%Y%m%d%H%M%S)
else
  echo "⚠️  No existing redis_connection_patch.py found. Continuing with new implementation."
fi

# 2. Apply the fixed version
echo "🔧 Applying the fixed Redis connection patch..."
if [ -f "redis_connection_patch_fixed.py" ]; then
  cp redis_connection_patch_fixed.py redis_connection_patch.py
  echo "✅ Fixed Redis connection patch applied."
else
  echo "❌ Error: redis_connection_patch_fixed.py not found!"
  exit 1
fi

# 3. Verify app.py imports and uses the patch
echo "🔍 Checking app.py for Redis connection patch usage..."
if grep -q "from redis_connection_patch import apply_all_patches" app.py && grep -q "apply_all_patches()" app.py; then
  echo "✅ app.py is properly configured to use the Redis connection patch."
else
  echo "⚠️  Warning: app.py might not be properly configured to use the Redis connection patch."
  echo "    Please ensure these lines exist in app.py:"
  echo "      from redis_connection_patch import apply_all_patches"
  echo "      apply_all_patches()"
  
  read -p "Do you want to check and update app.py automatically? (y/n): " auto_update
  if [[ $auto_update == "y" || $auto_update == "Y" ]]; then
    # Check if import is missing
    if ! grep -q "from redis_connection_patch import apply_all_patches" app.py; then
      echo "📝 Adding import statement to app.py..."
      sed -i.bak -e '/^import os/a\
# Redis connection patch for better stability\
from redis_connection_patch import apply_all_patches' app.py
    fi
    
    # Check if function call is missing
    if ! grep -q "apply_all_patches()" app.py; then
      echo "📝 Adding apply_all_patches() call to app.py..."
      # Add before redis_conn = redis.from_url
      sed -i.bak -e '/redis_conn = redis.from_url/i\
# Apply Redis connection patches for better stability\
apply_all_patches()' app.py
    fi
    
    echo "✅ Updated app.py successfully."
  fi
fi

# 4. Check if .env file exists and update Redis configuration
if [ -f ".env" ]; then
  echo "📝 Updating Redis configuration in .env file..."
  
  # Check if Redis settings exist, add if missing
  if ! grep -q "REDIS_TIMEOUT" .env; then
    echo "REDIS_TIMEOUT=5" >> .env
  fi
  
  if ! grep -q "REDIS_SOCKET_KEEPALIVE" .env; then
    echo "REDIS_SOCKET_KEEPALIVE=true" >> .env
  fi
  
  if ! grep -q "REDIS_CONNECTION_POOL_SIZE" .env; then
    echo "REDIS_CONNECTION_POOL_SIZE=10" >> .env
  fi
  
  echo "✅ Updated .env file with Redis configuration."
else
  echo "📝 Creating .env file with Redis configuration..."
  cat > .env << EOF
# Redis Connection Settings
REDIS_URL=${REDIS_URL:-redis://localhost:6379/0}
REDIS_TIMEOUT=5
REDIS_SOCKET_KEEPALIVE=true
REDIS_CONNECTION_POOL_SIZE=10
EOF
  echo "✅ Created .env file with Redis configuration."
fi

# 5. Test Redis connection with the new patch
echo "🧪 Testing Redis connection with the new patch..."
if [ -f "verify_redis_connection.py" ]; then
  python3 verify_redis_connection.py
  test_result=$?
  
  if [ $test_result -eq 0 ]; then
    echo "✅ Redis connection test passed!"
  else
    echo "⚠️  Redis connection test failed or had warnings."
    echo "   Review the test output for details."
  fi
else
  echo "⚠️  Warning: verify_redis_connection.py not found, skipping Redis connection test."
fi

# 6. Restart the service if running locally
if [ -z "$RENDER" ]; then  # Not running on Render
  echo "🔄 Restarting the service locally..."
  
  # Check if service is running with gunicorn
  if pgrep -f "gunicorn.*app:app" > /dev/null; then
    echo "📌 Found running service, restarting..."
    pkill -f "gunicorn.*app:app"
    sleep 2
    nohup gunicorn -c gunicorn_config.py app:app > gunicorn.log 2>&1 &
    echo "✅ Service restarted (PID: $!)."
  else
    echo "📌 No running service found. Starting new instance..."
    nohup gunicorn -c gunicorn_config.py app:app > gunicorn.log 2>&1 &
    echo "✅ Service started (PID: $!)."
  fi
else
  echo "🚀 Running on Render, deployment will restart the service automatically."
fi

echo ""
echo "=================================================="
echo "  DEPLOYMENT INSTRUCTIONS FOR RENDER"
echo "=================================================="
echo "1. Commit these changes to your repository:"
echo "   git add redis_connection_patch.py"
echo "   git commit -m \"Fix Redis connection handling with improved error recovery\""
echo "   git push origin main"
echo ""
echo "2. Add these environment variables in the Render dashboard:"
echo "   REDIS_TIMEOUT=5"
echo "   REDIS_SOCKET_KEEPALIVE=true" 
echo "   REDIS_CONNECTION_POOL_SIZE=10"
echo ""
echo "3. Deploy your application in the Render dashboard"
echo "=================================================="
