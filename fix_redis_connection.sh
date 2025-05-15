#!/bin/bash

# Redis Connection Fix for SHAP Microservice
# This script applies Redis connection improvements to the SHAP microservice

echo "===== SHAP Microservice Redis Connection Fix ====="
echo "This script will fix Redis connection issues in your SHAP microservice"
echo 

# Check if we're in the SHAP microservice directory
if [ ! -f "app.py" ]; then
  echo "Error: app.py not found in current directory"
  echo "Please run this script from the SHAP microservice directory"
  exit 1
fi

echo "Detected SHAP microservice in current directory"

# Create a backup of app.py
echo "Creating backup of app.py as app.py.bak"
cp app.py app.py.bak

# Generate the Redis connection patch
echo "Generating Redis connection patch..."
python3 generate_redis_patch.py

# Apply the patch
if [ ! -f "redis_connection_patch.py" ]; then
  echo "Error: redis_connection_patch.py not generated properly"
  exit 1
fi

echo "Applying Redis patch to app.py..."

# Add the import at the top of app.py
awk 'NR==1{print "from redis_connection_patch import apply_all_patches\n"}1' app.py > app.py.tmp1

# Find the Redis setup section and add the patch call before it
if grep -q "REDIS_URL" app.py.tmp1; then
  # Find line number of first REDIS_URL occurrence
  LINE_NUM=$(grep -n "REDIS_URL" app.py.tmp1 | head -n 1 | cut -d: -f1)
  
  # Insert patch call before that line
  awk -v line="$LINE_NUM" 'NR==line{print "# Apply Redis connection patches for better stability\napply_all_patches()\n"}1' app.py.tmp1 > app.py.tmp2
  
  # Replace app.py
  mv app.py.tmp2 app.py
  rm app.py.tmp1
  
  echo "✅ Successfully patched app.py with Redis connection improvements"
else
  # Fallback: just add it at the start of the file after imports
  echo "Could not find REDIS_URL in app.py, adding patch call after imports..."
  
  # Find the line number for # --- FLASK APP SETUP
  LINE_NUM=$(grep -n "# --- FLASK APP SETUP" app.py.tmp1 | head -n 1 | cut -d: -f1)
  
  if [ -n "$LINE_NUM" ]; then
    # Insert patch call before that line
    awk -v line="$LINE_NUM" 'NR==line{print "# Apply Redis connection patches for better stability\napply_all_patches()\n"}1' app.py.tmp1 > app.py.tmp2
    
    # Replace app.py
    mv app.py.tmp2 app.py
    rm app.py.tmp1
    
    echo "✅ Successfully patched app.py with Redis connection improvements"
  else
    # Just add it after the imports
    echo -e "\n# Apply Redis connection patches for better stability\napply_all_patches()\n" >> app.py.tmp1
    mv app.py.tmp1 app.py
    
    echo "✅ Added patch call to app.py"
  fi
fi

# Create .env file with Redis settings if it doesn't exist
if [ ! -f ".env" ]; then
  echo "Creating .env file with Redis connection settings..."
  cat > .env << EOF
# Redis connection settings
REDIS_TIMEOUT=5
REDIS_SOCKET_KEEPALIVE=true
REDIS_CONNECTION_POOL_SIZE=10
# Memory management
AGGRESSIVE_MEMORY_MANAGEMENT=true
EOF
else
  echo "Adding Redis connection settings to existing .env file..."
  echo -e "\n# Redis connection settings" >> .env
  echo "REDIS_TIMEOUT=5" >> .env
  echo "REDIS_SOCKET_KEEPALIVE=true" >> .env
  echo "REDIS_CONNECTION_POOL_SIZE=10" >> .env
  echo "AGGRESSIVE_MEMORY_MANAGEMENT=true" >> .env
fi

echo
echo "===== Redis Connection Fix Complete ====="
echo "The following changes have been made:"
echo "1. Added redis_connection_patch.py with improved Redis connection handling"
echo "2. Applied the patch to app.py"
echo "3. Added Redis connection settings to .env"
echo
echo "Next steps:"
echo "1. Commit and push these changes to your repository"
echo "2. Redeploy the SHAP microservice in Render"
echo "3. Monitor for Redis connection issues"
echo
echo "In the Render dashboard, add these environment variables:"
echo "REDIS_TIMEOUT=5"
echo "REDIS_SOCKET_KEEPALIVE=true" 
echo "REDIS_CONNECTION_POOL_SIZE=10"
echo "AGGRESSIVE_MEMORY_MANAGEMENT=true"
echo
