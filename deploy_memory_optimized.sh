#!/bin/bash
# filepath: /Users/voldeck/code/shap-microservice/deploy_memory_optimized.sh

# Enhanced deployment script with memory optimizations and Redis fixes
set -e

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE} SHAP Microservice Optimized Deployment to Render ${NC}"
echo -e "${BLUE}=================================================${NC}"

# Set optimized environment variables
echo -e "${YELLOW}Setting optimized environment variables...${NC}"
export MEMORY_OPTIMIZATION=true
export MAX_MEMORY_MB=475  # Increased from 450
export AGGRESSIVE_MEMORY_MANAGEMENT=false  # Disabled for better performance
export SHAP_MAX_BATCH_SIZE=500  # Increased from 300
export REDIS_HEALTH_CHECK_INTERVAL=30
export REDIS_SOCKET_KEEPALIVE=true
export REDIS_TIMEOUT=10

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
python3 -m pip install --force-reinstall --no-cache-dir -r requirements.txt memory-profiler psutil

# Apply Redis connection patches
echo -e "${YELLOW}Applying Redis connection patches...${NC}"
# Make sure the patch module is importable
if [ -f "redis_connection_patch.py" ]; then
    echo -e "${GREEN}✅ Redis connection patch found${NC}"
    # Make simple_worker.py executable
    chmod +x simple_worker.py
    echo -e "${GREEN}✅ Made simple_worker.py executable${NC}"
else
    echo -e "${RED}❌ Redis connection patch not found${NC}"
    exit 1
fi

# Apply compatibility patches
echo -e "${YELLOW}Applying compatibility patches...${NC}"
python3 fix_flask_werkzeug.py
python3 patch_shap.py

# Run setup and data preparation scripts
echo -e "${YELLOW}Running setup script with memory optimization...${NC}"
python3 setup_for_render.py
python3 fix_categorical_types.py
python3 fix_categorical_data.py

# Verify worker scripts
echo -e "${YELLOW}Verifying worker scripts...${NC}"
if [ -f "simple_worker.py" ]; then
    echo -e "${GREEN}✅ simple_worker.py found${NC}"
    # Make simple_worker.py executable
    chmod +x simple_worker.py
    echo -e "${GREEN}✅ Made simple_worker.py executable${NC}"
else
    echo -e "${RED}❌ simple_worker.py not found${NC}"
    exit 1
fi

# Check setup_worker.py for Connection import
if [ -f "setup_worker.py" ]; then
    if grep -q "from rq import Queue, Connection" setup_worker.py; then
        echo -e "${RED}❌ setup_worker.py still contains Connection import${NC}"
        echo -e "${YELLOW}Fixing setup_worker.py...${NC}"
        # Replace the import statement
        sed -i.bak 's/from rq import Queue, Connection, Worker/from rq import Queue, Worker  # Removed Connection import/g' setup_worker.py
        echo -e "${GREEN}✅ Fixed setup_worker.py${NC}"
    else
        echo -e "${GREEN}✅ setup_worker.py doesn't use Connection import${NC}"
    fi
    # Make setup_worker.py executable
    chmod +x setup_worker.py
    echo -e "${GREEN}✅ Made setup_worker.py executable${NC}"
else
    echo -e "${RED}❌ setup_worker.py not found${NC}"
fi

# Use both direct flag check and environment variable for flexibility
SKIP_TRAINING=false

# Check if skip flag file exists
if [ -f ".skip_training" ]; then
    echo -e "${YELLOW}🔄 Skip training flag detected (.skip_training file exists)${NC}"
    SKIP_TRAINING=true
fi

# Also check for environment variable override
if [ "$SKIP_MODEL_TRAINING" = "true" ]; then
    echo -e "${YELLOW}🔄 Skip training environment variable detected (SKIP_MODEL_TRAINING=true)${NC}"
    SKIP_TRAINING=true
fi

# Apply the skipping logic
if [ "$SKIP_TRAINING" = "true" ]; then
    echo -e "${YELLOW}🔄 Skipping model training step...${NC}"
    echo -e "${YELLOW}🔄 This will make deployment much faster!${NC}"
    
    # Check if model file exists
    if [ -f "models/xgboost_model.pkl" ] && [ -f "models/feature_names.txt" ]; then
        echo -e "${GREEN}✅ Using existing model files from repository${NC}"
    else
        echo -e "${YELLOW}⚠️  Warning: Model files not found but skip_training is enabled${NC}"
        echo -e "${YELLOW}⚠️  Will create minimal model as fallback${NC}"
        python3 create_minimal_model.py
        # Copy minimal model files to standard locations
        cp models/xgboost_minimal.pkl models/xgboost_model.pkl
        cp models/minimal_feature_names.txt models/feature_names.txt
        echo -e "${GREEN}✅ Copied minimal model to standard location${NC}"
    fi
else
    echo -e "${YELLOW}🔄 Training model (set SKIP_MODEL_TRAINING=true or create .skip_training to skip)...${NC}"
    # Create minimal model as backup first
    python3 create_minimal_model.py
    # Train full model
    python3 train_model.py
fi

# Check if model was created successfully
if [ -f "models/xgboost_model.pkl" ]; then
    echo -e "${GREEN}Model created successfully!${NC}"
    ls -la models/
else
    echo -e "${RED}Error: Model creation failed!${NC}"
    exit 1
fi

# Test application startup and port binding
echo -e "${YELLOW}Testing application startup and port binding...${NC}"
export PORT=10000
export FLASK_ENV=development
# Run with optimized memory settings
export AGGRESSIVE_MEMORY_MANAGEMENT=false
export MAX_MEMORY_MB=475
export SHAP_MAX_BATCH_SIZE=500

echo -e "${YELLOW}Starting application with memory-optimized settings...${NC}"
timeout 10 gunicorn app:app --bind 0.0.0.0:$PORT --timeout 30 &
GUNICORN_PID=$!
sleep 5

# Check if gunicorn is actually running and binding to the port
if ps -p $GUNICORN_PID > /dev/null; then
    echo -e "${GREEN}✅ Application started successfully and is binding to port $PORT${NC}"
    kill $GUNICORN_PID
    wait $GUNICORN_PID 2>/dev/null
else
    echo -e "${RED}❌ ERROR: Application failed to start or bind to port${NC}"
    echo -e "${RED}This is likely why Render deployment is failing with 'no open ports detected'${NC}"
    exit 1
fi

# Create verification scripts
echo -e "${YELLOW}Creating verification scripts package...${NC}"
chmod +x verify_config.py
chmod +x verify_redis_settings.py
if [ -f "verify_system_health.py" ]; then
    chmod +x verify_system_health.py
fi
if [ -f "run_test_job.py" ]; then
    chmod +x run_test_job.py
fi

# Run config verification
echo -e "${YELLOW}Running configuration verification...${NC}"
./verify_config.py

# Create deployment log
DEPLOY_DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="memory_optimization_deployment_$DEPLOY_DATE.log"
{
    echo "SHAP Microservice Memory Optimization Deployment"
    echo "==============================================="
    echo "Date: $(date)"
    echo ""
    echo "Memory Optimization Settings:"
    echo "- MAX_MEMORY_MB = $MAX_MEMORY_MB"
    echo "- AGGRESSIVE_MEMORY_MANAGEMENT = $AGGRESSIVE_MEMORY_MANAGEMENT"
    echo "- SHAP_MAX_BATCH_SIZE = $SHAP_MAX_BATCH_SIZE"
    echo ""
    echo "Redis Connection Settings:"
    echo "- REDIS_HEALTH_CHECK_INTERVAL = $REDIS_HEALTH_CHECK_INTERVAL"
    echo "- REDIS_SOCKET_KEEPALIVE = $REDIS_SOCKET_KEEPALIVE"
    echo "- REDIS_TIMEOUT = $REDIS_TIMEOUT"
    echo ""
    echo "Worker Configuration:"
    echo "- Using simple_worker.py"
    echo "- Connection import removed from setup_worker.py"
    echo ""
    echo "Configuration Verification Results:"
    ./verify_config.py
} > $LOG_FILE

echo -e "${GREEN}Deployment log created: $LOG_FILE${NC}"

# Instructions for Render deployment
echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE} DEPLOYMENT INSTRUCTIONS ${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}Everything is ready for deployment to Render!${NC}"
echo -e "${YELLOW}To deploy to Render:${NC}"
echo "1. Commit your changes: git add . && git commit -m 'Memory optimized deployment with Redis fixes'"
echo "2. Push to your repository: git push origin main"
echo "3. Deploy from Render dashboard or via Render CLI"
echo ""
echo -e "${YELLOW}After deployment:${NC}"
echo "1. Check worker logs to ensure it's running with simple_worker.py"
echo "2. Monitor memory usage to verify it stays under 512MB"
echo "3. Run test jobs to confirm batch size processing works correctly"

echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}Deployment preparation complete!${NC}"
