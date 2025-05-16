#!/bin/bash
# filepath: /Users/voldeck/code/shap-microservice/prepare_render_deployment.sh

# Minimal script to prepare deployment package without running local installations
set -e

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   Preparing SHAP Microservice Render Deployment   ${NC}"
echo -e "${BLUE}=================================================${NC}"

# Create deployment directory
DEPLOY_DIR="render_deploy"
echo -e "${YELLOW}Creating deployment directory...${NC}"
mkdir -p $DEPLOY_DIR

# Copy essential files for deployment
echo -e "${YELLOW}Copying essential files...${NC}"
# Main application files
cp app.py $DEPLOY_DIR/
cp optimize_memory.py $DEPLOY_DIR/
cp redis_connection_patch.py $DEPLOY_DIR/
cp simple_worker.py $DEPLOY_DIR/
cp setup_worker.py $DEPLOY_DIR/
cp render.yaml $DEPLOY_DIR/
cp requirements.txt $DEPLOY_DIR/
cp gunicorn_config.py $DEPLOY_DIR/

# Worker and memory patches
cp worker_process_fix.py $DEPLOY_DIR/ 2>/dev/null || :
cp memory_optimization.py $DEPLOY_DIR/ 2>/dev/null || :

# Verification scripts
cp verify_config.py $DEPLOY_DIR/
cp verify_redis_settings.py $DEPLOY_DIR/
cp verify_system_health.py $DEPLOY_DIR/ 2>/dev/null || :
cp run_test_job.py $DEPLOY_DIR/ 2>/dev/null || :

# Documentation
cp VERIFICATION-RESULTS.md $DEPLOY_DIR/ 2>/dev/null || :
cp MEMORY-OPTIMIZATIONS.md $DEPLOY_DIR/ 2>/dev/null || :

# Make scripts executable
echo -e "${YELLOW}Making scripts executable...${NC}"
chmod +x $DEPLOY_DIR/simple_worker.py
chmod +x $DEPLOY_DIR/setup_worker.py
chmod +x $DEPLOY_DIR/verify_config.py
chmod +x $DEPLOY_DIR/verify_redis_settings.py
chmod +x $DEPLOY_DIR/*.py 2>/dev/null || :

# Check worker script for Connection import
echo -e "${YELLOW}Checking setup_worker.py for Connection import...${NC}"
if grep -q "from rq import Queue, Connection" $DEPLOY_DIR/setup_worker.py; then
    echo -e "${YELLOW}Fixing Connection import in setup_worker.py...${NC}"
    # Replace the import statement
    sed -i.bak 's/from rq import Queue, Connection, Worker/from rq import Queue, Worker  # Removed Connection import/g' $DEPLOY_DIR/setup_worker.py
    echo -e "${GREEN}✅ Fixed Connection import${NC}"
fi

# Create deployment instructions
echo -e "${YELLOW}Creating deployment instructions...${NC}"
cat > $DEPLOY_DIR/README-DEPLOYMENT.md << EOF
# SHAP Microservice Deployment Instructions

This package contains the optimized SHAP microservice ready for deployment to Render.com.

## Optimizations Applied

1. **Memory Settings**
   - Standard memory threshold increased to 475MB
   - Aggressive memory management disabled for worker
   - Batch size increased to 500 rows

2. **Redis Connection Fixes**
   - Enhanced connection handling with failsafe methods
   - Better timeout and retry configuration
   - Fixed connection pool handling

3. **Worker Process Improvements**
   - Using simple_worker.py instead of the Context manager pattern
   - Fixed Connection import issue

## Deployment Steps

1. **Upload to Render**
   - Push these files to your Git repository connected to Render
   - Or use the Render dashboard to deploy manually

2. **Verify Deployment**
   - Check the worker logs to ensure it starts properly
   - Verify Redis connection with the /admin/redis_ping endpoint
   - Monitor memory usage to ensure it stays below 512MB

3. **Run Verification Scripts**
   - Use verify_config.py to check configuration
   - Use verify_redis_settings.py to check Redis connection
   - Use run_test_job.py to test end-to-end functionality

## Monitoring

- Monitor worker memory usage during job processing
- Watch for any timeouts or connection issues
- Check job processing times with different batch sizes
EOF

echo -e "${GREEN}✅ Deployment package prepared in $DEPLOY_DIR directory${NC}"

# Create a ZIP archive for easy upload
echo -e "${YELLOW}Creating ZIP archive...${NC}"
cd $DEPLOY_DIR
zip -r ../shap_microservice_render_deployment.zip .
cd ..

echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}Deployment package created: shap_microservice_render_deployment.zip${NC}"
echo -e "${YELLOW}To deploy to Render:${NC}"
echo "1. Upload this package to your repository"
echo "2. Connect your Render.com service to the repository"
echo "3. Deploy using the dashboard or Render CLI"
echo -e "${BLUE}=================================================${NC}"
