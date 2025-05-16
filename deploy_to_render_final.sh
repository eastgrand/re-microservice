#!/bin/bash
# SHAP Microservice Deployment Script to Render.com
# This script prepares and verifies all components before deployment

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   SHAP Microservice Render Deployment Script     ${NC}"
echo -e "${BLUE}=================================================${NC}"

# Step 1: Verify required files
echo -e "${YELLOW}Verifying required files...${NC}"
REQUIRED_FILES=(
    "app.py" 
    "render.yaml" 
    "redis_connection_patch.py" 
    "optimize_memory.py" 
    "simple_worker.py"
    "create_minimal_model.py"
    "requirements.txt"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ Found $file${NC}"
    else
        echo -e "${RED}❌ Missing $file - Required for deployment${NC}"
        exit 1
    fi
done

# Step 2: Create .skip_training file
echo -e "${YELLOW}Creating .skip_training flag file...${NC}"
echo "Skip model training during deployment - $(date)" > .skip_training
echo -e "${GREEN}✅ Created .skip_training flag file${NC}"

# Step 3: Create directories
echo -e "${YELLOW}Creating required directories...${NC}"
mkdir -p models
mkdir -p data
echo -e "${GREEN}✅ Created required directories${NC}"

# Step 4: Check for model files or create minimal model
echo -e "${YELLOW}Checking for model files...${NC}"
if [ -f "models/xgboost_model.pkl" ] && [ -f "models/feature_names.txt" ]; then
    echo -e "${GREEN}✅ Found existing model files${NC}"
else
    echo -e "${YELLOW}Model files not found. Creating minimal model...${NC}"
    python3 create_minimal_model.py
    
    if [ -f "models/xgboost_minimal.pkl" ] && [ -f "models/minimal_feature_names.txt" ]; then
        echo -e "${YELLOW}Copying minimal model files to standard locations...${NC}"
        cp models/xgboost_minimal.pkl models/xgboost_model.pkl
        cp models/minimal_feature_names.txt models/feature_names.txt
        echo -e "${GREEN}✅ Created and installed minimal model${NC}"
    else
        echo -e "${RED}❌ Failed to create minimal model files${NC}"
        exit 1
    fi
fi

# Step 5: Set optimized environment variables
echo -e "${YELLOW}Setting optimized environment variables...${NC}"
export MEMORY_OPTIMIZATION=true
export MAX_MEMORY_MB=475  # Increased from 450
export AGGRESSIVE_MEMORY_MANAGEMENT=false  # Disabled for better performance
export SHAP_MAX_BATCH_SIZE=500  # Increased from 300
export REDIS_HEALTH_CHECK_INTERVAL=30
export REDIS_SOCKET_KEEPALIVE=true
export REDIS_TIMEOUT=10
export SKIP_MODEL_TRAINING=true
echo -e "${GREEN}✅ Set optimized environment variables${NC}"

# Step 6: Update render.yaml configuration
echo -e "${YELLOW}Updating render.yaml with optimized settings...${NC}"
# Ensure worker settings are correctly configured
sed -i.bak 's/AGGRESSIVE_MEMORY_MANAGEMENT.*value: "true"/AGGRESSIVE_MEMORY_MANAGEMENT\n        value: "false"/g' render.yaml
sed -i.bak 's/SHAP_MAX_BATCH_SIZE.*value: "[0-9]*"/SHAP_MAX_BATCH_SIZE\n        value: "500"/g' render.yaml

# Compare files to ensure changes were made
if diff render.yaml render.yaml.bak > /dev/null; then
    echo -e "${YELLOW}No changes needed in render.yaml${NC}"
else
    echo -e "${GREEN}✅ Updated render.yaml with optimized settings${NC}"
fi

# Step 7: Prepare deployment package
echo -e "${YELLOW}Preparing deployment package...${NC}"
chmod +x ./prepare_render_deployment.sh
./prepare_render_deployment.sh
echo -e "${GREEN}✅ Deployment package prepared${NC}"

# Step 8: Deploy to Render.com
echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}           Ready to deploy to Render.com             ${NC}"
echo -e "${BLUE}======================================================${NC}"
echo -e "${YELLOW}You have two options for deployment:${NC}"
echo -e "${YELLOW}1. Manual deployment via Render Dashboard${NC}"
echo -e "   - Log into your Render account"
echo -e "   - Go to your project dashboard"
echo -e "   - Select 'Manual Deploy' and then 'Deploy latest commit'"
echo -e ""
echo -e "${YELLOW}2. Git-based deployment${NC}"
echo -e "   - Commit your changes: git add . && git commit -m 'Deploy optimized SHAP microservice'"
echo -e "   - Push to your repository: git push origin main"
echo -e "   - Render will automatically deploy the latest commit"
echo -e ""
echo -e "${YELLOW}Would you like to proceed with git-based deployment? (y/n)${NC}"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${YELLOW}Committing changes...${NC}"
    git add .
    git commit -m "Deploy optimized SHAP microservice with memory optimizations"
    echo -e "${YELLOW}Pushing changes to remote repository...${NC}"
    git push origin main
    echo -e "${GREEN}✅ Changes pushed! Render.com will now deploy automatically.${NC}"
else
    echo -e "${YELLOW}Please perform manual deployment through the Render Dashboard.${NC}"
fi

# Step 9: Instructions for verifying deployment
echo -e "${BLUE}======================================================${NC}"
echo -e "${BLUE}           Post-deployment verification              ${NC}"
echo -e "${BLUE}======================================================${NC}"
echo -e "${YELLOW}After deployment, verify the following:${NC}"
echo -e "1. Check worker logs in Render.com dashboard to ensure they start correctly"
echo -e "2. Monitor memory usage to ensure it stays below 512MB"
echo -e "3. Test job processing with a sample request"
echo -e "4. Verify that jobs move from 'started' to 'completed' state"
echo -e ""
echo -e "${GREEN}Deployment preparation complete!${NC}"
