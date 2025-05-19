#!/bin/bash
# Final deployment script to fix worker name collisions and deploy to Render.com

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   SHAP Microservice Final Deployment Script     ${NC}"
echo -e "${BLUE}=================================================${NC}"

# Step 1: Verify render.yaml has the correct configuration
echo -e "${YELLOW}Verifying render.yaml configuration...${NC}"
if grep -q "nesto-shap-worker" render.yaml; then
    echo -e "${GREEN}✅ render.yaml has the correct worker name${NC}"
else
    echo -e "${RED}❌ render.yaml has incorrect worker configuration${NC}"
    echo -e "${YELLOW}Updating render.yaml...${NC}"
    # Create backup
    cp render.yaml render.yaml.bak
    
    # Update worker configuration
    sed -i '' 's/name: nesto-mortgage-analytics-worker.*/name: nesto-shap-worker/g' render.yaml
    sed -i '' 's/python3 -c "import gc; gc.enable(); print(.*)chmod +x .\\/simple_worker.py/python3 -c "import gc; gc.enable(); print('"'"'Garbage collection enabled'"'"')" \&\& chmod +x .\\/simple_worker.py/g' render.yaml
    
    echo -e "${GREEN}✅ render.yaml updated with correct worker configuration${NC}"
fi

# Step 2: Commit and push changes
echo -e "${YELLOW}Committing changes...${NC}"
git add render.yaml simple_worker.py WORKER-NAME-COLLISION-FIX.md
git commit -m "Fix worker name collision issue and update Render configuration"

echo -e "${YELLOW}Pushing changes to remote repository...${NC}"
git push origin main

echo -e "${GREEN}✅ Changes pushed to repository${NC}"

echo -e "${BLUE}=================================================${NC}"
echo -e "${YELLOW}Deployment Instructions${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e ""
echo -e "1. ${YELLOW}Login to your Render.com dashboard${NC}"
echo -e "   Go to: https://dashboard.render.com/"
echo -e ""
echo -e "2. ${YELLOW}Delete all existing worker services${NC}"
echo -e "   - Click on each worker service"
echo -e "   - Go to 'Settings'"
echo -e "   - Scroll down to 'Danger Zone'"
echo -e "   - Click 'Delete Service'"
echo -e ""
echo -e "3. ${YELLOW}Deploy a new service blueprint${NC}"
echo -e "   - Click 'New+' and select 'Blueprint'"
echo -e "   - Select your GitHub repository"
echo -e "   - Click 'Apply'"
echo -e ""
echo -e "4. ${YELLOW}Check deployment logs${NC}"
echo -e "   - Monitor logs for both web service and worker"
echo -e "   - Verify the worker starts without name collision errors"
echo -e ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}✅ Deployment preparation complete!${NC}"
echo -e "${BLUE}=================================================${NC}"
