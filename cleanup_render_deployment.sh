#!/bin/bash
# Render Deployment Cleanup Script
# This script helps clean up multiple worker instances in Render.com

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   SHAP Microservice Render Deployment Cleanup   ${NC}"
echo -e "${BLUE}=================================================${NC}"

# Step 1: Verify the updated render.yaml file
echo -e "${YELLOW}Checking render.yaml configuration...${NC}"
if grep -q "nesto-shap-worker" render.yaml; then
    echo -e "${GREEN}✅ render.yaml has been updated with the correct worker name${NC}"
else
    echo -e "${RED}❌ render.yaml still has incorrect worker configuration${NC}"
    echo -e "${YELLOW}Please make sure to update the worker name to 'nesto-shap-worker'${NC}"
    exit 1
fi

# Step 2: Instructions for deployment
echo -e "${YELLOW}Steps to clean up multiple workers in Render:${NC}"
echo -e ""
echo -e "1. ${YELLOW}Login to your Render.com dashboard${NC}"
echo -e "   Go to: https://dashboard.render.com/"
echo -e ""
echo -e "2. ${YELLOW}Delete all existing worker services${NC}"
echo -e "   - Click on each worker service"
echo -e "   - Go to 'Settings'"
echo -e "   - Scroll down to the 'Danger Zone'"
echo -e "   - Click 'Delete Service'"
echo -e "   - Confirm deletion"
echo -e ""
echo -e "3. ${YELLOW}Push updated render.yaml with the corrected configuration${NC}"
echo -e "   Run the following commands:"
echo -e ""
echo -e "   git add render.yaml"
echo -e "   git commit -m \"Fix worker configuration in render.yaml\""
echo -e "   git push origin main"
echo -e ""
echo -e "4. ${YELLOW}Deploy a new single worker from the dashboard${NC}"
echo -e "   - After pushing changes, go to Render dashboard"
echo -e "   - Choose 'New+' and select 'Blueprint'"
echo -e "   - Select your GitHub repository"
echo -e "   - Click 'Apply'"
echo -e ""
echo -e "5. ${YELLOW}Verify only one worker is created${NC}"
echo -e "   The worker will be named 'nesto-shap-worker'"
echo -e ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${YELLOW}Note: If you want to do this automatically, run:${NC}"
echo -e ""
echo -e "git add render.yaml"
echo -e "git commit -m \"Fix worker configuration in render.yaml\""
echo -e "git push origin main"
echo -e ""
echo -e "${BLUE}=================================================${NC}"
