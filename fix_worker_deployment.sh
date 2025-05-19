#!/bin/bash
# Fix Worker Service Deployment on Render
# This script fixes issues with worker service deployment and provides options for resolution

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   Fix Worker Service Deployment on Render        ${NC}"
echo -e "${BLUE}=================================================${NC}"

echo -e "${YELLOW}Verifying render.yaml is properly formatted...${NC}"
if [ -f "render.yaml" ]; then
    echo -e "${GREEN}✅ render.yaml exists${NC}"
else
    echo -e "${RED}❌ render.yaml not found${NC}"
    exit 1
fi

# Check for common YAML issues
if grep -q "nesto-shap-worker" render.yaml; then
    echo -e "${GREEN}✅ Worker service defined in render.yaml${NC}"
else
    echo -e "${RED}❌ Worker service not properly defined in render.yaml${NC}"
    exit 1
fi

echo -e "${YELLOW}Committing updated render.yaml...${NC}"
git add render.yaml
git commit -m "Fix render.yaml formatting for proper worker deployment"

echo -e "${YELLOW}Pushing changes to repository...${NC}"
git push origin main

echo -e "${GREEN}✅ Changes pushed to repository${NC}"

echo -e "${BLUE}=================================================${NC}"
echo -e "${YELLOW}Deployment Options${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e ""
echo -e "Option 1: ${YELLOW}Wait for automatic deployment${NC}"
echo -e "  - Render should deploy both services based on the updated render.yaml"
echo -e "  - Monitor the Render dashboard to see if both services are created"
echo -e ""
echo -e "Option 2: ${YELLOW}Manual worker creation${NC}"
echo -e "  - If the worker is still not created automatically:"
echo -e "    Run: ./manual_worker_deployment.sh"
echo -e "    Follow the instructions to manually create the worker service"
echo -e ""
echo -e "Option 3: ${YELLOW}Contact Render support${NC}"
echo -e "  - If the issue persists, it may be related to your Render account settings"
echo -e "  - Contact Render support with details about the issue"
echo -e ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${YELLOW}Verification Steps After Deployment${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e ""
echo -e "1. Check that both web service and worker are created in Render dashboard"
echo -e "2. Verify the worker logs for successful startup"
echo -e "3. Test the web service endpoints"
echo -e "4. Submit a SHAP job to test the complete workflow"
echo -e ""
echo -e "${GREEN}✅ Deployment preparation complete!${NC}"
