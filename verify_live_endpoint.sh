#!/bin/bash
# filepath: /Users/voldeck/code/shap-microservice/verify_live_endpoint.sh
#
# This script runs the verification against the live Render.com endpoint
# instead of localhost

set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BOLD}SHAP Microservice Live Verification${NC}"
echo -e "------------------------------------"

# Use the known production URL 
LIVE_URL="https://nesto-mortgage-analytics.onrender.com"

# Try to get API key from environment or use default
if [ -z "$API_KEY" ]; then
    echo -e "${YELLOW}⚠️ No API_KEY environment variable found, trying to find one...${NC}"
    
    # Try to extract from various scripts
    for file in $(grep -l "API_KEY.*HFqkccbN3LV5CaB" *.py *.sh 2>/dev/null); do
        KEY_LINE=$(grep "API_KEY.*HFqkccbN3LV5CaB" "$file" | head -1)
        if [ ! -z "$KEY_LINE" ]; then
            echo -e "${GREEN}✅ Found API key reference in $file${NC}"
            API_KEY="HFqkccbN3LV5CaB"
            break
        fi
    done
    
    if [ -z "$API_KEY" ]; then
        echo -e "${RED}❌ Could not find API key automatically.${NC}"
        read -p "Please enter the API key manually: " API_KEY
        
        if [ -z "$API_KEY" ]; then
            echo -e "${RED}❌ No API key provided. Cannot continue.${NC}"
            exit 1
        fi
    fi
fi

echo -e "${YELLOW}🔄 Verifying SHAP microservice at: ${LIVE_URL}${NC}"
echo -e "${YELLOW}🔑 Using API key: ${API_KEY}${NC}"

# Export variables for the Python script
export API_KEY
export SHAP_SERVICE_URL="${LIVE_URL}"

# Run the verification script
echo -e "${YELLOW}🔄 Running verification script...${NC}"
echo ""
python3 verify_shap_fix.py "${LIVE_URL}"

# Get the exit code
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✅ Verification completed successfully!${NC}"
    echo -e "${GREEN}The SHAP microservice is working correctly at ${LIVE_URL}${NC}"
else
    echo -e "${RED}${BOLD}❌ Verification failed!${NC}"
    echo -e "${RED}The SHAP microservice at ${LIVE_URL} is not working correctly.${NC}"
    echo -e "${YELLOW}Please check the logs above for more details.${NC}"
fi

exit $EXIT_CODE
