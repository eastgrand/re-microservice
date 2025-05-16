#!/bin/bash
# filepath: /Users/voldeck/code/shap-microservice/check_render_deployment.sh

# Script to check deployment status on Render.com
# This assumes you have the Render CLI installed

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}    SHAP Microservice Deployment Status Check    ${NC}"
echo -e "${BLUE}=================================================${NC}"

# Check if render CLI is available
if ! command -v render &> /dev/null; then
    echo -e "${YELLOW}Render CLI not found. For automatic checking:${NC}"
    echo "1. Install Render CLI: https://render.com/docs/cli"
    echo "2. Login with: render login"
    echo ""
    echo -e "${YELLOW}For manual checking:${NC}"
    echo "1. Visit Render dashboard: https://dashboard.render.com"
    echo "2. Check service logs for 'nesto-mortgage-analytics'"
    echo "3. Check worker logs for 'nesto-mortgage-analytics-worker'"
    exit 0
fi

# Check service status
echo -e "${YELLOW}Checking service status...${NC}"
echo -e "${YELLOW}Web service:${NC}"
render services info nesto-mortgage-analytics
echo ""

echo -e "${YELLOW}Worker service:${NC}"
render services info nesto-mortgage-analytics-worker
echo ""

# Check service logs
echo -e "${YELLOW}Checking recent web service logs...${NC}"
render services logs nesto-mortgage-analytics --last 30
echo ""

echo -e "${YELLOW}Checking recent worker logs...${NC}"
render services logs nesto-mortgage-analytics-worker --last 30
echo ""

# Test the API endpoint if available
echo -e "${YELLOW}Testing API health endpoint...${NC}"
curl -s https://nesto-mortgage-analytics.onrender.com/health | jq .
echo ""

# Test Redis connection
echo -e "${YELLOW}Testing Redis connection...${NC}"
curl -s https://nesto-mortgage-analytics.onrender.com/admin/redis_ping | jq .
echo ""

echo -e "${BLUE}=================================================${NC}"
echo -e "${YELLOW}Deployment Status Indicators:${NC}"
echo "1. Check for 'Worker started' in worker logs"
echo "2. Look for 'Redis connection successful' in logs"
echo "3. Verify health endpoint returns 'status: healthy'" 
echo "4. Verify Redis ping returns 'success: true'"
echo ""
echo -e "${YELLOW}Memory Usage Check:${NC}"
echo "Look for memory usage logs in worker output:"
echo "- Should show values below 500MB"
echo "- If close to 512MB, consider further optimizations"
echo ""
echo -e "${YELLOW}To submit a test job:${NC}"
echo "./run_test_job.py --service-url https://nesto-mortgage-analytics.onrender.com --sample-size 50"
echo -e "${BLUE}=================================================${NC}"
