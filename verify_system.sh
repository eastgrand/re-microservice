#!/bin/bash
# System Verification Script for SHAP Microservice
# This script makes the verification tools executable and runs them

set -e

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   SHAP Microservice System Verification Suite    ${NC}"
echo -e "${BLUE}=================================================${NC}"

# Make scripts executable
echo -e "${YELLOW}Making verification scripts executable...${NC}"
chmod +x verify_redis_settings.py
chmod +x verify_system_health.py
chmod +x run_test_job.py
chmod +x simple_worker.py
echo -e "${GREEN}Scripts are now executable${NC}"

# Check system health
echo -e "\n${YELLOW}Running system health check...${NC}"
./verify_system_health.py --all

# Store the result
HEALTH_CHECK_RESULT=$?

# Check Redis settings specifically
echo -e "\n${YELLOW}Running Redis connection verification...${NC}"
./verify_redis_settings.py

# Store the result
REDIS_CHECK_RESULT=$?

# Get service URL from environment or use default
SERVICE_URL=${SHAP_SERVICE_URL:-"https://nesto-mortgage-analytics.onrender.com"}

# Ask if the user wants to run a test job against the service
echo -e "\n${YELLOW}Do you want to run a test job through the SHAP service?${NC}"
echo -e "This will submit a real job to: ${SERVICE_URL}"
read -p "Run test job? (y/n): " RUN_TEST_JOB

if [[ $RUN_TEST_JOB == "y" || $RUN_TEST_JOB == "Y" ]]; then
    echo -e "\n${YELLOW}Running test job...${NC}"
    ./run_test_job.py --service-url "$SERVICE_URL" --sample-size 50
    JOB_TEST_RESULT=$?
else
    echo -e "\n${YELLOW}Skipping test job run${NC}"
    JOB_TEST_RESULT=0  # Skip this test
fi

# Print summary
echo -e "\n${BLUE}=================================================${NC}"
echo -e "${BLUE}                VERIFICATION SUMMARY               ${NC}"
echo -e "${BLUE}=================================================${NC}"

if [ $HEALTH_CHECK_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ System health check: PASSED${NC}"
else
    echo -e "${RED}❌ System health check: FAILED${NC}"
fi

if [ $REDIS_CHECK_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ Redis connection check: PASSED${NC}"
else
    echo -e "${RED}❌ Redis connection check: FAILED${NC}"
fi

if [[ $RUN_TEST_JOB == "y" || $RUN_TEST_JOB == "Y" ]]; then
    if [ $JOB_TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✅ Test job execution: PASSED${NC}"
    else
        echo -e "${RED}❌ Test job execution: FAILED${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ Test job execution: SKIPPED${NC}"
fi

# Overall result
echo -e "\n${BLUE}Overall verification result:${NC}"
if [ $HEALTH_CHECK_RESULT -eq 0 ] && [ $REDIS_CHECK_RESULT -eq 0 ] && ([ $JOB_TEST_RESULT -eq 0 ] || [[ $RUN_TEST_JOB != "y" && $RUN_TEST_JOB != "Y" ]]); then
    echo -e "${GREEN}✅ All checks PASSED${NC}"
    echo -e "${GREEN}The SHAP microservice appears to be configured correctly${NC}"
    echo -e "${GREEN}with the recent memory optimizations and Redis connection fixes.${NC}"
    echo -e "\n${BLUE}Recommended next steps:${NC}"
    echo -e "1. Monitor worker performance with the optimized settings"
    echo -e "2. Verify job processing speed with the larger batch size"
    echo -e "3. Check memory usage during peak load periods"
else
    echo -e "${RED}❌ Some checks FAILED${NC}"
    echo -e "${RED}Please review the logs above for details.${NC}"
    echo -e "\n${BLUE}Recommended next steps:${NC}"
    echo -e "1. Address any failed checks"
    echo -e "2. Review the worker and Redis configuration"
    echo -e "3. Check memory optimization settings"
    echo -e "4. Verify the worker is using the correct script (simple_worker.py)"
fi

echo -e "\n${BLUE}=================================================${NC}"

exit $(( HEALTH_CHECK_RESULT || REDIS_CHECK_RESULT || (JOB_TEST_RESULT * (RUN_TEST_JOB == "y" || RUN_TEST_JOB == "Y")) ))
