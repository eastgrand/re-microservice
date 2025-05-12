#!/bin/bash

# This script will test the API endpoints of the SHAP-XGBoost microservice

# Configuration - modify these as needed
API_URL="http://localhost:8081"
API_KEY="change_me_in_production" # This should match what's in your .env file

# Text colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "SHAP-XGBoost API Test Script"
echo "==========================="
echo

# Test 1: Health Check
echo -n "Testing health check endpoint... "
HEALTH_RESPONSE=$(curl -s -X GET "${API_URL}/health" -H "X-API-KEY: ${API_KEY}")
if [[ $HEALTH_RESPONSE == *"healthy"* ]]; then
    echo -e "${GREEN}PASS${NC}"
    echo "SHAP Version: $(echo $HEALTH_RESPONSE | grep -o '"shap_version":"[^"]*"' | cut -d'"' -f4)"
else
    echo -e "${RED}FAIL${NC}"
    echo $HEALTH_RESPONSE | python -m json.tool
fi
echo

# Test 2: Metadata
echo -n "Testing metadata endpoint... "
METADATA_RESPONSE=$(curl -s -X GET "${API_URL}/metadata" -H "X-API-KEY: ${API_KEY}")
if [[ $METADATA_RESPONSE == *"success"* && $METADATA_RESPONSE == *"true"* ]]; then
    echo -e "${GREEN}PASS${NC}"
    echo "Dataset has $(echo $METADATA_RESPONSE | grep -o '"record_count":[0-9]*' | cut -d':' -f2) records"
else
    echo -e "${RED}FAIL${NC}"
    echo $METADATA_RESPONSE | python -m json.tool
fi
echo

# Test 3: Basic Analysis
echo -n "Testing analysis endpoint... "
ANALYSIS_RESPONSE=$(curl -s -X POST "${API_URL}/analyze" \
    -H "Content-Type: application/json" \
    -H "X-API-KEY: ${API_KEY}" \
    -d '{
        "analysis_type": "correlation",
        "target_variable": "Nike_Sales",
        "demographic_filters": ["Age < 40"]
    }')
if [[ $ANALYSIS_RESPONSE == *"success"* && $ANALYSIS_RESPONSE == *"true"* ]]; then
    echo -e "${GREEN}PASS${NC}"
    echo "Analysis summary: $(echo $ANALYSIS_RESPONSE | grep -o '"summary":"[^"]*"' | cut -d'"' -f4)"
else
    echo -e "${RED}FAIL${NC}"
    echo $ANALYSIS_RESPONSE | python -m json.tool
fi
echo

# Test 4: Analysis with Multiple Filters
echo -n "Testing analysis with multiple filters... "
ANALYSIS_RESPONSE=$(curl -s -X POST "${API_URL}/analyze" \
    -H "Content-Type: application/json" \
    -H "X-API-KEY: ${API_KEY}" \
    -d '{
        "analysis_type": "ranking",
        "target_variable": "Nike_Sales",
        "demographic_filters": ["Age < 30", "Income > 70000"]
    }')
if [[ $ANALYSIS_RESPONSE == *"success"* && $ANALYSIS_RESPONSE == *"true"* ]]; then
    echo -e "${GREEN}PASS${NC}"
    NUM_RESULTS=$(echo $ANALYSIS_RESPONSE | grep -o '"results":\[[^]]*\]' | grep -o 'nike_sales' | wc -l)
    echo "Got ${NUM_RESULTS} results matching the filters"
else
    echo -e "${RED}FAIL${NC}"
    echo $ANALYSIS_RESPONSE | python -m json.tool
fi
echo

echo "All tests completed"
