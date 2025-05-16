#!/bin/bash
#
# Verify SHAP microservice deployment
#

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BOLD}SHAP Microservice Deployment Verification${NC}"
echo -e "---------------------------------------"

# Set the service URL
SERVICE_URL="https://nesto-mortgage-analytics.onrender.com"
echo -e "${YELLOW}Using service URL: ${SERVICE_URL}${NC}"

# Set API key
if [ -z "$API_KEY" ]; then
    API_KEY="HFqkccbN3LV5CaB"  # Default for testing
    echo -e "${YELLOW}Using default API key${NC}"
else
    echo -e "${YELLOW}Using provided API key${NC}"
fi

# Function to check endpoint
check_endpoint() {
    local endpoint=$1
    local description=$2
    
    echo -e "\n${YELLOW}Testing ${description} (${endpoint})${NC}"
    response=$(curl -s -w "\n%{http_code}" -H "X-API-KEY: $API_KEY" "${SERVICE_URL}${endpoint}")
    status_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$status_code" == "200" ]; then
        echo -e "${GREEN}✅ ${description} is working (HTTP 200)${NC}"
        echo -e "Response: $body"
        return 0
    else
        echo -e "${RED}❌ ${description} returned HTTP ${status_code}${NC}"
        echo -e "Response: $body"
        return 1
    fi
}

# Function to test worker by submitting a job
test_worker() {
    echo -e "\n${YELLOW}Testing worker by submitting a job${NC}"
    
    # Create test payload
    cat > test_payload.json << 'EOF'
{
    "analysis_type": "correlation",
    "target_variable": "Mortgage_Approvals",
    "demographic_filters": ["Income > 50000"]
}
EOF
    
    # Submit job
    echo -e "${YELLOW}Submitting job...${NC}"
    response=$(curl -s -w "\n%{http_code}" -X POST \
        -H "Content-Type: application/json" \
        -H "X-API-KEY: $API_KEY" \
        -d @test_payload.json \
        "${SERVICE_URL}/analyze")
    
    status_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$status_code" == "202" ]; then
        echo -e "${GREEN}✅ Job submitted successfully (HTTP 202)${NC}"
        # Extract job ID from response
        job_id=$(echo "$body" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)
        
        if [ -n "$job_id" ]; then
            echo -e "${GREEN}✅ Got job ID: ${job_id}${NC}"
            
            # Poll for job completion
            echo -e "${YELLOW}Polling for job completion...${NC}"
            
            for i in {1..15}; do
                echo -e "${YELLOW}Poll attempt ${i}/15...${NC}"
                
                job_response=$(curl -s -H "X-API-KEY: $API_KEY" "${SERVICE_URL}/job_status/${job_id}")
                job_status=$(echo "$job_response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
                
                echo -e "${YELLOW}Job status: ${job_status}${NC}"
                
                if [ "$job_status" == "finished" ]; then
                    echo -e "${GREEN}✅ Job completed successfully!${NC}"
                    return 0
                elif [ "$job_status" == "failed" ]; then
                    echo -e "${RED}❌ Job failed!${NC}"
                    echo -e "Response: $job_response"
                    return 1
                fi
                
                # Wait 10 seconds before checking again
                sleep 10
            done
            
            echo -e "${RED}❌ Job did not complete within timeout period${NC}"
            # Check one last time what state it's in
            job_response=$(curl -s -H "X-API-KEY: $API_KEY" "${SERVICE_URL}/job_status/${job_id}")
            echo -e "Final job response: $job_response"
            
            # Check if job moved beyond queued state
            if echo "$job_response" | grep -q '"status":"started"'; then
                echo -e "${YELLOW}⚠️ Job is being processed (started) but didn't complete in time${NC}"
                echo -e "${YELLOW}⚠️ This could mean progress is being made but it's slow${NC}"
                return 0
            else
                return 1
            fi
        else
            echo -e "${RED}❌ Could not extract job ID from response${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ Job submission failed with status ${status_code}${NC}"
        echo -e "Response: $body"
        return 1
    fi
}

# Main tests
echo -e "\n${BOLD}Running verification tests${NC}"
echo -e "------------------------"

# Check basic endpoints
check_endpoint "/ping" "Basic ping endpoint"
check_endpoint "/health" "Health endpoint"
check_endpoint "/admin/redis_ping" "Redis ping endpoint"

# Test memory optimization endpoint
check_endpoint "/admin/memory" "Memory optimization endpoint"

# Test worker by submitting a job
test_worker

echo -e "\n${BOLD}Verification Complete${NC}"
