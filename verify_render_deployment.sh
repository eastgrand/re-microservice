#!/bin/bash
# Post-Deployment Verification Script for SHAP Microservice
# This script helps verify that the SHAP microservice has been deployed correctly
# Updated May 15, 2025: Added worker registration check

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   SHAP Microservice Deployment Verification      ${NC}"
echo -e "${BLUE}=================================================${NC}"

# Get the service URL from user input
echo -e "${YELLOW}Enter the URL of your deployed service:${NC}"
read -r SERVICE_URL

if [[ -z "$SERVICE_URL" ]]; then
    echo -e "${RED}No URL provided. Exiting.${NC}"
    exit 1
fi

# Step 1: Check if the service is responding
echo -e "${YELLOW}Checking if the service is responding...${NC}"
if curl -s --head "$SERVICE_URL" | grep "200\|302" > /dev/null; then
    echo -e "${GREEN}✅ Service is up and responding${NC}"
else
    echo -e "${RED}❌ Service is not responding correctly${NC}"
    echo -e "${YELLOW}Trying healthcheck endpoint...${NC}"
    
    # Try the healthcheck endpoint as fallback
    if curl -s --head "$SERVICE_URL/health" | grep "200\|302" > /dev/null; then
        echo -e "${GREEN}✅ Healthcheck endpoint is responding${NC}"
    else
        echo -e "${RED}❌ Healthcheck endpoint is not responding${NC}"
        echo -e "${YELLOW}Check Render dashboard for deployment status and logs${NC}"
    fi
fi

# Step 2: Check Redis connection
echo -e "${YELLOW}Checking Redis connection (via API)...${NC}"
REDIS_CHECK=$(curl -s "$SERVICE_URL/redis-check" || echo "Failed to connect")
if [[ "$REDIS_CHECK" == *"connected"* ]]; then
    echo -e "${GREEN}✅ Redis connection successful${NC}"
else
    echo -e "${RED}❌ Redis connection failed${NC}"
    echo -e "${YELLOW}Check Redis connection settings in Render environment variables${NC}"
fi

# Step 3: Check worker status
echo -e "${YELLOW}Checking worker status (via API)...${NC}"
WORKER_CHECK=$(curl -s "$SERVICE_URL/worker-status" || echo "Failed to get worker status")
if [[ "$WORKER_CHECK" == *"active"* ]]; then
    echo -e "${GREEN}✅ Worker is active and processing jobs${NC}"
    echo -e "$WORKER_CHECK"
elif [[ "$WORKER_CHECK" == *"worker"* ]]; then
    echo -e "${YELLOW}⚠️ Worker information available but status unclear${NC}"
    echo -e "$WORKER_CHECK"
else
    echo -e "${RED}❌ Cannot verify worker status${NC}"
    echo -e "${YELLOW}Check the worker logs in Render dashboard${NC}"
    echo -e "${YELLOW}Make sure the worker service is running${NC}"
fi

# Step 4: Check memory usage
echo -e "${YELLOW}Checking memory usage (via API)...${NC}"
MEMORY_CHECK=$(curl -s "$SERVICE_URL/memory-stats" || echo "Failed to get memory stats")
if [[ "$MEMORY_CHECK" == *"memory_usage_mb"* ]]; then
    echo -e "${GREEN}✅ Memory stats available${NC}"
    echo -e "$MEMORY_CHECK"
    
    # Extract memory usage value if possible
    MEMORY_USAGE=$(echo $MEMORY_CHECK | grep -o '"memory_usage_mb":[0-9]*\.[0-9]*' | cut -d':' -f2)
    if [[ ! -z "$MEMORY_USAGE" ]]; then
        echo -e "${YELLOW}Current memory usage: ${MEMORY_USAGE}MB${NC}"
        if (( $(echo "$MEMORY_USAGE > 450" | bc -l) )); then
            echo -e "${RED}⚠️ Memory usage is high (>450MB)${NC}"
        else
            echo -e "${GREEN}✅ Memory usage is within acceptable limits${NC}"
        fi
    fi
else
    echo -e "${RED}❌ Could not retrieve memory stats${NC}"
    echo -e "${YELLOW}Check logs in Render dashboard${NC}"
fi

# Step 5: Submit a test job
echo -e "${YELLOW}Would you like to submit a test SHAP job? (y/n)${NC}"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${YELLOW}Submitting test job...${NC}"
    
    # Create test job data
    JOB_DATA='{
      "data": [
        {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0},
        {"feature1": 4.0, "feature2": 5.0, "feature3": 6.0}
      ],
      "options": {
        "max_rows": 2,
        "batch_size": 2
      }
    }'
    
    # Submit job
    JOB_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" -d "$JOB_DATA" "$SERVICE_URL/api/submit-shap-job")
    echo -e "$JOB_RESPONSE"
    
    # Extract job ID
    JOB_ID=$(echo $JOB_RESPONSE | grep -o '"job_id":"[^"]*"' | cut -d':' -f2 | tr -d '"')
    
    if [[ ! -z "$JOB_ID" ]]; then
        echo -e "${GREEN}✅ Job submitted with ID: $JOB_ID${NC}"
        
        # Poll for job status
        echo -e "${YELLOW}Polling for job status...${NC}"
        MAX_ATTEMPTS=10
        ATTEMPT=0
        JOB_COMPLETED=false
        
        while [[ $ATTEMPT -lt $MAX_ATTEMPTS ]]; do
            ATTEMPT=$((ATTEMPT+1))
            echo -e "${YELLOW}Checking job status (attempt $ATTEMPT/$MAX_ATTEMPTS)...${NC}"
            
            JOB_STATUS=$(curl -s "$SERVICE_URL/api/job-status/$JOB_ID")
            echo -e "Job status: $JOB_STATUS"
            
            if [[ "$JOB_STATUS" == *"completed"* ]]; then
                echo -e "${GREEN}✅ Job completed successfully!${NC}"
                JOB_COMPLETED=true
                break
            elif [[ "$JOB_STATUS" == *"failed"* ]]; then
                echo -e "${RED}❌ Job failed${NC}"
                break
            fi
            
            echo -e "${YELLOW}Waiting 10 seconds before next check...${NC}"
            sleep 10
        done
        
        if [[ "$JOB_COMPLETED" == false ]]; then
            echo -e "${RED}❌ Job did not complete within expected time${NC}"
            echo -e "${YELLOW}Check worker logs in Render dashboard${NC}"
        fi
    else
        echo -e "${RED}❌ Failed to submit test job${NC}"
    fi
else
    echo -e "${YELLOW}Skipping test job submission${NC}"
fi

# Step 6: Final verification summary
echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}           Verification Summary                  ${NC}"
echo -e "${BLUE}=================================================${NC}"
echo -e "${YELLOW}1. Service Accessibility:${NC} Check if endpoints are accessible"
echo -e "${YELLOW}2. Redis Connection:${NC} Verify Redis connection is stable"
echo -e "${YELLOW}3. Worker Status:${NC} Ensure worker is active and processing jobs"
echo -e "${YELLOW}4. Memory Usage:${NC} Monitor in Render dashboard, should stay under 512MB"
echo -e "${YELLOW}5. Job Processing:${NC} Ensure jobs progress from 'started' to 'completed'"
echo -e "${YELLOW}6. Error Handling:${NC} Check logs for any recurring errors"
echo -e ""
echo -e "${GREEN}Verification process complete!${NC}"
echo -e "${YELLOW}For ongoing monitoring, use the Render dashboard.${NC}"
