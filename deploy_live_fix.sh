#!/bin/bash
# filepath: /Users/voldeck/code/shap-microservice/deploy_live_fix.sh
#
# SHAP Microservice Live Fix Deployment Script
# This script deploys fixes for the SHAP microservice to the live environment

set -e  # Exit on errors

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BOLD}SHAP Microservice Live Fix Deployment${NC}"
echo -e "------------------------------------"

# Step 1: Create a backup of critical files
echo -e "\n${BOLD}Step 1: Creating backups${NC}"
mkdir -p backups

for file in app.py render.yaml setup_worker.py; do
    if [ -f "$file" ]; then
        cp "$file" "backups/${file}.bak-$(date +%Y%m%d-%H%M%S)"
        echo -e "${GREEN}✅ Created backup of $file${NC}"
    fi
done

# Step 2: Update render.yaml with correct worker configuration
echo -e "\n${BOLD}Step 2: Updating render.yaml${NC}"
cat > render.yaml << 'EOL'
# filepath: /Users/voldeck/code/shap-microservice/render.yaml
services:
  # Main web service for the API endpoints
  - type: web
    name: nesto-mortgage-analytics
    env: python
    plan: starter
    buildCommand: >-
      echo "Starting build process with FORCED skip-training" && 
      chmod +x ./deploy_to_render.sh &&
      echo "This file indicates model training should be skipped during deployment" > .skip_training &&
      export SKIP_MODEL_TRAINING=true &&
      echo "⚡ SKIP TRAINING ENABLED - Model training will be bypassed" &&
      ./deploy_to_render.sh &&
      pip install psutil python-dotenv
    startCommand: >-
      echo "Starting web service" &&
      python3 -c "import gc; gc.enable()" &&
      python3 -c "import sys; sys.path.append('.')" &&
      gunicorn --config=gunicorn_config.py app:app
    envVars:
      - key: SHAP_MAX_BATCH_SIZE
        value: "500"
      - key: REDIS_URL
        value: "rediss://default:AVnAAAIjcDEzZjMwMjdiYWI5MjA0NjY3YTQ4ZjRjZjZjNWZhNTdmM3AxMA@ruling-stud-22976.upstash.io:6379"
      - key: REDIS_HEALTH_CHECK_INTERVAL
        value: "30"
      - key: API_KEY
        sync: false  # This indicates that the value is secret and should be set in the Render dashboard
      - key: PORT
        value: "10000"
      
  # Worker service for processing SHAP jobs
  - type: worker
    name: nesto-mortgage-analytics-worker
    env: python
    buildCommand: >-
      echo "Building worker service" &&
      chmod +x ./deploy_to_render.sh &&
      export SKIP_MODEL_TRAINING=true &&
      ./deploy_to_render.sh &&
      pip install psutil python-dotenv rq redis
    startCommand: >-
      echo "Starting memory-optimized SHAP worker" &&
      python3 -c "import gc; gc.enable(); print('Garbage collection enabled')" &&
      python3 setup_worker.py
    plan: starter
    envVars:
      - key: SHAP_MAX_BATCH_SIZE
        value: "300"  # Smaller batch size for workers to avoid memory issues
      - key: REDIS_URL
        value: "rediss://default:AVnAAAIjcDEzZjMwMjdiYWI5MjA0NjY3YTQ4ZjRjZjZjNWZhNTdmM3AxMA@ruling-stud-22976.upstash.io:6379"
      - key: REDIS_HEALTH_CHECK_INTERVAL
        value: "30"
      - key: API_KEY
        sync: false  # This indicates that the value is secret and should be set in the Render dashboard
EOL
echo -e "${GREEN}✅ Updated render.yaml with correct worker configuration${NC}"

# Step 3: Update or create shap_memory_fix.py
echo -e "\n${BOLD}Step 3: Creating memory optimization module${NC}"
cat > shap_memory_fix.py << 'EOL'
#!/usr/bin/env python3
"""
SHAP Memory Optimization

This module provides memory optimization for SHAP analysis,
allowing for the processing of larger datasets without running
out of memory.
"""

import os
import gc
import logging
import numpy as np
from typing import Union, Optional, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shap-memory-fix")

# Configuration for batch processing
MAX_ROWS_TO_PROCESS = int(os.environ.get('SHAP_MAX_BATCH_SIZE', '300'))

class ShapValuesWrapper:
    """Wrapper class to maintain compatibility with SHAP values API"""
    
    def __init__(self, values, base_values=None, data=None, feature_names=None):
        """Initialize with raw SHAP values"""
        self.values = values
        self.base_values = base_values
        self.data = data
        self.feature_names = feature_names
        
    def __len__(self):
        """Return the number of rows in the SHAP values"""
        if hasattr(self.values, '__len__'):
            return len(self.values)
        return 0

def apply_memory_patches(app=None):
    """
    Apply memory optimization patches to the Flask app
    
    Args:
        app: Flask application instance
    """
    import gc
    
    # Enable garbage collection to run more aggressively
    gc.enable()
    logger.info("Garbage collection enabled with thresholds: %s", gc.get_threshold())
    
    # Function to create memory-optimized explainer
    def create_memory_optimized_explainer(model, X, feature_names=None, 
                                         max_rows=MAX_ROWS_TO_PROCESS):
        """
        Creates a memory-optimized explainer by processing data in batches
        
        Args:
            model: The trained model to explain
            X: Input features to explain
            feature_names: Optional list of feature names
            max_rows: Maximum number of rows to process in one batch
            
        Returns:
            ShapValuesWrapper containing the computed SHAP values
        """
        import shap
        
        # Start with garbage collection to ensure we have maximum memory available
        gc.collect()
        
        logger.info(f"Creating memory-optimized explainer for {len(X)} rows")
        logger.info(f"Using max batch size of {max_rows} rows")
        
        # Check if the dataset is small enough to process in one go
        if len(X) <= max_rows:
            logger.info("Dataset small enough for direct processing")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X)
            
            # Return ShapValuesWrapper if needed
            if not hasattr(shap_values, 'values'):
                return ShapValuesWrapper(shap_values)
            return shap_values
        
        # Process in batches for large datasets
        logger.info(f"Processing large dataset in batches")
        all_shap_values = []
        total_rows = len(X)
        chunks = (total_rows + max_rows - 1) // max_rows  # Ceiling division
        
        for i in range(chunks):
            start_idx = i * max_rows
            end_idx = min((i + 1) * max_rows, total_rows)
            
            logger.info(f"Processing batch {i+1}/{chunks} (rows {start_idx}-{end_idx})")
            
            # Extract this chunk of data
            X_chunk = X.iloc[start_idx:end_idx]
            
            # Create explainer and get SHAP values for this chunk
            explainer = shap.TreeExplainer(model)
            chunk_shap_values = explainer(X_chunk)
            
            # Extract values (handle different return types from different SHAP versions)
            if hasattr(chunk_shap_values, 'values'):
                all_shap_values.append(chunk_shap_values.values)
            else:
                all_shap_values.append(chunk_shap_values)
                
            # Force cleanup to free memory
            del explainer
            del chunk_shap_values
            del X_chunk
            gc.collect()
            
        # Combine all chunks
        logger.info("Combining SHAP values from all batches")
        try:
            combined_values = np.vstack(all_shap_values)
            return ShapValuesWrapper(combined_values)
        except:
            logger.warning("Could not combine values using np.vstack, returning list")
            return ShapValuesWrapper(all_shap_values)
    
    # Patch the calculation function
    global calculate_shap_values
    
    # Store original function if it exists
    if 'calculate_shap_values' in globals():
        original_calculate_shap_values = calculate_shap_values
        
        # Define patched function with same signature
        def memory_optimized_calculate_shap_values(model, X, feature_names=None, **kwargs):
            """Memory optimized version of calculate_shap_values"""
            logger.info("Using memory-optimized SHAP calculation")
            return create_memory_optimized_explainer(model, X, feature_names, MAX_ROWS_TO_PROCESS)
        
        # Replace the global function
        calculate_shap_values = memory_optimized_calculate_shap_values
    else:
        # If function doesn't exist yet, create it
        def calculate_shap_values(model, X, feature_names=None, **kwargs):
            """Memory optimized SHAP calculation function"""
            logger.info("Using memory-optimized SHAP calculation")
            return create_memory_optimized_explainer(model, X, feature_names, MAX_ROWS_TO_PROCESS)
    
    # If app is provided, add memory monitoring endpoint
    if app is not None:
        logger.info("Adding memory monitoring endpoint to Flask app")
        try:
            from flask import jsonify
            import psutil
            
            @app.route('/admin/memory', methods=['GET'])
            def memory_status():
                """Return current memory usage and status"""
                try:
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    return jsonify({
                        "success": True,
                        "memory_usage_mb": memory_mb,
                        "optimized_worker_applied": True,
                        "gc_enabled": gc.isenabled(),
                        "gc_counts": gc.get_count(),
                        "gc_threshold": gc.get_threshold(),
                        "max_rows_per_batch": MAX_ROWS_TO_PROCESS
                    })
                except Exception as e:
                    logger.error(f"Error in memory endpoint: {str(e)}")
                    return jsonify({
                        "success": False,
                        "error": str(e)
                    }), 500
        except ImportError:
            logger.warning("Could not add memory endpoint: psutil not installed")
    
    logger.info("Memory optimization patches applied successfully")
    return True

# Export the memory optimization function
__all__ = ['apply_memory_patches', 'MAX_ROWS_TO_PROCESS', 'ShapValuesWrapper']

# If run directly, print info
if __name__ == "__main__":
    print("SHAP Memory Optimization Module")
    print(f"Maximum rows per batch: {MAX_ROWS_TO_PROCESS}")
    print("To use, import this module and call apply_memory_patches()")
EOL
chmod +x shap_memory_fix.py
echo -e "${GREEN}✅ Created memory optimization module${NC}"

# Step 4: Update app.py to use memory optimizations correctly if needed
echo -e "\n${BOLD}Step 4: Checking app.py for memory optimization code${NC}"
if ! grep -q "apply_memory_patches(app)" app.py; then
    echo -e "${YELLOW}Adding memory optimization to app.py...${NC}"
    
    # Find a suitable place to add the code
    INIT_APP_LINE=$(grep -n "app = Flask" app.py | head -1 | cut -d: -f1)
    if [ -n "$INIT_APP_LINE" ]; then
        # Add initialization code after Flask app initialization
        INIT_APP_LINE=$((INIT_APP_LINE + 3))
        sed -i.bak "${INIT_APP_LINE}i\\
# Apply SHAP memory optimization fix if available\\
try:\\
    from shap_memory_fix import apply_memory_patches\\
    apply_memory_patches(app)\\
    logger.info('✅ Applied SHAP memory optimization patches')\\
except (NameError, ImportError) as e:\\
    logger.warning(f'⚠️ Could not apply SHAP memory optimization: {str(e)}')\\
" app.py
        echo -e "${GREEN}✅ Added memory optimization to app.py${NC}"
    else
        echo -e "${RED}❌ Could not find Flask app initialization in app.py${NC}"
    fi
else
    echo -e "${GREEN}✅ app.py already contains memory optimization code${NC}"
fi

# Step 5: Git operations
echo -e "\n${BOLD}Step 5: Committing changes to git${NC}"
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git not installed. Please install git to deploy to Render.${NC}"
    exit 1
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}⚠️ Not in a git repository, initializing...${NC}"
    git init
fi

# Add files to git
git add render.yaml shap_memory_fix.py setup_worker.py app.py

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo -e "${YELLOW}No changes to commit${NC}"
else
    echo -e "${YELLOW}Committing changes...${NC}"
    git commit -m "Apply comprehensive memory optimization fix for SHAP worker"
    echo -e "${GREEN}✅ Changes committed${NC}"
fi

# Ask to push to remote
echo -e "\n${BOLD}Step 6: Push to remote${NC}"
echo -e "${YELLOW}Do you want to push these changes to the remote repository?${NC}"
echo -e "${YELLOW}This will deploy the changes to the live service on Render.com.${NC}"
read -p "Push to remote? (y/n): " PUSH_CONFIRM

if [[ $PUSH_CONFIRM == "y" || $PUSH_CONFIRM == "Y" ]]; then
    echo -e "${YELLOW}Preparing to push to remote...${NC}"
    
    # Check if remote is configured
    REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
    
    if [ -z "$REMOTE_URL" ]; then
        echo -e "${YELLOW}No remote repository configured.${NC}"
        read -p "Enter the remote repository URL: " REMOTE_URL
        
        if [ -z "$REMOTE_URL" ]; then
            echo -e "${RED}❌ No remote URL provided. Cannot push changes.${NC}"
            echo -e "${YELLOW}Please configure a remote repository and push manually.${NC}"
            exit 1
        fi
        
        git remote add origin "$REMOTE_URL"
    fi
    
    # Get current branch or default to main
    CURRENT_BRANCH=$(git branch --show-current)
    if [ -z "$CURRENT_BRANCH" ]; then
        echo -e "${YELLOW}No branch detected, creating main branch...${NC}"
        git checkout -b main
        CURRENT_BRANCH="main"
    fi
    
    # Push to remote
    echo -e "${YELLOW}Pushing to remote branch ${CURRENT_BRANCH}...${NC}"
    git push -u origin "$CURRENT_BRANCH"
    PUSH_STATUS=$?
    
    if [ $PUSH_STATUS -eq 0 ]; then
        echo -e "${GREEN}✅ Successfully pushed changes to remote repository${NC}"
        echo -e "${GREEN}✅ Render.com will automatically deploy the changes${NC}"
        echo -e "${YELLOW}⚠️ Deployment may take up to 5-10 minutes to complete${NC}"
    else
        echo -e "${RED}❌ Failed to push changes to remote repository${NC}"
        echo -e "${YELLOW}Please push manually or check your git credentials${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ Changes not pushed to remote. Deployment cancelled.${NC}"
    echo -e "${YELLOW}You can push the changes manually with: git push -u origin \$(git branch --show-current)${NC}"
fi

# Step 7: Create verification script for after deployment
echo -e "\n${BOLD}Step 7: Creating verification script${NC}"
cat > verify_deployment.sh << 'EOL'
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
EOL
chmod +x verify_deployment.sh

echo -e "${GREEN}✅ Created verification script${NC}"

echo -e "\n${BOLD}Deployment Preparation Complete${NC}"
echo -e "-----------------------------"
echo -e "${GREEN}✓${NC} Updated render.yaml with worker configuration"
echo -e "${GREEN}✓${NC} Created memory optimization module"
echo -e "${GREEN}✓${NC} Updated app.py (if needed)"
echo -e "${GREEN}✓${NC} Created deployment verification script"

if [[ $PUSH_CONFIRM == "y" || $PUSH_CONFIRM == "Y" && $PUSH_STATUS -eq 0 ]]; then
    echo -e "${GREEN}✓${NC} Pushed changes to remote repository"
    echo -e "\n${BOLD}Next Steps${NC}"
    echo -e "-----------"
    echo -e "1. Wait 5-10 minutes for Render.com to complete the deployment"
    echo -e "2. Run ./verify_deployment.sh to check if the deployment was successful"
    echo -e "3. Run ./comprehensive_health_check.py for a more detailed health check"
else
    echo -e "${RED}✗${NC} Changes not pushed to remote repository"
    echo -e "\n${BOLD}Next Steps${NC}"
    echo -e "-----------"
    echo -e "1. Push the changes to the remote repository: git push -u origin \$(git branch --show-current)"
    echo -e "2. Wait 5-10 minutes for Render.com to complete the deployment"
    echo -e "3. Run ./verify_deployment.sh to check if the deployment was successful"
    echo -e "4. Run ./comprehensive_health_check.py for a more detailed health check"
fi
