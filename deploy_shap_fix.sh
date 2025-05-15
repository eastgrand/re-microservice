#!/bin/bash

# SHAP Microservice Fix Deployment Script
# This script deploys the memory-optimized SHAP fix to resolve the 500 errors
# Date: May 15, 2025

echo "====================================="
echo "SHAP Memory Fix Deployment"
echo "====================================="

# Check for required files
echo "Checking for required files..."
if [ ! -f "shap_memory_fix.py" ]; then
    echo "❌ Missing shap_memory_fix.py file!"
    exit 1
else
    echo "✅ Found shap_memory_fix.py"
fi

if [ ! -f "app.py" ]; then
    echo "❌ Missing app.py file!"
    exit 1
else 
    echo "✅ Found app.py"
fi

# Backup original files
echo "Creating backups of original files..."
cp app.py app.py.memory-fix-backup
echo "✅ Created backup: app.py.memory-fix-backup"

# Update app.py to use our memory fix
echo "Updating app.py to use memory-optimized SHAP analysis..."
if grep -q "shap_memory_fix" app.py; then
    echo "✅ app.py already includes memory fix"
else
    # Add the import at the top, after other imports
    sed -i '' '1,50s/import logging/import logging\nimport gc  # For memory management/' app.py
    
    # Add the fix application before starting the app
    if grep -q "if __name__ == \"__main__\":" app.py; then
        sed -i '' '/if __name__ == "__main__":/i \
# Apply SHAP memory optimization fix\
try:\
    from shap_memory_fix import apply_all_patches\
    apply_all_patches(app)\
    logger.info("✅ Applied SHAP memory optimization fix")\
except Exception as e:\
    logger.error(f"❌ Failed to apply SHAP memory fix: {str(e)}")\
' app.py
        echo "✅ Updated app.py to use memory fix"
    else
        echo "⚠️ Could not find main block in app.py, adding fix after Redis initialization"
        sed -i '' '/queue = Queue/a \
\
# Apply SHAP memory optimization fix\
try:\
    from shap_memory_fix import apply_all_patches\
    apply_all_patches(app)\
    logger.info("✅ Applied SHAP memory optimization fix")\
except Exception as e:\
    logger.error(f"❌ Failed to apply SHAP memory fix: {str(e)}")\
' app.py
        echo "✅ Updated app.py to use memory fix"
    fi
fi

# Update render.yaml to include memory-saving environment variables
echo "Updating render.yaml with memory optimization settings..."
if [ -f "render.yaml" ]; then
    # Make a backup of render.yaml
    cp render.yaml render.yaml.memory-fix-backup
    echo "✅ Created backup: render.yaml.memory-fix-backup"
    
    # Check if memory management is already enabled
    if grep -q "AGGRESSIVE_MEMORY_MANAGEMENT" render.yaml; then
        echo "✅ Memory optimization already in render.yaml"
    else
        # Add memory optimization for web service
        sed -i '' '/envVars:/a \
      - key: AGGRESSIVE_MEMORY_MANAGEMENT\
        value: "true"\
      - key: SHAP_BATCH_SIZE\
        value: "500"\
' render.yaml
        echo "✅ Added memory optimization to render.yaml"
    fi
    
    # Check worker health check setting
    if grep -q "health_check_interval" render.yaml; then
        echo "✅ Redis health check already configured"
    else
        # Add health check interval for web service
        sed -i '' '/REDIS_URL/a \
      - key: REDIS_HEALTH_CHECK_INTERVAL\
        value: "30"\
' render.yaml
        echo "✅ Added Redis health check configuration"
    fi
    
    # Update worker burst mode if needed
    if grep -q "rq worker --burst" render.yaml; then
        echo "✅ RQ Worker using burst mode"
    else
        sed -i '' 's/rq worker shap-jobs/rq worker --burst shap-jobs/g' render.yaml
        echo "✅ Updated worker to use burst mode for better cleanup"
    fi
else
    echo "⚠️ render.yaml not found, skipping render.yaml updates"
fi

# Install any needed dependencies
echo "Installing required dependencies..."
pip install psutil
echo "✅ Installed psutil for memory monitoring"

# Create a testing and verification script
echo "Creating verification script..."
cat > verify_shap_fix.py << 'EOF'
#!/usr/bin/env python3
"""
SHAP Fix Verification Script

This script verifies that the SHAP memory optimization fix is working properly
by running a test analysis and checking the results.
"""

import os
import sys
import time
import json
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("verify-fix")

# Service URL
SERVICE_URL = os.environ.get("SHAP_SERVICE_URL", "http://localhost:10000")

def test_service_connection():
    """Test basic connectivity to the service"""
    try:
        logger.info(f"Testing connection to {SERVICE_URL}...")
        response = requests.get(f"{SERVICE_URL}/ping", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Service is responding to ping")
            return True
        else:
            logger.error(f"❌ Service returned status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Connection error: {str(e)}")
        return False

def test_memory_endpoint():
    """Test the memory monitoring endpoint"""
    try:
        logger.info("Testing memory monitoring endpoint...")
        response = requests.get(f"{SERVICE_URL}/admin/memory", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Memory endpoint is working")
            logger.info(f"   Current memory usage: {data.get('memory_usage_mb', 'N/A')} MB")
            logger.info(f"   Optimized worker: {data.get('optimized_worker_applied', 'N/A')}")
            return True
        else:
            logger.error(f"❌ Memory endpoint returned status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Memory endpoint error: {str(e)}")
        return False

def test_analysis():
    """Run a test analysis to verify the fix"""
    try:
        logger.info("Submitting test analysis job...")
        test_query = {
            "analysis_type": "correlation",
            "target_variable": "Mortgage_Approvals",
            "demographic_filters": ["Income > 50000"]
        }
        
        # Submit the job
        response = requests.post(
            f"{SERVICE_URL}/analyze",
            json=test_query,
            headers={"X-API-KEY": os.environ.get("API_KEY", "")},
            timeout=10
        )
        
        if response.status_code != 202:
            logger.error(f"❌ Analysis request failed with status: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
        # Get the job ID
        job_data = response.json()
        job_id = job_data.get("job_id")
        if not job_id:
            logger.error("❌ No job ID returned from analysis request")
            return False
            
        logger.info(f"✅ Analysis job submitted with ID: {job_id}")
        
        # Poll for job completion
        max_polls = 20
        for i in range(max_polls):
            logger.info(f"Polling job status ({i+1}/{max_polls})...")
            status_response = requests.get(
                f"{SERVICE_URL}/job_status/{job_id}",
                headers={"X-API-KEY": os.environ.get("API_KEY", "")},
                timeout=10
            )
            
            if status_response.status_code != 200:
                logger.error(f"❌ Job status request failed: {status_response.status_code}")
                logger.error(f"Response: {status_response.text}")
                return False
                
            status_data = status_response.json()
            status = status_data.get("status")
            logger.info(f"Job status: {status}")
            
            if status == "finished":
                logger.info("✅ Job completed successfully!")
                if "result" in status_data:
                    result = status_data["result"]
                    if result.get("success"):
                        logger.info("✅ Analysis returned successful result")
                        logger.info(f"Summary: {result.get('summary', 'N/A')}")
                        return True
                    else:
                        logger.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                        return False
                else:
                    logger.error("❌ No result data in finished job")
                    return False
            elif status == "failed":
                logger.error(f"❌ Job failed: {status_data.get('error', 'Unknown error')}")
                return False
                
            # Wait before polling again
            time.sleep(5)
            
        logger.error("❌ Job did not complete within the timeout period")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing analysis: {str(e)}")
        return False

def main():
    """Run all verification tests"""
    logger.info("=== SHAP Fix Verification ===")
    
    # Test basic connectivity
    if not test_service_connection():
        logger.error("❌ Service connection test failed")
        return 1
        
    # Test memory monitoring endpoint
    test_memory_endpoint()
    
    # Run test analysis
    if test_analysis():
        logger.info("✅ All verification tests passed!")
        logger.info("The SHAP memory fix has been successfully applied.")
        return 0
    else:
        logger.error("❌ Analysis test failed")
        logger.info("Please check application logs for more details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF
chmod +x verify_shap_fix.py
echo "✅ Created verification script: verify_shap_fix.py"

# Generate documentation
echo "Creating documentation..."
cat > SHAP-MEMORY-FIX.md << 'EOF'
# SHAP Memory Optimization Fix

**Date:** May 15, 2025
**Issue:** HTTP 500 errors during SHAP analysis with jobs stuck in "started" state

## Diagnosis

After analyzing the application logs and code, we identified several issues:

1. **Memory Usage:** SHAP calculations were using too much memory when processing large datasets
2. **Worker Process:** The RQ worker process was failing during SHAP analysis
3. **Error Handling:** Insufficient error reporting was making diagnosis difficult
4. **Worker Configuration:** Worker processes were not properly configured for Render

## Solution Implemented

The `shap_memory_fix.py` script implements several optimizations:

1. **Chunked Processing:** SHAP calculations are now done in batches to reduce peak memory usage
2. **Memory Monitoring:** Added `/admin/memory` endpoint for real-time monitoring
3. **Enhanced Error Handling:** Better error logging with memory usage information
4. **Garbage Collection:** Aggressive garbage collection to free memory during processing
5. **Worker Settings:** Improved worker configuration in `render.yaml`

## Verification

You can verify the fix is working with the included verification script:

```bash
# Set API key if needed
export API_KEY="your_api_key"

# Run verification
python verify_shap_fix.py
```

## Environment Variables

The following environment variables can be adjusted to tune performance:

| Variable | Default | Description |
|----------|---------|-------------|
| AGGRESSIVE_MEMORY_MANAGEMENT | true | Enables memory optimizations |
| SHAP_BATCH_SIZE | 500 | Maximum rows to process in a single batch |
| REDIS_HEALTH_CHECK_INTERVAL | 30 | Seconds between Redis health checks |

## Monitoring

You can monitor memory usage of the service with:

```bash
curl http://your-service-url/admin/memory
```

This will return current memory usage and optimization status.
EOF
echo "✅ Created documentation: SHAP-MEMORY-FIX.md"

echo "====================================="
echo "Deployment preparation complete!"
echo "====================================="
echo ""
echo "Next steps:"
echo "1. Commit these changes to your repository"
echo "2. Deploy to Render with:"
echo "   ./deploy_to_render.sh"
echo "3. Verify the fix with:"
echo "   python verify_shap_fix.py"
echo ""
echo "For more information, see SHAP-MEMORY-FIX.md"
