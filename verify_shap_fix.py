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
