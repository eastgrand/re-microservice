#!/usr/bin/env python3
"""
Test SHAP Microservice Deployment

This script tests the deployed SHAP microservice by:
1. Verifying basic connectivity (ping)
2. Checking Redis health endpoints
3. Submitting a small test job
4. Polling for job results
"""

import os
import sys
import json
import time
import requests
from urllib.parse import urljoin
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("deployment-test")

# Configuration
DEFAULT_URL = "https://nesto-mortgage-analytics.onrender.com"
DEFAULT_API_KEY = "HFqkccbN3LV5CaB"  # Default key for testing
REQUEST_TIMEOUT = 15  # seconds

def make_request(url, method="GET", headers=None, json_data=None, timeout=REQUEST_TIMEOUT):
    """Make an HTTP request with proper error handling"""
    if headers is None:
        headers = {}
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=timeout)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data, timeout=timeout)
        else:
            logger.error(f"Unsupported HTTP method: {method}")
            return None
        
        # Try to parse as JSON
        try:
            result = response.json()
        except:
            result = {"text": response.text}
        
        # Include status in the result
        result["_status"] = response.status_code
        
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return {"error": str(e), "_status": 0}

def test_basic_connectivity(base_url, api_key=None):
    """Test basic service connectivity"""
    
    logger.info(f"Testing basic connectivity to {base_url}...")
    
    ping_url = urljoin(base_url, "/ping")
    headers = {}
    if api_key:
        headers["X-API-KEY"] = api_key
        
    response = make_request(ping_url, headers=headers)
    
    if response and response.get("_status") == 200:
        logger.info(f"✅ Basic connectivity test passed: {json.dumps(response, indent=2)}")
        return True
    else:
        logger.error(f"❌ Basic connectivity test failed: {json.dumps(response, indent=2)}")
        return False

def test_redis_health(base_url, api_key=None):
    """Test the Redis health endpoint"""
    
    logger.info(f"Testing Redis health at {base_url}...")
    
    # Try both admin/redis_ping and regular health endpoint
    redis_urls = [
        urljoin(base_url, "/admin/redis_ping"),
        urljoin(base_url, "/health")
    ]
    
    headers = {}
    if api_key:
        headers["X-API-KEY"] = api_key
        
    redis_ok = False
    
    for url in redis_urls:
        logger.info(f"Checking endpoint: {url}")
        response = make_request(url, headers=headers)
        
        if response and response.get("_status") == 200:
            # Check different indicators based on endpoint
            if "admin/redis_ping" in url:
                if response.get("success") and response.get("ping"):
                    logger.info(f"✅ Redis ping endpoint returned successful: {json.dumps(response, indent=2)}")
                    redis_ok = True
                else:
                    logger.warning(f"⚠️ Redis ping endpoint returned, but indicates issues: {json.dumps(response, indent=2)}")
            elif "health" in url:
                if response.get("redis_connected"):
                    logger.info(f"✅ Health endpoint reports Redis is connected: {json.dumps(response, indent=2)}")
                    redis_ok = True
                else:
                    logger.warning(f"⚠️ Health endpoint indicates Redis is NOT connected: {json.dumps(response, indent=2)}")
        else:
            logger.warning(f"⚠️ Redis health check failed at {url}: {json.dumps(response, indent=2)}")
    
    return redis_ok

def submit_test_job(base_url, api_key=None):
    """Submit a small test job to the service"""
    
    logger.info(f"Submitting test analysis job to {base_url}...")
    
    analyze_url = urljoin(base_url, "/analyze")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-KEY"] = api_key
        
    # Simple test data
    test_data = {
        "analysis_type": "correlation",
        "target_variable": "Mortgage_Approvals",
        "demographic_filters": ["Income > 50000"]
    }
    
    response = make_request(analyze_url, method="POST", headers=headers, json_data=test_data)
    
    if response and response.get("_status") == 202:
        job_id = response.get("job_id")
        if job_id:
            logger.info(f"✅ Job submitted successfully! Job ID: {job_id}")
            return job_id
        else:
            logger.error(f"❌ Job submitted but no job_id returned: {json.dumps(response, indent=2)}")
            return None
    else:
        logger.error(f"❌ Failed to submit job: {json.dumps(response, indent=2)}")
        return None

def poll_job_status(base_url, job_id, api_key=None, max_retries=10, delay=5):
    """Poll for job status until complete or max retries reached"""
    
    logger.info(f"Polling job status for job: {job_id}...")
    
    job_status_url = urljoin(base_url, f"/job_status/{job_id}")
    headers = {}
    if api_key:
        headers["X-API-KEY"] = api_key
    
    for attempt in range(max_retries):
        logger.info(f"Polling attempt {attempt+1}/{max_retries}...")
        
        response = make_request(job_status_url, headers=headers)
        
        if response and response.get("_status") == 200:
            status = response.get("status")
            
            logger.info(f"Job status: {status}")
            
            if status == "finished":
                logger.info(f"✅ Job completed successfully!")
                if "result" in response:
                    result_summary = response.get("result", {}).get("summary", "No summary available")
                    feature_importance = response.get("result", {}).get("feature_importance", [])
                    
                    logger.info(f"Result summary: {result_summary}")
                    if feature_importance:
                        logger.info(f"Top features: {json.dumps(feature_importance[:3], indent=2)}")
                
                return True
            elif status == "failed":
                logger.error(f"❌ Job failed: {json.dumps(response, indent=2)}")
                return False
            else:
                # Still processing, wait and retry
                logger.info(f"Job is still processing ({status}), waiting {delay} seconds...")
                time.sleep(delay)
        else:
            logger.error(f"❌ Error checking job status: {json.dumps(response, indent=2)}")
            time.sleep(delay)
    
    logger.error(f"❌ Job did not complete within {max_retries} polling attempts")
    return False

def main():
    # Get base URL and API key
    base_url = os.environ.get("SHAP_SERVICE_URL", DEFAULT_URL)
    api_key = os.environ.get("API_KEY", DEFAULT_API_KEY)
    
    # Override from command line if provided
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    logger.info(f"Testing SHAP microservice deployment at {base_url}")
    
    # Step 1: Test basic connectivity
    if not test_basic_connectivity(base_url, api_key):
        logger.error("Basic connectivity test failed, cannot proceed.")
        sys.exit(1)
        
    # Step 2: Test Redis health
    redis_ok = test_redis_health(base_url, api_key)
    if not redis_ok:
        logger.warning("Redis health check failed, but continuing with job submission test.")
    
    # Step 3: Submit a test job
    job_id = submit_test_job(base_url, api_key)
    if not job_id:
        logger.error("Failed to submit job, cannot test job processing.")
        sys.exit(1)
    
    # Step 4: Poll for job results
    job_completed = poll_job_status(base_url, job_id, api_key)
    
    # Summary
    logger.info("\nTest Summary:")
    logger.info(f"- Basic Connectivity: ✅ Passed")
    logger.info(f"- Redis Health: {'✅ Passed' if redis_ok else '❌ Failed'}")
    logger.info(f"- Job Submission: ✅ Passed (Job ID: {job_id})")
    logger.info(f"- Job Processing: {'✅ Passed' if job_completed else '❌ Failed'}")
    
    # Exit with appropriate status code
    if job_completed:
        logger.info("✅ All critical tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ One or more critical tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
