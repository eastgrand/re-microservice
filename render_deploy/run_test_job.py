#!/usr/bin/env python3
"""
SHAP Test Job Runner
- Submits a test job to the SHAP microservice
- Monitors the job status until completion
- Verifies the results are correctly returned
"""

import os
import sys
import time
import json
import logging
import argparse
import requests
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test-job-runner")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a test job through the SHAP microservice')
parser.add_argument('--service-url', help='URL for the web service', 
                   default=os.environ.get('SHAP_SERVICE_URL', 'http://localhost:10000'))
parser.add_argument('--api-key', help='API key for the service', 
                   default=os.environ.get('API_KEY', ''))
parser.add_argument('--timeout', type=int, default=300, 
                   help='Maximum time to wait for job completion (seconds)')
parser.add_argument('--sample-size', type=int, default=100, 
                   help='Number of rows to use for the test (smaller=faster)')

def run_test_job(service_url, api_key, timeout=300, sample_size=100):
    """Run a test job through the SHAP microservice"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-KEY"] = api_key
    
    # Prepare the test job payload
    job_data = {
        "analysis_type": "correlation",
        "target_variable": "Mortgage_Approvals",
        "sample_size": sample_size,  # Use a smaller sample for faster testing
        "demographic_filters": []  # No filters for simple test
    }
    
    # Submit the job
    logger.info(f"Submitting test job to {service_url}/analyze")
    try:
        response = requests.post(
            f"{service_url}/analyze", 
            headers=headers,
            json=job_data,
            timeout=30
        )
        
        if response.status_code != 202:
            logger.error(f"Failed to submit job. Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
        # Get the job ID
        job_data = response.json()
        job_id = job_data.get("job_id")
        
        if not job_id:
            logger.error("No job ID returned")
            return False
            
        logger.info(f"Job submitted successfully. Job ID: {job_id}")
        
        # Poll for job completion
        start_time = time.time()
        elapsed_time = 0
        status = "pending"
        
        while elapsed_time < timeout and status not in ["completed", "failed", "error"]:
            # Wait before polling again
            time.sleep(5)
            
            # Check job status
            logger.info(f"Checking status of job {job_id}...")
            try:
                status_response = requests.get(
                    f"{service_url}/job_status/{job_id}",
                    headers=headers,
                    timeout=10
                )
                
                if status_response.status_code != 200:
                    logger.warning(f"Failed to get job status. Status code: {status_response.status_code}")
                    logger.warning(f"Response: {status_response.text}")
                else:
                    status_data = status_response.json()
                    status = status_data.get("status")
                    logger.info(f"Job status: {status}")
                    
                    if status == "processing":
                        # Log progress if available
                        if "progress" in status_data:
                            progress = status_data["progress"]
                            logger.info(f"Progress: {progress}%")
                    
                    elif status == "completed":
                        # Job completed successfully
                        logger.info(f"Job completed in {elapsed_time:.1f} seconds")
                        
                        # Check if results are available
                        if "result" in status_data:
                            result = status_data["result"]
                            logger.info(f"Job result summary: {len(result)} data points returned")
                            
                            # Verify result structure
                            if isinstance(result, dict) and "shap_values" in result:
                                logger.info("✅ Result contains SHAP values - SUCCESS")
                                
                                # Save the result to a file for further analysis
                                with open(f"test_job_result_{job_id}.json", "w") as f:
                                    json.dump(status_data, f, indent=2)
                                logger.info(f"Results saved to test_job_result_{job_id}.json")
                                
                                return True
                            else:
                                logger.error("Result does not contain expected SHAP values")
                                return False
                        else:
                            logger.error("Job completed but no results returned")
                            return False
                    
                    elif status == "failed" or status == "error":
                        # Job failed
                        logger.error(f"Job failed: {status_data.get('error', 'Unknown error')}")
                        return False
            
            except Exception as e:
                logger.warning(f"Error checking job status: {str(e)}")
            
            elapsed_time = time.time() - start_time
        
        if status not in ["completed"]:
            logger.error(f"Job did not complete within timeout ({timeout} seconds)")
            return False
        
    except Exception as e:
        logger.error(f"Error running test job: {str(e)}", exc_info=True)
        return False

def main():
    """Main function to run a test job"""
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info(f"SHAP Microservice Test Job - {datetime.now().isoformat()}")
    logger.info("=" * 60)
    logger.info(f"Service URL: {args.service_url}")
    logger.info(f"Timeout: {args.timeout} seconds")
    logger.info(f"Sample size: {args.sample_size} rows")
    
    success = run_test_job(args.service_url, args.api_key, args.timeout, args.sample_size)
    
    if success:
        logger.info("=" * 60)
        logger.info("✅ Test job completed successfully!")
        logger.info("=" * 60)
        return 0
    else:
        logger.info("=" * 60)
        logger.info("❌ Test job failed!")
        logger.info("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
