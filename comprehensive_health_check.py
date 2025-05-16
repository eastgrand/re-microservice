#!/usr/bin/env python3
# filepath: /Users/voldeck/code/shap-microservice/comprehensive_health_check.py
"""
Comprehensive Health Check for SHAP Microservice

This script performs a comprehensive health check on the SHAP microservice,
checking all critical components and providing detailed diagnostics.
"""

import os
import sys
import time
import json
import logging
import requests
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("health-check")

# Service URL
DEFAULT_URL = "https://nesto-mortgage-analytics.onrender.com"
SERVICE_URL = os.environ.get("SHAP_SERVICE_URL", DEFAULT_URL)

# Default API Key
DEFAULT_API_KEY = "HFqkccbN3LV5CaB"
API_KEY = os.environ.get("API_KEY", DEFAULT_API_KEY)

# Test data
TEST_ANALYSIS = {
    "analysis_type": "correlation",
    "target_variable": "Mortgage_Approvals",
    "demographic_filters": ["Income > 50000"]
}

class HealthCheck:
    """Health check manager for the SHAP microservice"""
    
    def __init__(self, service_url, api_key):
        """Initialize the health check"""
        self.service_url = service_url.rstrip('/')
        self.api_key = api_key
        self.headers = {"X-API-KEY": api_key} if api_key else {}
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0
        
    def log_success(self, message):
        """Log a success message"""
        logger.info(f"✅ {message}")
        self.checks_passed += 1
        
    def log_warning(self, message):
        """Log a warning message"""
        logger.warning(f"⚠️ {message}")
        self.warnings += 1
        
    def log_failure(self, message):
        """Log a failure message"""
        logger.error(f"❌ {message}")
        self.checks_failed += 1
    
    def check_basic_connectivity(self):
        """Check basic connectivity to the service"""
        logger.info("Checking basic connectivity...")
        try:
            response = requests.get(f"{self.service_url}/ping", timeout=10)
            if response.status_code == 200:
                self.log_success("Service is responding to ping")
                return True
            else:
                self.log_failure(f"Service returned status code {response.status_code} for ping")
                return False
        except Exception as e:
            self.log_failure(f"Connection error: {str(e)}")
            return False
    
    def check_health_endpoint(self):
        """Check the health endpoint"""
        logger.info("Checking health endpoint...")
        try:
            response = requests.get(
                f"{self.service_url}/health",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.log_success("Health endpoint reports service is healthy")
                    return True
                else:
                    self.log_warning(f"Health endpoint reports status: {data.get('status')}")
                    return False
            else:
                self.log_failure(f"Health endpoint returned status code {response.status_code}")
                return False
        except Exception as e:
            self.log_failure(f"Health endpoint error: {str(e)}")
            return False
    
    def check_redis_connection(self):
        """Check the Redis connection"""
        logger.info("Checking Redis connection...")
        try:
            response = requests.get(
                f"{self.service_url}/admin/redis_ping",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    ping_time = data.get("response_time_ms", 0)
                    self.log_success(f"Redis connection is working (ping: {ping_time} ms)")
                    return True
                else:
                    self.log_failure("Redis ping failed")
                    return False
            else:
                self.log_failure(f"Redis ping endpoint returned status code {response.status_code}")
                return False
        except Exception as e:
            self.log_failure(f"Redis ping error: {str(e)}")
            return False
    
    def check_memory_optimization(self):
        """Check if memory optimization is applied"""
        logger.info("Checking memory optimization...")
        try:
            response = requests.get(
                f"{self.service_url}/admin/memory",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("optimized_worker_applied"):
                    self.log_success("Memory optimization is applied")
                    logger.info(f"Current memory usage: {data.get('memory_usage_mb')} MB")
                    logger.info(f"Max rows per batch: {data.get('max_rows_per_batch')}")
                    return True
                else:
                    self.log_warning("Memory optimization is not applied")
                    return False
            else:
                self.log_failure(f"Memory endpoint returned status code {response.status_code}")
                return False
        except Exception as e:
            self.log_failure(f"Memory endpoint error: {str(e)}")
            return False
    
    def check_queue_status(self):
        """Check the queue status"""
        logger.info("Checking queue status...")
        try:
            response = requests.get(
                f"{self.service_url}/admin/queue_status",
                headers=self.headers,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                queued = data.get("queued_jobs", 0)
                in_progress = data.get("in_progress_jobs", 0)
                completed = data.get("completed_jobs", 0)
                
                self.log_success(f"Queue status: {queued} queued, {in_progress} in progress, {completed} completed")
                
                if queued > 5:
                    self.log_warning(f"Large number of queued jobs: {queued}")
                    
                return True
            else:
                self.log_failure(f"Queue status endpoint returned status code {response.status_code}")
                return False
        except Exception as e:
            self.log_failure(f"Queue status error: {str(e)}")
            return False
    
    def test_analysis_flow(self, wait_time=120):
        """Test the full analysis flow"""
        logger.info("Testing analysis flow...")
        
        # Submit a job
        try:
            response = requests.post(
                f"{self.service_url}/analyze",
                json=TEST_ANALYSIS,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code != 202:
                self.log_failure(f"Job submission failed with status code {response.status_code}")
                return False
                
            data = response.json()
            job_id = data.get("job_id")
            
            if not job_id:
                self.log_failure("No job ID returned")
                return False
                
            self.log_success(f"Job submitted with ID {job_id}")
            
            # Poll for job completion
            logger.info(f"Polling for job completion (timeout: {wait_time}s)...")
            start_time = time.time()
            poll_interval = 5
            polls = 0
            
            while time.time() - start_time < wait_time:
                polls += 1
                try:
                    status_response = requests.get(
                        f"{self.service_url}/job_status/{job_id}",
                        headers=self.headers,
                        timeout=10
                    )
                    
                    if status_response.status_code != 200:
                        self.log_warning(f"Job status check failed with code {status_response.status_code}")
                        time.sleep(poll_interval)
                        continue
                        
                    status_data = status_response.json()
                    status = status_data.get("status")
                    logger.info(f"Poll {polls}: Job status is {status}")
                    
                    if status == "finished":
                        self.log_success("Job completed successfully")
                        return True
                    elif status == "failed":
                        self.log_failure(f"Job failed: {status_data.get('error')}")
                        return False
                    elif status == "started":
                        # This is progress from queued
                        self.log_success("Job is being processed (started status)")
                        
                    # Wait before polling again
                    time.sleep(poll_interval)
                except Exception as e:
                    logger.error(f"Error checking job status: {str(e)}")
                    time.sleep(poll_interval)
            
            # If we get here, the job didn't complete in time
            self.log_warning(f"Job did not complete within {wait_time} seconds")
            
            # Check if we at least moved beyond queued status
            try:
                final_status_response = requests.get(
                    f"{self.service_url}/job_status/{job_id}",
                    headers=self.headers,
                    timeout=10
                )
                
                if final_status_response.status_code == 200:
                    final_status_data = final_status_response.json()
                    final_status = final_status_data.get("status")
                    
                    if final_status != "queued":
                        self.log_success(f"Job progressed to {final_status} status")
                        return True
                    else:
                        self.log_failure("Job remained in queued status")
                        return False
            except:
                pass
                
            return False
        except Exception as e:
            self.log_failure(f"Error in analysis flow test: {str(e)}")
            return False
    
    def run_all_checks(self, skip_analysis_test=False):
        """Run all health checks"""
        logger.info(f"===== SHAP Microservice Health Check =====")
        logger.info(f"Service URL: {self.service_url}")
        logger.info(f"API Key: {'Set' if self.api_key else 'Not set'}")
        
        # Run the checks
        basic_ok = self.check_basic_connectivity()
        
        if not basic_ok:
            logger.error("Basic connectivity failed, aborting remaining checks")
            return False
            
        health_ok = self.check_health_endpoint()
        redis_ok = self.check_redis_connection()
        memory_ok = self.check_memory_optimization()
        queue_ok = self.check_queue_status()
        
        # Only run analysis test if not skipped
        analysis_ok = True
        if not skip_analysis_test:
            analysis_ok = self.test_analysis_flow()
        else:
            logger.info("Skipping analysis flow test")
        
        # Summarize results
        logger.info("===== Health Check Summary =====")
        logger.info(f"Checks passed: {self.checks_passed}")
        logger.info(f"Checks failed: {self.checks_failed}")
        logger.info(f"Warnings: {self.warnings}")
        
        overall_status = "HEALTHY" if self.checks_failed == 0 else "UNHEALTHY"
        if self.checks_failed == 0 and self.warnings > 0:
            overall_status = "DEGRADED"
            
        logger.info(f"Overall status: {overall_status}")
        
        return self.checks_failed == 0

def main():
    """Main function"""
    # Parse command line arguments
    skip_analysis = "--skip-analysis" in sys.argv
    
    # Allow command line URL override
    service_url = SERVICE_URL
    for arg in sys.argv:
        if arg.startswith("http"):
            service_url = arg
            break
    
    # Run the health check
    checker = HealthCheck(service_url, API_KEY)
    success = checker.run_all_checks(skip_analysis)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
