#!/usr/bin/env python3
# filepath: /Users/voldeck/code/shap-microservice/check_worker_status.py
"""
SHAP Worker Status Check Script

This script checks the status of worker processes on the deployed SHAP microservice.
"""

import os
import sys
import json
import requests
import logging
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("worker-status")

# Service URL
DEFAULT_URL = "https://nesto-mortgage-analytics.onrender.com"
SERVICE_URL = os.environ.get("SHAP_SERVICE_URL", DEFAULT_URL)

# Default API Key
DEFAULT_API_KEY = "HFqkccbN3LV5CaB"
API_KEY = os.environ.get("API_KEY", DEFAULT_API_KEY)

def check_service_status():
    """Check the basic service status"""
    try:
        logger.info(f"Checking service at {SERVICE_URL}...")
        
        # Check ping endpoint
        response = requests.get(f"{SERVICE_URL}/ping", timeout=10)
        
        if response.status_code == 200:
            logger.info(f"✅ Service is up and responding to ping")
            return True
        else:
            logger.error(f"❌ Service ping failed with status code {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Service connection error: {str(e)}")
        return False

def check_queue_status():
    """Check Redis queue status"""
    try:
        logger.info("Checking queue status...")
        response = requests.get(
            f"{SERVICE_URL}/admin/queue_status",
            headers={"x-api-key": API_KEY},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Queue status endpoint working")
            logger.info(f"Queue stats:")
            logger.info(f"  - Jobs queued: {data.get('queued_jobs', 'N/A')}")
            logger.info(f"  - Jobs in progress: {data.get('in_progress_jobs', 'N/A')}")
            logger.info(f"  - Jobs completed: {data.get('completed_jobs', 'N/A')}")
            logger.info(f"  - Redis connection: {'✅ Working' if data.get('redis_connected', False) else '❌ Failed'}")
            return True
        else:
            logger.error(f"❌ Queue status check failed with status code {response.status_code}")
            if response.text:
                logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Queue status check error: {str(e)}")
        return False

def check_worker_diagnostics():
    """Check worker diagnostics endpoint if available"""
    try:
        logger.info("Checking worker diagnostics...")
        response = requests.get(
            f"{SERVICE_URL}/admin/worker_status",
            headers={"x-api-key": API_KEY},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Worker diagnostics endpoint working")
            logger.info(f"Worker stats:")
            logger.info(f"  - Active workers: {data.get('active_workers', 'N/A')}")
            logger.info(f"  - Worker errors: {data.get('worker_errors', 'N/A')}")
            logger.info(f"  - Last job processed: {data.get('last_processed_job_time', 'N/A')}")
            
            # Calculate time since last job if available
            last_job_time = data.get('last_processed_job_time')
            if last_job_time:
                try:
                    last_job_dt = datetime.fromisoformat(last_job_time.replace('Z', '+00:00'))
                    now = datetime.now(timezone.utc)
                    delta = now - last_job_dt
                    logger.info(f"  - Time since last job: {delta}")
                except:
                    pass
                    
            return True
        else:
            logger.error(f"❌ Worker diagnostics check failed with status code {response.status_code}")
            if response.text:
                logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Worker diagnostics check error: {str(e)}")
        return False

def check_memory_status():
    """Check memory status endpoint"""
    try:
        logger.info("Checking memory status...")
        response = requests.get(
            f"{SERVICE_URL}/admin/memory",
            headers={"x-api-key": API_KEY},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Memory status endpoint working")
            logger.info(f"Memory stats:")
            logger.info(f"  - Current memory usage: {data.get('memory_usage_mb', 'N/A')} MB")
            logger.info(f"  - Optimized worker: {'✅ Applied' if data.get('optimized_worker_applied', False) else '❌ Not applied'}")
            logger.info(f"  - GC enabled: {data.get('gc_enabled', 'N/A')}")
            logger.info(f"  - Max rows per batch: {data.get('max_rows_per_batch', 'N/A')}")
            return True
        else:
            logger.error(f"❌ Memory status check failed with status code {response.status_code}")
            if response.text:
                logger.error(f"Response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Memory status check error: {str(e)}")
        return False

def main():
    """Main function"""
    logger.info("=== SHAP Worker Status Check ===")
    logger.info(f"Service URL: {SERVICE_URL}")
    logger.info(f"API Key: {'✅ Set' if API_KEY else '❌ Not set'}")
    
    # Check basic service status
    if not check_service_status():
        logger.error("❌ Basic service check failed")
        return 1
        
    # Try checking different diagnostic endpoints
    queue_ok = check_queue_status()
    worker_ok = check_worker_diagnostics()
    memory_ok = check_memory_status()
    
    if queue_ok and worker_ok and memory_ok:
        logger.info("✅ All diagnostics passed")
        return 0
    else:
        logger.warning("⚠️ Some diagnostic checks failed")
        
        if not memory_ok:
            logger.warning("⚠️ Memory optimization may not be applied correctly")
        
        return 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        SERVICE_URL = sys.argv[1]
    sys.exit(main())
