#!/usr/bin/env python3
"""
Direct Worker Process Test

This script allows you to run the analysis_worker function directly
outside of the Redis/RQ queue to diagnose issues with the worker function itself.
"""

import os
import sys
import json
import time
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("worker-test")

def run_analysis_worker_direct():
    """Run the analysis_worker function directly without using the queue"""
    try:
        # Import the analysis_worker function directly from app.py
        logger.info("Importing analysis_worker from app...")
        from app import analysis_worker, ensure_model_loaded
        
        # Ensure model is loaded
        logger.info("Ensuring model is loaded...")
        ensure_model_loaded()
        
        # Create a simple test query
        test_query = {
            "analysis_type": "correlation",
            "target_variable": "Mortgage_Approvals",
            "demographic_filters": ["Income > 50000"]
        }
        
        logger.info(f"Running analysis with test query: {test_query}")
        start_time = time.time()
        
        # Run the analysis worker with our test query
        result = analysis_worker(test_query)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
        
        # Check result
        if result.get("success"):
            logger.info("✅ Analysis succeeded!")
            # Log feature importance
            features = result.get("feature_importance", [])
            if features:
                logger.info("Top 3 features:")
                for i, feat in enumerate(features[:3]):
                    logger.info(f"  {i+1}. {feat['feature']}: {feat['importance']:.6f}")
        else:
            logger.error(f"❌ Analysis failed: {result.get('error')}")
            if 'traceback' in result:
                logger.error(f"Traceback: {result['traceback']}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error running analysis worker: {str(e)}")
        logger.error(f"TRACEBACK: {traceback.format_exc()}")
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

def main():
    """Main function to run the direct worker test"""
    logger.info("=== Direct Worker Process Test ===")
    
    # Run the analysis worker directly
    result = run_analysis_worker_direct()
    
    # Print summary
    logger.info("\n=== TEST RESULTS ===")
    if result.get("success", False):
        logger.info("✅ Worker function ran successfully!")
        logger.info("\nThis indicates the problem is NOT with the analysis code itself.")
        logger.info("The issue is likely with:")
        logger.info("1. The Redis/RQ worker process not running or failing to start")
        logger.info("2. Redis connection issues between the web app and worker")
        logger.info("3. Resource limitations in the production environment")
        logger.info("\nRecommended actions:")
        logger.info("1. Check that worker process is running on Render")
        logger.info("2. Verify Redis connection parameters")
        logger.info("3. Check worker process logs for errors")
    else:
        logger.info("❌ Worker function failed!")
        logger.info("\nThis indicates the problem IS with the analysis code itself.")
        logger.info("The specific error should be in the traceback above.")
        logger.info("\nRecommended actions:")
        logger.info("1. Fix the specific error shown in the traceback")
        logger.info("2. Check for memory issues during analysis")
        logger.info("3. Verify all necessary data files are present")
    
    return 0 if result.get("success", False) else 1

if __name__ == "__main__":
    sys.exit(main())
