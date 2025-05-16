#!/usr/bin/env python3
"""
SHAP Microservice Performance Monitor

This script helps monitor the performance of the SHAP microservice
after implementing memory optimizations. It submits test jobs and
measures processing times and resource usage.
"""

import os
import sys
import time
import logging
import argparse
import json
import statistics
from datetime import datetime, timedelta
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("performance-monitor")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Monitor SHAP microservice performance')
    parser.add_argument('--url', default='https://nesto-mortgage-analytics.onrender.com',
                        help='Base URL of the SHAP microservice')
    parser.add_argument('--api-key', default=os.environ.get('API_KEY'),
                        help='API key for authentication')
    parser.add_argument('--num-jobs', type=int, default=3,
                        help='Number of test jobs to submit')
    parser.add_argument('--rows', type=int, default=100,
                        help='Number of rows in test data')
    parser.add_argument('--columns', type=int, default=10,
                        help='Number of columns in test data')
    parser.add_argument('--timeout', type=int, default=180,
                        help='Timeout for each job in seconds')
    return parser.parse_args()

def get_memory_usage(base_url, api_key=None, timeout=10):
    """Get current memory usage from the service"""
    try:
        headers = {"X-API-KEY": api_key} if api_key else {}
        response = requests.get(f"{base_url}/admin/memory", headers=headers, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return data.get('memory_usage_mb')
        return None
    except Exception as e:
        logger.error(f"Error getting memory usage: {str(e)}")
        return None

def generate_test_data(num_rows, num_cols):
    """Generate test data for performance testing"""
    import numpy as np
    
    # Create synthetic data
    data = {}
    for i in range(num_cols):
        col_name = f"feature_{i+1}"
        data[col_name] = np.random.random(num_rows).tolist()
    
    # Add a target column
    data["target"] = np.random.random(num_rows).tolist()
    
    return data

def submit_job(base_url, api_key, test_data, timeout=180):
    """Submit a test job to the SHAP service"""
    if not api_key:
        logger.error("❌ Cannot submit test job: API key not provided")
        return None
    
    try:
        logger.info(f"Submitting test job with {len(next(iter(test_data.values())))} rows and {len(test_data)} columns...")
        headers = {"X-API-KEY": api_key}
        
        # Create test payload
        payload = {
            "analysis_type": "shap",
            "prediction_data": test_data,
            "target_variable": "target",
            "max_rows": len(next(iter(test_data.values())))
        }
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/v1/submit_job",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200 or response.status_code == 202:
            data = response.json()
            if data.get('success'):
                job_id = data.get('job_id')
                logger.info(f"✅ Job submitted successfully with ID: {job_id}")
                
                # Wait for job to complete
                result = wait_for_job_completion(base_url, api_key, job_id, timeout)
                if result:
                    result['submit_time'] = time.time() - start_time
                    return result
        
        logger.error(f"❌ Job submission failed with status code: {response.status_code}")
        logger.error(f"Response: {response.text}")
        return None
    except Exception as e:
        logger.error(f"❌ Error submitting job: {str(e)}")
        return None

def wait_for_job_completion(base_url, api_key, job_id, timeout=180):
    """Wait for a job to complete and measure performance metrics"""
    if not job_id or not api_key:
        logger.error("❌ Cannot check job status: job_id or API key not provided")
        return None
    
    headers = {"X-API-KEY": api_key}
    start_time = time.time()
    end_time = start_time + timeout
    poll_count = 0
    
    while time.time() < end_time:
        try:
            poll_count += 1
            response = requests.get(
                f"{base_url}/api/v1/job_status/{job_id}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('status')
                
                if status == 'completed':
                    processing_time = time.time() - start_time
                    logger.info(f"✅ Job completed in {processing_time:.2f} seconds (polling: {poll_count} times)")
                    return {
                        'job_id': job_id,
                        'processing_time': processing_time,
                        'poll_count': poll_count,
                        'status': 'completed',
                    }
                elif status == 'failed':
                    logger.error(f"❌ Job failed: {data.get('error', 'Unknown error')}")
                    return {
                        'job_id': job_id,
                        'processing_time': time.time() - start_time,
                        'poll_count': poll_count,
                        'status': 'failed',
                        'error': data.get('error', 'Unknown error')
                    }
                else:
                    logger.info(f"⏳ Job status: {status} (elapsed: {time.time() - start_time:.2f}s)")
            else:
                logger.warning(f"⚠️ Error checking job status: HTTP {response.status_code}")
                
            # Wait before polling again
            time.sleep(3)
        except Exception as e:
            logger.warning(f"⚠️ Error checking job status: {str(e)}")
            time.sleep(3)
    
    logger.error(f"❌ Job timed out after {timeout} seconds")
    return {
        'job_id': job_id,
        'processing_time': timeout,
        'poll_count': poll_count,
        'status': 'timeout'
    }

def run_performance_test(args):
    """Run a series of performance tests"""
    results = []
    test_data = generate_test_data(args.rows, args.columns)
    
    logger.info(f"Starting performance test with {args.num_jobs} jobs")
    logger.info(f"Each job uses {args.rows} rows and {args.columns} columns")
    
    # Get initial memory usage
    initial_memory = get_memory_usage(args.url, args.api_key)
    if initial_memory:
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
    
    for i in range(args.num_jobs):
        logger.info(f"Running job {i+1}/{args.num_jobs}")
        result = submit_job(args.url, args.api_key, test_data, args.timeout)
        if result:
            results.append(result)
            
            # Give the system time to recover between jobs
            time.sleep(5)
    
    # Get final memory usage
    final_memory = get_memory_usage(args.url, args.api_key)
    if final_memory:
        logger.info(f"Final memory usage: {final_memory:.2f} MB")
    
    return results, initial_memory, final_memory

def analyze_results(results, initial_memory, final_memory):
    """Analyze and display test results"""
    if not results:
        logger.error("❌ No test results to analyze")
        return
    
    processing_times = [r.get('processing_time', 0) for r in results if r.get('status') == 'completed']
    submit_times = [r.get('submit_time', 0) for r in results if r.get('status') == 'completed']
    poll_counts = [r.get('poll_count', 0) for r in results if r.get('status') == 'completed']
    
    completed_count = sum(1 for r in results if r.get('status') == 'completed')
    failed_count = sum(1 for r in results if r.get('status') == 'failed')
    timeout_count = sum(1 for r in results if r.get('status') == 'timeout')
    
    print("\n" + "="*50)
    print(" SHAP Microservice Performance Test Results ")
    print("="*50)
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total jobs: {len(results)}")
    print(f"Completed: {completed_count}")
    print(f"Failed: {failed_count}")
    print(f"Timed out: {timeout_count}")
    
    if processing_times:
        print("\nProcessing time statistics:")
        print(f"  Min: {min(processing_times):.2f} seconds")
        print(f"  Max: {max(processing_times):.2f} seconds")
        print(f"  Avg: {statistics.mean(processing_times):.2f} seconds")
        if len(processing_times) > 1:
            print(f"  Median: {statistics.median(processing_times):.2f} seconds")
            print(f"  Std Dev: {statistics.stdev(processing_times):.2f} seconds")
    
    if submit_times:
        print("\nJob submission time statistics:")
        print(f"  Min: {min(submit_times):.2f} seconds")
        print(f"  Max: {max(submit_times):.2f} seconds")
        print(f"  Avg: {statistics.mean(submit_times):.2f} seconds")
    
    if poll_counts:
        print("\nPolling count statistics:")
        print(f"  Min: {min(poll_counts)} polls")
        print(f"  Max: {max(poll_counts)} polls")
        print(f"  Avg: {statistics.mean(poll_counts):.2f} polls")
    
    if initial_memory and final_memory:
        print("\nMemory usage:")
        print(f"  Initial: {initial_memory:.2f} MB")
        print(f"  Final: {final_memory:.2f} MB")
        print(f"  Change: {final_memory - initial_memory:.2f} MB")
    
    print("\nDetailed results:")
    for i, result in enumerate(results):
        status = result.get('status', 'unknown')
        if status == 'completed':
            print(f"  Job {i+1}: ✅ Completed in {result.get('processing_time', 0):.2f}s")
        elif status == 'failed':
            print(f"  Job {i+1}: ❌ Failed - {result.get('error', 'unknown error')}")
        elif status == 'timeout':
            print(f"  Job {i+1}: ⏱ Timed out after {result.get('processing_time', 0):.2f}s")
        else:
            print(f"  Job {i+1}: ⚠️ Unknown status: {status}")
    
    print("\n" + "="*50)
    
    # Return success if most jobs completed
    return completed_count >= len(results) / 2

def main():
    """Main function"""
    args = parse_args()
    
    if not args.api_key:
        logger.error("❌ API key is required. Please provide it via --api-key parameter or API_KEY environment variable.")
        return 1
    
    logger.info(f"Starting SHAP microservice performance monitoring")
    logger.info(f"Service URL: {args.url}")
    
    try:
        results, initial_memory, final_memory = run_performance_test(args)
        success = analyze_results(results, initial_memory, final_memory)
        
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error running performance test: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
