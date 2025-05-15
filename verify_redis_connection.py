#!/usr/bin/env python3
"""
SHAP Microservice Redis Connection Test

This script tests the Redis connection in the SHAP microservice
to verify that the connection improvements are working.
"""

import os
import sys
import time
import json
import http.client
import urllib.parse
from urllib.parse import urlparse

# Helper function to make HTTP requests without external dependencies
def make_http_request(url, method='GET', headers=None, body=None, timeout=10):
    """Make an HTTP request using http.client"""
    parsed_url = urlparse(url)
    
    # Determine if HTTP or HTTPS
    if parsed_url.scheme == 'https':
        conn = http.client.HTTPSConnection(parsed_url.netloc, timeout=timeout)
    else:
        conn = http.client.HTTPConnection(parsed_url.netloc, timeout=timeout)
    
    path = parsed_url.path
    if parsed_url.query:
        path += '?' + parsed_url.query
        
    headers = headers or {}
    
    try:
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        
        # Read response data
        response_data = response.read()
        status = response.status
        
        # Parse JSON if possible
        content = None
        if response_data:
            try:
                content = json.loads(response_data.decode('utf-8'))
            except json.JSONDecodeError:
                content = response_data.decode('utf-8')
        
        return {
            'status': status,
            'content': content,
            'ok': 200 <= status < 300
        }
    except Exception as e:
        print(f"Request error: {str(e)}")
        return {
            'status': 0,
            'content': f"Error: {str(e)}",
            'ok': False,
            'error': str(e)
        }
    finally:
        conn.close()

def test_redis_ping():
    """Test the Redis ping endpoint"""
    base_url = os.environ.get('SHAP_SERVICE_URL', 'http://localhost:8081')
    api_key = os.environ.get('API_KEY', 'HFqkccbN3LV5CaB')
    
    # Strip trailing slash if present
    base_url = base_url.rstrip('/')
    
    ping_url = f"{base_url}/admin/redis_ping"
    
    print(f"Testing Redis ping at: {ping_url}")
    
    response = make_http_request(
        ping_url, 
        headers={'x-api-key': api_key}
    )
    
    print(f"Status code: {response['status']}")
    
    if response['ok']:
        print(f"Redis ping successful: {json.dumps(response['content'], indent=2)}")
        return True
    else:
        print(f"Redis ping failed: {response['content']}")
        return False

def test_health_endpoint():
    """Test the health endpoint"""
    base_url = os.environ.get('SHAP_SERVICE_URL', 'http://localhost:8081')
    api_key = os.environ.get('API_KEY', 'HFqkccbN3LV5CaB')
    
    # Strip trailing slash if present
    base_url = base_url.rstrip('/')
    
    health_url = f"{base_url}/health"
    
    print(f"Testing health endpoint at: {health_url}")
    
    response = make_http_request(
        health_url, 
        headers={'x-api-key': api_key}
    )
    
    print(f"Status code: {response['status']}")
    
    if response['ok']:
        result = response['content']
        print(f"Health check successful: {json.dumps(result, indent=2)}")
        return result.get('redis_connected', False)
    else:
        print(f"Health check failed: {response['content']}")
        return False

def run_quick_job():
    """Submit a small job to test the Redis queue"""
    base_url = os.environ.get('SHAP_SERVICE_URL', 'http://localhost:8081')
    api_key = os.environ.get('API_KEY', 'HFqkccbN3LV5CaB')
    
    # Strip trailing slash if present
    base_url = base_url.rstrip('/')
    
    analyze_url = f"{base_url}/analyze"
    
    print(f"Submitting test job to: {analyze_url}")
    
    # Simple test data
    test_data = {
        "analysis_type": "correlation",
        "target_variable": "Mortgage_Approvals",
        "data": [
            {"Mortgage_Approvals": 1, "Age": 30, "Income": 50000},
            {"Mortgage_Approvals": 0, "Age": 25, "Income": 30000}
        ]
    }
    
    # Convert to JSON string for the request
    json_data = json.dumps(test_data)
    
    response = make_http_request(
        analyze_url,
        method='POST',
        headers={
            'Content-Type': 'application/json',
            'x-api-key': api_key
        },
        body=json_data,
        timeout=20
    )
    
    print(f"Status code: {response['status']}")
    
    if response['ok']:
        result = response['content']
        job_id = result.get('job_id')
        print(f"Job submitted successfully: {job_id}")
        
        # Check job status
        if job_id:
            job_status_url = f"{base_url}/job_status/{job_id}"
            
            print(f"Checking job status at: {job_status_url}")
            
            # Poll for status 5 times
            for i in range(5):
                time.sleep(2)  # Wait 2 seconds between polls
                
                status_response = make_http_request(
                    job_status_url,
                    headers={'x-api-key': api_key},
                    timeout=10
                )
                
                if status_response['ok']:
                    status_data = status_response['content']
                    status = status_data.get('status')
                    print(f"Poll {i+1}: Job status: {status}")
                    
                    if status == 'finished':
                        print("✅ Job completed successfully!")
                        return True
                else:
                    print(f"Failed to check job status: {status_response['status']}")
            
            print("⚠️ Job did not complete within the timeout period")
            return False
        else:
            print("❌ No job ID returned")
            return False
    else:
        print(f"Failed to submit job: {response['content']}")
        return False

def main():
    """Run all the tests"""
    print("===== SHAP Microservice Redis Connection Test =====")
    print("Testing the Redis connection fixes...")
    print()
    
    # Test 1: Redis Ping
    print("Test 1: Redis Ping")
    print("-----------------")
    redis_ping_result = test_redis_ping()
    print()
    
    # Test 2: Health Endpoint
    print("Test 2: Health Endpoint")
    print("----------------------")
    health_result = test_health_endpoint()
    print()
    
    # Test 3: Job Submission
    print("Test 3: Job Submission")
    print("--------------------")
    job_result = run_quick_job()
    print()
    
    # Summary
    print("===== Test Results =====")
    print(f"Redis Ping: {'✅ PASS' if redis_ping_result else '❌ FAIL'}")
    print(f"Health Check: {'✅ PASS' if health_result else '❌ FAIL'}")
    print(f"Job Submission: {'✅ PASS' if job_result else '⚠️ INCONCLUSIVE'}")
    
    # Overall result
    if redis_ping_result and health_result:
        print("\n✅ Redis connection is working properly!")
        return 0
    else:
        print("\n❌ Redis connection issues still present")
        return 1

if __name__ == "__main__":
    sys.exit(main())
