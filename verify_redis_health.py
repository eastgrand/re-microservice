#!/usr/bin/env python3
"""
Verify Redis Health Check Endpoint

This script tests if the Redis health check endpoint is properly registered
and functioning in the SHAP microservice.
"""

import os
import sys
import json
import http.client
import urllib.parse
import time

def test_redis_health_endpoint(base_url, api_key=None):
    """Test the Redis health check endpoint"""
    # Parse the URL
    if base_url.startswith('http://') or base_url.startswith('https://'):
        parsed_url = urllib.parse.urlparse(base_url)
        protocol = parsed_url.scheme
        host = parsed_url.netloc
        base_path = parsed_url.path.rstrip('/')
    else:
        # Assume http:// if no protocol specified
        protocol = 'http'
        if ':' in base_url:
            host = base_url  # Includes port
        else:
            host = f"{base_url}:5000"  # Default Flask port
        base_path = ''

    # Endpoints to test
    endpoints = [
        '/admin/redis_ping',  # Redis health check endpoint
        '/health',            # General health endpoint
        '/ping'               # Basic ping endpoint
    ]
    
    results = {}
    
    print(f"Testing Redis health endpoints at {protocol}://{host}")
    
    # Connect to the server
    if protocol == 'https':
        conn = http.client.HTTPSConnection(host, timeout=10)
    else:
        conn = http.client.HTTPConnection(host, timeout=10)

    # Test each endpoint
    for endpoint in endpoints:
        full_path = f"{base_path}{endpoint}"
        print(f"\nTesting endpoint: {full_path}")
        
        headers = {}
        if api_key:
            headers['x-api-key'] = api_key
            
        start_time = time.time()
        try:
            conn.request('GET', full_path, headers=headers)
            response = conn.getresponse()
            status = response.status
            reason = response.reason
            
            # Try to decode response as JSON
            try:
                response_data = json.loads(response.read().decode('utf-8'))
                pretty_response = json.dumps(response_data, indent=2)
            except:
                response_data = {}
                pretty_response = "[Non-JSON response]"
                
            elapsed = time.time() - start_time
            
            results[endpoint] = {
                'status': status,
                'reason': reason,
                'response': response_data,
                'time_ms': round(elapsed * 1000, 2)
            }
            
            print(f"Status: {status} {reason}")
            print(f"Time: {round(elapsed * 1000, 2)}ms")
            print(f"Response: {pretty_response}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            results[endpoint] = {
                'error': str(e)
            }
    
    conn.close()
    
    # Check if Redis health is properly reported
    redis_health = False
    if '/admin/redis_ping' in results and results['/admin/redis_ping'].get('status') == 200:
        response_data = results['/admin/redis_ping'].get('response', {})
        if response_data.get('success') == True and response_data.get('ping') == True:
            redis_health = True
            print("\n✅ Redis health check endpoint is working properly!")
        else:
            print("\n⚠️ Redis health check endpoint responded but reported an issue.")
    else:
        print("\n❌ Redis health check endpoint is not responding properly.")

    if '/health' in results and results['/health'].get('status') == 200:
        response_data = results['/health'].get('response', {})
        if response_data.get('redis_connected') == True:
            print("✅ Health endpoint reports Redis is connected.")
        else:
            print("⚠️ Health endpoint reports Redis is NOT connected.")
    
    return redis_health, results

if __name__ == "__main__":
    # Get URL from command line or environment
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = os.environ.get('SHAP_SERVICE_URL', 'http://localhost:5000')
        
    # Get API key from environment
    api_key = os.environ.get('API_KEY', 'HFqkccbN3LV5CaB')  # Default key if not specified
    
    print(f"Testing Redis health for service at: {base_url}")
    redis_health, results = test_redis_health_endpoint(base_url, api_key)
    
    # Exit with appropriate status code
    sys.exit(0 if redis_health else 1)
