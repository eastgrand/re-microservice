#!/usr/bin/env python3
import requests
import json
import sys
from urllib.parse import urljoin

def test_cors(base_url, api_key):
    """Test CORS preflight and actual request to /analyze endpoint."""
    
    # Test data
    test_data = {
        "analysis_type": "correlation",
        "target_variable": "Mortgage_Approvals",
        "demographic_filters": ["high income"]
    }
    
    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': api_key,
        'Origin': 'http://localhost:3000'  # Simulate frontend origin
    }
    
    # 1. Test OPTIONS preflight request
    print("\n=== Testing OPTIONS Preflight Request ===")
    try:
        options_response = requests.options(
            urljoin(base_url, '/analyze'),
            headers=headers
        )
        print(f"Status Code: {options_response.status_code}")
        print("Response Headers:")
        for key, value in options_response.headers.items():
            print(f"  {key}: {value}")
        print("\nResponse Body:")
        print(options_response.text)
    except Exception as e:
        print(f"Error during OPTIONS request: {str(e)}")
    
    # 2. Test actual POST request
    print("\n=== Testing POST Request ===")
    try:
        post_response = requests.post(
            urljoin(base_url, '/analyze'),
            headers=headers,
            json=test_data
        )
        print(f"Status Code: {post_response.status_code}")
        print("Response Headers:")
        for key, value in post_response.headers.items():
            print(f"  {key}: {value}")
        print("\nResponse Body:")
        print(json.dumps(post_response.json(), indent=2))
    except Exception as e:
        print(f"Error during POST request: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_cors.py <base_url> <api_key>")
        print("Example: python test_cors.py http://localhost:10000 your_api_key")
        sys.exit(1)
    
    base_url = sys.argv[1]
    api_key = sys.argv[2]
    
    print(f"Testing CORS for {base_url}")
    test_cors(base_url, api_key) 