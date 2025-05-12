#!/usr/bin/env python3
"""
Test script for the Render deployment of the SHAP microservice.
This script tests the basic endpoints to verify the service is working correctly.
"""

import requests
import json
import sys
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test the SHAP microservice endpoints on Render')
    parser.add_argument('--url', default='https://xgboost-qeb6.onrender.com', 
                        help='Base URL of the deployed service')
    parser.add_argument('--api-key', default='',
                        help='API key for authenticated endpoints')
    args = parser.parse_args()
    
    base_url = args.url
    api_key = args.api_key
    
    # Headers for authenticated requests
    headers = {}
    if api_key:
        headers['X-API-KEY'] = api_key
    
    print(f"Testing endpoints at {base_url}")
    
    # Test 1: Root endpoint (no authentication required)
    print("\n1. Testing root endpoint...")
    try:
        print(f"Sending GET request to {base_url}/")
        response = requests.get(f"{base_url}/", timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error accessing root endpoint: {e}")
    
    # Test 2: Ping endpoint (no authentication required)
    print("\n2. Testing ping endpoint...")
    try:
        response = requests.get(f"{base_url}/ping", timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error accessing ping endpoint: {e}")
    
    # Only test authenticated endpoints if API key is provided
    if not api_key:
        print("\nSkipping authenticated endpoints. Provide --api-key to test them.")
        return
    
    # Test 3: Health endpoint (requires authentication)
    print("\n3. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", headers=headers, timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error accessing health endpoint: {e}")
    
    # Test 4: Metadata endpoint 
    print("\n4. Testing metadata endpoint...")
    try:
        response = requests.get(f"{base_url}/metadata", headers=headers, timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("Dataset statistics: ")
            if 'statistics' in result and 'Income' in result['statistics']:
                print(f"Income stats: {json.dumps(result['statistics']['Income'], indent=2)}")
            else:
                print("No Income statistics available")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error accessing metadata endpoint: {e}")
        
    # Test 5: Simple analysis (requires authentication)
    print("\n5. Testing analyze endpoint with minimal query...")
    try:
        data = {
            "analysis_type": "ranking",
            "target_variable": "Income",
            "demographic_filters": []
        }
        response = requests.post(
            f"{base_url}/analyze", 
            headers=headers, 
            json=data,
            timeout=60  # Longer timeout for analysis
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            # Show summary and first result only to keep output manageable
            print(f"Summary: {result.get('summary', 'No summary')}")
            print("First result:")
            if result.get('results') and len(result['results']) > 0:
                print(json.dumps(result['results'][0], indent=2))
            else:
                print("No results returned")
            
            # Show feature importance
            print("\nTop feature importance:")
            if result.get('feature_importance') and len(result['feature_importance']) > 0:
                for i, feature in enumerate(result['feature_importance'][:3]):
                    print(f"{i+1}. {feature['feature']}: {feature['importance']:.4f}")
            else:
                print("No feature importance returned")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error accessing analyze endpoint: {e}")

if __name__ == "__main__":
    main()
