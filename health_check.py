#!/usr/bin/env python3
"""
Health check script for the Render deployment
This script attempts to connect to the service running on the specified PORT
and reports on its status.
"""

import requests
import os
import sys
import time

def check_health(max_attempts=5, delay=5):
    """Check if the service is healthy by making requests to the health endpoints"""
    port = os.environ.get('PORT', '5000')
    url_base = f"http://localhost:{port}"
    
    endpoints = [
        '/',
        '/ping', 
        '/health'
    ]
    
    print(f"Checking health of service on port {port}")
    
    for attempt in range(1, max_attempts + 1):
        print(f"\nAttempt {attempt}/{max_attempts}")
        
        for endpoint in endpoints:
            url = f"{url_base}{endpoint}"
            try:
                print(f"Checking {url}...")
                if endpoint == '/health':
                    headers = {'X-API-KEY': os.environ.get('API_KEY', '')}
                    response = requests.get(url, headers=headers, timeout=10)
                else:
                    response = requests.get(url, timeout=10)
                
                print(f"Status: {response.status_code}")
                if response.status_code == 200:
                    print(f"Success! Endpoint {endpoint} is responding")
                else:
                    print(f"Warning: Endpoint {endpoint} returned status {response.status_code}")
            except Exception as e:
                print(f"Error: {str(e)}")
        
        if attempt < max_attempts:
            print(f"Waiting {delay} seconds before next attempt...")
            time.sleep(delay)
    
    print("\nHealth check complete")

if __name__ == "__main__":
    check_health()
