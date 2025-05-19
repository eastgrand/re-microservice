#!/usr/bin/env python3
"""
Test the Flask API directly from Python
"""
import json
import subprocess
import time
import os
import sys
import signal
from pprint import pprint

# We'll use subprocess with curl instead of requests
# since requests might not be installed

# Configuration
PORT = 8081
API_URL = f"http://localhost:{PORT}"
API_KEY = "change_me_in_production"  # Should match .env

def run_api():
    """Start the Flask API as a subprocess"""
    print("Starting Flask API...")
    proc = subprocess.Popen(
        ["python", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Wait for the server to start
    time.sleep(3)
    return proc

def test_health():
    """Test the health endpoint"""
    print("\n----- Testing Health Endpoint -----")
    response = requests.get(
        f"{API_URL}/health",
        headers={"X-API-KEY": API_KEY}
    )
    
    if response.status_code == 200 and "healthy" in response.text:
        print("✅ Health check passed!")
        data = response.json()
        print(f"SHAP Version: {data.get('shap_version', 'unknown')}")
        return True
    else:
        print("❌ Health check failed!")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def test_analysis():
    """Test the analysis endpoint"""
    print("\n----- Testing Analysis Endpoint -----")
    payload = {
        "analysis_type": "correlation",
        "target_variable": "Nike_Sales",
        "demographic_filters": ["Age < 40"]
    }
    
    response = requests.post(
        f"{API_URL}/analyze",
        headers={
            "Content-Type": "application/json",
            "X-API-KEY": API_KEY
        },
        json=payload
    )
    
    if response.status_code == 200 and response.json().get("success") == True:
        print("✅ Analysis test passed!")
        data = response.json()
        print(f"Summary: {data.get('summary', 'No summary available')}")
        
        # Print top 3 feature importances
        importance = data.get("feature_importance", [])
        if importance:
            print("\nTop 3 Features by Importance:")
            for i, feat in enumerate(importance[:3]):
                print(f"  {i+1}. {feat['feature']}: {feat['importance']:.4f}")
        
        return True
    else:
        print("❌ Analysis test failed!")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return False

def main():
    # Test SHAP library directly
    print("Testing SHAP library directly...")
    shap_result = subprocess.run(
        ["python", "quick_shap_test.py"],
        capture_output=True, 
        text=True
    )
    
    print(shap_result.stdout)
    if shap_result.stderr:
        print("ERRORS:")
        print(shap_result.stderr)
    
    # Start API and run tests
    flask_process = run_api()
    
    try:
        # Run tests
        health_passed = test_health()
        if health_passed:
            analysis_passed = test_analysis()
        else:
            analysis_passed = False
            
        # Print summary
        print("\n----- Test Results Summary -----")
        print(f"SHAP Library Test: {'✅ Passed' if 'SUCCESS' in shap_result.stdout else '❌ Failed'}")
        print(f"API Health Check: {'✅ Passed' if health_passed else '❌ Failed'}")
        print(f"API Analysis Test: {'✅ Passed' if analysis_passed else '❌ Failed'}")
        
        if health_passed and analysis_passed and 'SUCCESS' in shap_result.stdout:
            print("\n✅ All tests passed! The service is ready for deployment.")
        else:
            print("\n❌ Some tests failed. Please fix the issues before deploying.")
            
    finally:
        # Kill the Flask process
        print("\nShutting down Flask API...")
        flask_process.terminate()
        flask_process.wait()
        
if __name__ == "__main__":
    main()
