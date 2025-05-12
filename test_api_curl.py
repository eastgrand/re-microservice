#!/usr/bin/env python3
"""
Test the Flask API directly from Python using curl via subprocess
"""
import json
import subprocess
import time
import os
import sys
import signal
from pprint import pprint

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
    """Test the health endpoint using curl"""
    print("\n----- Testing Health Endpoint -----")
    
    curl_cmd = f"curl -s -X GET '{API_URL}/health' -H 'X-API-KEY: {API_KEY}'"
    result = subprocess.run(curl_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0 and "healthy" in result.stdout:
        print("✅ Health check passed!")
        try:
            data = json.loads(result.stdout)
            print(f"SHAP Version: {data.get('shap_version', 'unknown')}")
        except json.JSONDecodeError:
            print("Could not parse JSON response")
        return True
    else:
        print("❌ Health check failed!")
        print(f"Return code: {result.returncode}")
        print(f"Response: {result.stdout}")
        print(f"Error: {result.stderr}")
        return False

def test_analysis():
    """Test the analysis endpoint using curl"""
    print("\n----- Testing Analysis Endpoint -----")
    
    payload = json.dumps({
        "analysis_type": "correlation",
        "target_variable": "Nike_Sales",
        "demographic_filters": ["Age < 40"]
    })
    
    curl_cmd = f"curl -s -X POST '{API_URL}/analyze' " \
               f"-H 'Content-Type: application/json' " \
               f"-H 'X-API-KEY: {API_KEY}' " \
               f"-d '{payload}'"
    
    result = subprocess.run(curl_cmd, shell=True, capture_output=True, text=True)
    
    try:
        response_data = json.loads(result.stdout)
        if result.returncode == 0 and response_data.get("success") == True:
            print("✅ Analysis test passed!")
            print(f"Summary: {response_data.get('summary', 'No summary available')}")
            
            # Print top 3 feature importances
            importance = response_data.get("feature_importance", [])
            if importance:
                print("\nTop 3 Features by Importance:")
                for i, feat in enumerate(importance[:3]):
                    print(f"  {i+1}. {feat['feature']}: {feat['importance']:.4f}")
            
            return True
        else:
            print("❌ Analysis test failed!")
            print(f"Return code: {result.returncode}")
            print(f"Response: {json.dumps(response_data, indent=2)}")
            return False
    except json.JSONDecodeError:
        print("❌ Analysis test failed - could not parse JSON response!")
        print(f"Response: {result.stdout}")
        print(f"Error: {result.stderr}")
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
