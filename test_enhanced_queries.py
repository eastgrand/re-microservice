#!/usr/bin/env python3
"""
Test script for the enhanced query processing system.
This script tests the integrated query processing system with various queries.
"""

import json
import sys
import os
import time
import requests
from colorama import init, Fore, Style

# Initialize colorama
init()

# Constants
API_KEY = os.environ.get("SHAP_API_KEY", "local_test_key")
BASE_URL = os.environ.get("SHAP_API_URL", "http://localhost:5000")

def colorize(text, color):
    """Add color to text for terminal output"""
    colors = {
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'blue': Fore.BLUE,
        'purple': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'white': Fore.WHITE,
        'end': Style.RESET_ALL
    }
    
    return f"{colors.get(color, '')}{text}{colors['end']}"

def make_api_request(query, query_type=None):
    """Make API request to the SHAP microservice"""
    
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": API_KEY
    }
    
    if query_type == "nl":
        # Natural language query
        payload = {
            "query": query
        }
    else:
        # Structured query
        payload = {
            "analysis_type": "correlation",
            "target_variable": "Mortgage_Approvals",
            "demographic_filters": []
        }
        
    # Print the request
    print(f"\n{colorize('REQUEST:', 'blue')} {json.dumps(payload, indent=2)}")
    
    # Make the request
    try:
        response = requests.post(f"{BASE_URL}/analyze", headers=headers, json=payload)
        
        # Check response
        if response.status_code == 202:
            job_id = response.json().get("job_id")
            print(colorize(f"Job submitted with ID: {job_id}", "green"))
            
            # Poll for job status
            return poll_job_status(job_id)
        else:
            print(colorize(f"Error: {response.status_code} - {response.text}", "red"))
            return None
    except Exception as e:
        print(colorize(f"Error making request: {str(e)}", "red"))
        return None

def poll_job_status(job_id, max_attempts=20, delay=1):
    """Poll for job status and return results when complete"""
    
    headers = {
        "X-API-KEY": API_KEY
    }
    
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(f"{BASE_URL}/job_status/{job_id}", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                
                if status == "finished":
                    print(colorize(f"Job complete after {attempt} polls", "green"))
                    return data.get("result")
                elif status == "failed":
                    print(colorize(f"Job failed: {data.get('error')}", "red"))
                    return None
                else:
                    print(f"Job status: {status} (poll {attempt}/{max_attempts})")
            else:
                print(colorize(f"Error checking status: {response.status_code} - {response.text}", "red"))
                return None
        except Exception as e:
            print(colorize(f"Error polling job: {str(e)}", "red"))
            return None
            
        # Wait before next poll
        time.sleep(delay)
        
    print(colorize(f"Max polling attempts ({max_attempts}) reached", "yellow"))
    return None

def print_results(results):
    """Print formatted results"""
    
    if not results:
        print(colorize("No results to display", "yellow"))
        return
        
    print(f"\n{colorize('RESULTS:', 'green')}")
    
    # Check if using the new format
    if "query_type" in results and "insights" in results:
        # New format with insights
        print(colorize(f"Query Type: {results['query_type'].upper()}", "cyan"))
        print(colorize(f"Title: {results.get('title', 'No title')}", "cyan"))
        print(colorize(f"Description: {results.get('description', 'No description')}", "cyan"))
        
        if "insights" in results:
            print(colorize("\nInsights:", "purple"))
            for insight in results["insights"]:
                print(f"• {insight}")
                
        if "factors" in results:
            print(colorize("\nTop Factors:", "purple"))
            for i, factor in enumerate(results["factors"][:5]):
                direction = "↑" if factor.get("direction") == "positive" else "↓"
                print(f"{i+1}. {factor.get('display_name', factor.get('name', 'Unknown'))} {direction} "
                    f"(importance: {factor.get('importance', 0):.4f})")
                
        if "rankings" in results:
            print(colorize("\nTop Rankings:", "purple"))
            for i, rank in enumerate(results["rankings"][:5]):
                print(f"{i+1}. {rank.get('name', 'Unknown')} ({rank.get('value', 0):.2f})")
    else:
        # Standard format
        print(colorize(f"Summary: {results.get('summary', 'No summary available')}", "cyan"))
        
        if "feature_importance" in results:
            print(colorize("\nTop Factors:", "purple"))
            for i, factor in enumerate(results.get("feature_importance", [])[:5]):
                print(f"{i+1}. {factor.get('feature', 'Unknown')} (importance: {factor.get('importance', 0):.4f})")
                
def test_query_processing():
    """Test the query processing system with various queries"""
    
    # Define test queries
    test_queries = [
        # Correlation queries
        "What factors influence mortgage approvals?",
        "How does income affect mortgage approval rates?",
        "What is the relationship between unemployment and mortgage approvals?",
        
        # Ranking queries
        "Show me the top 5 areas with highest approval rates",
        "Which regions have the highest mortgage approval rates?",
        
        # Mixed queries
        "Compare mortgage approvals between urban and rural areas",
        "Show me a map of the regions with high approval rates",
        
        # Complex queries
        "How have mortgage approval rates in high income areas changed since 2020?",
        "What factors are driving the difference in approval rates between urban and rural areas?"
    ]
    
    print(colorize(f"\n=== TESTING QUERY PROCESSING SYSTEM ===\n", "blue"))
    print(colorize(f"API URL: {BASE_URL}", "blue"))
    print(colorize(f"Using API Key: {'*' * len(API_KEY)}", "blue"))
    
    # Test each query
    for i, query in enumerate(test_queries):
        print(colorize(f"\n\n=== TEST QUERY {i+1}/{len(test_queries)} ===", "yellow"))
        print(colorize(f"QUERY: {query}", "purple"))
        
        # Make the request
        results = make_api_request(query, "nl")
        
        # Print the results
        print_results(results)
        
        # Pause briefly between requests
        time.sleep(1)
    
    print(colorize(f"\n\n=== TESTING COMPLETE ===\n", "green"))

if __name__ == "__main__":
    test_query_processing()
