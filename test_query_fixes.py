#!/usr/bin/env python3
"""
Test script for the enhanced query classifier.
This script tests the previously unclassified queries.
"""

import sys
from query_processing import QueryType
from query_processing.classifier import QueryClassifier, process_query

def test_previously_unclassified_queries():
    """Test the queries that were previously unclassified"""
    
    # Define ANSI color codes for terminal output
    GREEN = '\033[92m' if sys.stdout.isatty() else ''
    YELLOW = '\033[93m' if sys.stdout.isatty() else ''
    RED = '\033[91m' if sys.stdout.isatty() else ''
    ENDC = '\033[0m' if sys.stdout.isatty() else ''
    
    # Previously unclassified queries
    test_cases = [
        "Are mortgage approvals higher in urban or rural regions?",
        "What areas have high approvals but low income?",
        "Which demographic factors predict high approval rates?",
        "Show the trend in approval rates in high income areas",
        "Historical pattern of approvals in urban areas"
    ]
    
    classifier = QueryClassifier()
    
    print("\n=== Testing Previously Unclassified Queries ===\n")
    
    for query in test_cases:
        query_type, confidence = classifier.classify(query)
        params = classifier.extract_parameters(query, query_type)
        
        # Determine color based on confidence
        if confidence > 0.3:
            confidence_color = GREEN
        elif confidence > 0.1:
            confidence_color = YELLOW
        else:
            confidence_color = RED
            
        print(f"Query: \"{query}\"")
        print(f"  Type: {query_type.value}")
        print(f"  Confidence: {confidence_color}{confidence:.2f}{ENDC}")
        
        # Check for time-series error
        if "error" in params:
            print(f"  {YELLOW}Error: {params['error']['code']}{ENDC}")
            print(f"  {YELLOW}Message: {params['error']['message']}{ENDC}")
        elif query_type == QueryType.TREND:
            print(f"  {RED}Warning: Trend query detected but no error message was added{ENDC}")
            
        # Print useful parameters that were extracted
        if "target_variable" in params:
            print(f"  Target: {params['target_variable']}")
        if "demographic_filters" in params:
            print(f"  Filters: {params['demographic_filters']}")
        if "comparison_groups" in params:
            print(f"  Groups: {[g['name'] for g in params['comparison_groups']]}")
        
        print()
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    test_previously_unclassified_queries()
