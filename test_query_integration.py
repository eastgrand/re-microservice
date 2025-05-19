#!/usr/bin/env python3
"""
Integration Test Script for SHAP Microservice Query Processing

This script performs integration testing of the query processing system
with the Flask application. It tests:

1. Direct function calls to the query processing module
2. Integration with the Flask application via internal API calls
"""

import json
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("integration-test")

# Import query processing modules
from query_processing import QueryType
from query_processing.classifier import QueryClassifier, process_query
from query_processing.integration import process_natural_language_query, detect_query_type

# Test queries
TEST_QUERIES = [
    "What factors influence mortgage approvals?",
    "Show me the top 10 areas with highest approval rates",
    "Compare mortgage approvals between urban and rural areas",
    "How have approval rates changed over the past 5 years?",
    "Show me a map of mortgage approvals across Canada",
    "What factors drive the difference between high and low income areas in terms of approvals?"
]

def test_query_classifier():
    """Test the QueryClassifier directly"""
    logger.info("Testing QueryClassifier...")
    
    classifier = QueryClassifier()
    
    results = []
    for query in TEST_QUERIES:
        query_type, confidence = classifier.classify(query)
        params = classifier.extract_parameters(query, query_type)
        
        results.append({
            "query": query,
            "type": query_type.value,
            "confidence": confidence,
            "parameters": params
        })
    
    logger.info("QueryClassifier test results:")
    for i, result in enumerate(results):
        logger.info(f"{i+1}. '{result['query']}' => {result['type']} (confidence: {result['confidence']:.2f})")
        logger.info(f"   Parameters: {json.dumps(result['parameters'])}\n")
    
    return results

def test_process_query():
    """Test the process_query function"""
    logger.info("Testing process_query...")
    
    results = []
    for query in TEST_QUERIES:
        result = process_query(query)
        results.append(result)
    
    logger.info("process_query test results:")
    for i, result in enumerate(results):
        logger.info(f"{i+1}. '{result['query_text']}' => {result['analysis_type']}")
        # Remove query_text, analysis_type and confidence from the printed output
        params = {k:v for k,v in result.items() if k not in ['query_text', 'analysis_type', 'confidence']}
        logger.info(f"   Parameters: {json.dumps(params)}\n")
    
    return results

def test_integration_api():
    """Test the integration API functions"""
    logger.info("Testing integration API...")
    
    results = []
    for query in TEST_QUERIES:
        # Test query type detection
        test_request = {"query": query}
        is_nl, nl_query = detect_query_type(test_request)
        
        # Process natural language query
        if is_nl and nl_query:
            analysis_params = process_natural_language_query(nl_query)
            
            results.append({
                "query": query,
                "is_natural_language": is_nl,
                "processed_params": analysis_params
            })
    
    logger.info("Integration API test results:")
    for i, result in enumerate(results):
        logger.info(f"{i+1}. '{result['query']}' => Natural language: {result['is_natural_language']}")
        logger.info(f"   Processed params: {json.dumps(result['processed_params'])}\n")
    
    return results

def validate_results(results):
    """Validate test results"""
    logger.info("Validating results...")
    
    # Check if we have the expected number of results
    if len(results) != len(TEST_QUERIES):
        logger.warning(f"Expected {len(TEST_QUERIES)} results, got {len(results)}")
    
    # Check for correct query type mapping
    expected_types = [
        "correlation",  # What factors influence
        "ranking",      # Show me the top 10
        "comparison",   # Compare
        "trend",        # How have changed over time
        "geographic",   # Show me a map
        "mixed"         # What factors drive the difference (mixed correlation/comparison)
    ]
    
    for i, (result, expected_type) in enumerate(zip(results, expected_types)):
        if result.get("analysis_type") != expected_type and result.get("type") != expected_type:
            actual_type = result.get("analysis_type") or result.get("type") or "unknown"
            logger.warning(f"Query {i+1} expected type '{expected_type}', got '{actual_type}'")
    
    logger.info("Validation complete")

def run_all_tests():
    """Run all integration tests"""
    logger.info("Running all integration tests...")
    
    # Test direct classifier
    classifier_results = test_query_classifier()
    
    # Test process_query function
    process_results = test_process_query()
    
    # Test integration API
    integration_results = test_integration_api()
    
    # Validate results
    validate_results(process_results)
    
    logger.info("All tests completed!")
    
if __name__ == "__main__":
    run_all_tests()
