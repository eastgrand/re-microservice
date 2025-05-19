#!/usr/bin/env python3
"""
Query Processing Integration Module

This module provides integration between the query processing system 
and the Flask application. It handles:
1. Converting natural language queries to structured analysis parameters
2. Formatting analysis results based on the query type
3. Providing a simplified API for the Flask routes
"""

import logging
from typing import Dict, Any, Optional, Tuple

from query_processing.processor import process_query, integrate_with_analysis, format_analysis_results

logger = logging.getLogger("query-integration")

def process_natural_language_query(query_text: str) -> Dict[str, Any]:
    """
    Process a natural language query for the Flask API
    
    Args:
        query_text: The natural language query from the user
        
    Returns:
        Dict containing structured parameters for the analysis worker
    """
    try:
        # Process the query to get structured parameters
        analysis_params = integrate_with_analysis(query_text)
        
        # Log the processed query
        logger.info(f"Processed query: '{query_text}' to parameters: {analysis_params}")
        
        # Return the structured parameters for the analysis worker
        return analysis_params
    except Exception as e:
        logger.error(f"Error processing query '{query_text}': {str(e)}")
        
        # Fall back to basic parameters
        return {
            "analysis_type": "correlation",
            "target_variable": "Mortgage_Approvals",
            "original_query": query_text
        }

def format_results_for_response(analysis_results: Dict[str, Any], query_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format analysis results for the API response
    
    Args:
        analysis_results: Raw analysis results from the worker
        query_params: Original query parameters
        
    Returns:
        Dict containing formatted results for API response
    """
    try:
        # Format the results based on the query type
        formatted_results = format_analysis_results(analysis_results, query_params)
        
        # Add metadata
        formatted_results["original_query"] = query_params.get("original_query", "")
        formatted_results["success"] = True
        
        return formatted_results
    except Exception as e:
        logger.error(f"Error formatting results: {str(e)}")
        
        # Return original results with error flag
        return {
            "success": True,  # Still returning success since we have some results
            "original_query": query_params.get("original_query", ""),
            "error_in_formatting": str(e),
            "raw_results": analysis_results
        }

def detect_query_type(request_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Detect if the request contains a natural language query or structured parameters
    
    Args:
        request_data: The request data from the client
        
    Returns:
        Tuple of (is_natural_language, query_text)
        - is_natural_language: True if the request contains a natural language query
        - query_text: The natural language query text if found, None otherwise
    """
    # Check for natural language query field
    if "query" in request_data and isinstance(request_data["query"], str):
        return True, request_data["query"]
    
    # Check for question field
    if "question" in request_data and isinstance(request_data["question"], str):
        return True, request_data["question"]
        
    # Check for prompt field
    if "prompt" in request_data and isinstance(request_data["prompt"], str):
        return True, request_data["prompt"]
    
    # Check if text field exists and other structured fields don't
    if "text" in request_data and isinstance(request_data["text"], str):
        # If the request has both text and structured parameters, it's ambiguous
        if "analysis_type" in request_data or "target_variable" in request_data:
            return False, None
        return True, request_data["text"]
    
    # Not a natural language query
    return False, None
