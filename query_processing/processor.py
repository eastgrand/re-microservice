#!/usr/bin/env python3
"""
Query Processor Module

This module processes natural language queries into structured analysis parameters
and provides integration with the Flask application.
"""

import logging
from typing import Dict, Any, Tuple, Optional, Union

from query_processing import QueryType
from query_processing.classifier import QueryClassifier
from query_processing.handlers.correlation_handler import CorrelationQueryHandler
from query_processing.handlers.ranking_handler import RankingQueryHandler
from query_processing.handlers.comparison_handler import ComparisonQueryHandler
from query_processing.handlers.trend_handler import TrendQueryHandler
from query_processing.handlers.geographic_handler import GeographicQueryHandler
from query_processing.handlers.mixed_handler import MixedQueryHandler

logger = logging.getLogger("query-processor")

# Handler mapping
HANDLERS = {
    QueryType.CORRELATION: CorrelationQueryHandler(),
    QueryType.RANKING: RankingQueryHandler(),
    QueryType.COMPARISON: ComparisonQueryHandler(),
    QueryType.TREND: TrendQueryHandler(),
    QueryType.GEOGRAPHIC: GeographicQueryHandler(),
    QueryType.MIXED: MixedQueryHandler()
}

def process_query(query_text: str) -> Dict[str, Any]:
    """
    Process a natural language query into structured parameters
    
    Args:
        query_text: The natural language query
        
    Returns:
        Dict containing query_type and extracted parameters
    """
    # Classify the query
    classifier = QueryClassifier()
    query_type, confidence = classifier.classify(query_text)
    
    # Extract parameters based on query type
    params = classifier.extract_parameters(query_text, query_type)
    
    # Combine results
    result = {
        "query_text": query_text,
        "analysis_type": query_type.value,
        "confidence": confidence,
        **params
    }
    
    # Set default target if none extracted
    if "target_variable" not in result:
        result["target_variable"] = "Mortgage_Approvals"
    
    return result

def integrate_with_analysis(query_text: str) -> Dict[str, Any]:
    """
    Integrate the query processing with the SHAP analysis workflow
    
    Args:
        query_text: The natural language query
        
    Returns:
        Dict containing structured parameters for the analysis worker
    """
    # Process the query
    query_params = process_query(query_text)
    
    # Get the appropriate handler based on the query type
    query_type_str = query_params.get("analysis_type", "correlation")
    
    # Convert string to QueryType enum
    query_type = None
    for qt in QueryType:
        if qt.value == query_type_str:
            query_type = qt
            break
    
    if not query_type:
        query_type = QueryType.CORRELATION
    
    # Get the handler (or use default)
    handler = HANDLERS.get(query_type, CorrelationQueryHandler())
    
    # Prepare the analysis parameters
    try:
        analysis_params = handler.prepare_analysis(query_params)
        
        # Add the original query text for reference
        analysis_params["original_query"] = query_text
        
        return analysis_params
        
    except Exception as e:
        logger.error(f"Error preparing analysis parameters: {str(e)}")
        
        # Fall back to basic parameters
        return {
            "analysis_type": "correlation",
            "target_variable": query_params.get("target_variable", "Mortgage_Approvals"),
            "demographic_filters": query_params.get("demographic_filters", []),
            "original_query": query_text
        }

def format_analysis_results(analysis_results: Dict[str, Any], query_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format analysis results based on query type
    
    Args:
        analysis_results: Raw results from the analysis
        query_params: Original query parameters
        
    Returns:
        Dict containing formatted results for presentation
    """
    # Get the appropriate handler based on the query type
    query_type_str = query_params.get("analysis_type", "correlation")
    
    # Convert string to QueryType enum
    query_type = None
    for qt in QueryType:
        if qt.value == query_type_str:
            query_type = qt
            break
    
    if not query_type:
        query_type = QueryType.CORRELATION
    
    # Get the handler (or use default)
    handler = HANDLERS.get(query_type, CorrelationQueryHandler())
    
    # Format the results
    try:
        formatted_results = handler.format_results(analysis_results, query_params)
        return formatted_results
    except Exception as e:
        logger.error(f"Error formatting analysis results: {str(e)}")
        
        # Return the raw results as a fallback
        return analysis_results
