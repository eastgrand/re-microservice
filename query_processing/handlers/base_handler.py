#!/usr/bin/env python3
"""
Base Query Handler

This module defines the base class for all query handlers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

logger = logging.getLogger("query-handler")

class QueryHandler(ABC):
    """
    Abstract base class for query handlers.
    Each query type has a specialized handler that inherits from this class.
    """
    
    def __init__(self):
        """Initialize the handler"""
        pass
        
    @abstractmethod
    def prepare_analysis(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the analysis parameters for the query type
        
        Args:
            query_params: Parameters extracted from the natural language query
            
        Returns:
            Dict containing structured parameters for analysis
        """
        pass
    
    @abstractmethod
    def format_results(self, analysis_results: Dict[str, Any], query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the analysis results based on the query type
        
        Args:
            analysis_results: Raw results from the analysis
            query_params: Original query parameters
            
        Returns:
            Dict containing formatted results for presentation
        """
        pass
        
    def validate_params(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and apply defaults to query parameters
        
        Args:
            query_params: Parameters extracted from the natural language query
            
        Returns:
            Dict containing validated parameters
        """
        # Common validation logic
        if "target_variable" not in query_params:
            query_params["target_variable"] = "Mortgage_Approvals"
            
        return query_params
