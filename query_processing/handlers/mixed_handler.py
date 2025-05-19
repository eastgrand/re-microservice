#!/usr/bin/env python3
"""
Mixed Query Handler

This module handles mixed-type queries that combine multiple query types.
"""

import logging
from typing import Dict, Any, List, Optional

from .base_handler import QueryHandler
from .correlation_handler import CorrelationQueryHandler
from .ranking_handler import RankingQueryHandler
from .comparison_handler import ComparisonQueryHandler
from .trend_handler import TrendQueryHandler
from .geographic_handler import GeographicQueryHandler

logger = logging.getLogger("mixed-handler")

class MixedQueryHandler(QueryHandler):
    """
    Handler for mixed queries that combine multiple types, e.g.,
    "Compare the trend of approval rates between urban and rural areas since 2020"
    """
    
    def __init__(self):
        """Initialize with handlers for all query types"""
        super().__init__()
        self.handlers = {
            "correlation": CorrelationQueryHandler(),
            "ranking": RankingQueryHandler(),
            "comparison": ComparisonQueryHandler(),
            "trend": TrendQueryHandler(),
            "geographic": GeographicQueryHandler()
        }
    
    def prepare_analysis(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the analysis parameters for mixed query types
        
        Args:
            query_params: Parameters extracted from the natural language query
            
        Returns:
            Dict containing structured parameters for analysis
        """
        # Validate parameters
        validated = self.validate_params(query_params)
        
        # For mixed queries, we identify the primary and secondary query types
        primary_type = self._identify_primary_type(validated)
        
        # Get the handler for the primary type
        handler = self.handlers.get(primary_type, self.handlers["correlation"])
        
        # Prepare base parameters using the primary handler
        analysis_params = handler.prepare_analysis(validated)
        
        # Mark as mixed query type
        analysis_params["analysis_type"] = "mixed"
        analysis_params["primary_type"] = primary_type
        
        # Include any parameters from other detected types
        self._enrich_with_secondary_parameters(analysis_params, validated)
        
        return analysis_params
        
    def format_results(self, analysis_results: Dict[str, Any], query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format mixed query analysis results for presentation
        
        Args:
            analysis_results: Raw results from the analysis
            query_params: Original query parameters
            
        Returns:
            Dict containing formatted results for presentation
        """
        # This is a placeholder implementation
        # Will be implemented in a future update
        return {
            "query_type": "mixed",
            "message": "Mixed query handler is not yet fully implemented"
        }
    
    def _identify_primary_type(self, query_params: Dict[str, Any]) -> str:
        """Identify the primary query type from the mixed query"""
        # Default to correlation if no hints available
        if "mixed_primary_type" in query_params:
            return query_params["mixed_primary_type"]
            
        # Look for strong indicators of different query types
        if "comparison_groups" in query_params:
            return "comparison"
        elif "time_period" in query_params:
            return "trend"
        elif "regions" in query_params or "map_type" in query_params:
            return "geographic"
        elif "limit" in query_params:
            return "ranking"
        
        # Default to correlation
        return "correlation"
    
    def _enrich_with_secondary_parameters(self, analysis_params: Dict[str, Any], query_params: Dict[str, Any]) -> None:
        """Add parameters from secondary query types to enrich the analysis"""
        # Include time period for trend analysis
        if "time_period" in query_params:
            analysis_params["time_period"] = query_params["time_period"]
            
        # Include comparison groups
        if "comparison_groups" in query_params:
            analysis_params["comparison_groups"] = query_params["comparison_groups"]
            
        # Include geographic info
        if "regions" in query_params:
            analysis_params["regions"] = query_params["regions"]
        
        if "map_type" in query_params:
            analysis_params["map_type"] = query_params["map_type"]
    
    def validate_params(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters for mixed query analysis
        
        Args:
            query_params: Parameters extracted from the natural language query
            
        Returns:
            Dict containing validated parameters
        """
        # Call parent validation
        validated = super().validate_params(query_params)
        
        # Mixed queries can have parameters from multiple query types
        # We don't need additional validation here since each specific 
        # handler will validate its own parameters
            
        return validated
