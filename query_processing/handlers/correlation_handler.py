#!/usr/bin/env python3
"""
Correlation Query Handler

This module handles correlation-type queries.
"""

import logging
from typing import Dict, Any, List, Optional

from .base_handler import QueryHandler

logger = logging.getLogger("correlation-handler")

class CorrelationQueryHandler(QueryHandler):
    """
    Handler for correlation queries, e.g., "What factors influence mortgage approvals?"
    """
    
    def prepare_analysis(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the analysis parameters for correlation analysis
        
        Args:
            query_params: Parameters extracted from the natural language query
            
        Returns:
            Dict containing structured parameters for analysis
        """
        # Validate parameters
        validated = self.validate_params(query_params)
        
        # Prepare analysis parameters
        analysis_params = {
            "analysis_type": "correlation",
            "target_variable": validated["target_variable"],
            "demographic_filters": validated.get("demographic_filters", []),
            "top_n_factors": 10  # Default to top 10 factors
        }
        
        # Extract specific correlation relationship if available
        if "correlation_relationship" in validated:
            relationship = validated["correlation_relationship"]
            if "variable1" in relationship and "variable2" in relationship:
                # Special case: user asked for relationship between two specific variables
                analysis_params["specific_variables"] = [relationship["variable1"], relationship["variable2"]]
                analysis_params["relationship_type"] = relationship.get("direction", "correlation")
            elif "target" in relationship and relationship.get("type") == "multi_factor":
                # Special case: user asked for all factors affecting a variable
                analysis_params["target_variable"] = relationship["target"]
                analysis_params["top_n_factors"] = 15  # Show more factors in this case
        
        return analysis_params
        
    def format_results(self, analysis_results: Dict[str, Any], query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format correlation analysis results for presentation
        
        Args:
            analysis_results: Raw results from the analysis
            query_params: Original query parameters
            
        Returns:
            Dict containing formatted results for presentation
        """
        # Extract key elements from results
        target = query_params.get("target_variable", "Mortgage_Approvals")
        shap_values = analysis_results.get("shap_values", [])
        
        # Format the results for correlation query type
        formatted_results = {
            "query_type": "correlation",
            "target_variable": target,
            "title": f"Factors influencing {target.replace('_', ' ')}",
            "description": f"The following factors have the strongest influence on {target.replace('_', ' ')}",
            "factors": []
        }
        
        # Add factors with their influence scores
        for feature, importance in shap_values:
            # Convert feature name to more readable format
            display_name = feature.replace("_", " ").title()
            
            # Determine the direction of influence
            direction = "positive" if importance > 0 else "negative"
            
            formatted_results["factors"].append({
                "name": feature,
                "display_name": display_name,
                "importance": abs(importance),
                "direction": direction
            })
        
        # Add natural language insights
        formatted_results["insights"] = self._generate_insights(formatted_results["factors"], target)
        
        return formatted_results
    
    def _generate_insights(self, factors: List[Dict[str, Any]], target: str) -> List[str]:
        """Generate natural language insights about the correlations"""
        insights = []
        
        if not factors:
            return ["No significant correlations were found."]
        
        # Generate insight about the top factor
        top_factor = factors[0]
        target_display = target.replace("_", " ")
        direction_word = "increases" if top_factor["direction"] == "positive" else "decreases"
        
        insights.append(
            f"{top_factor['display_name']} has the strongest relationship with {target_display}. "
            f"As {top_factor['display_name']} increases, {target_display} typically {direction_word}."
        )
        
        # Generate insight about groups of factors
        positive_factors = [f["display_name"] for f in factors[:5] if f["direction"] == "positive"]
        negative_factors = [f["display_name"] for f in factors[:5] if f["direction"] == "negative"]
        
        if positive_factors:
            factors_text = ", ".join(positive_factors[:-1]) + (" and " + positive_factors[-1] if len(positive_factors) > 1 else positive_factors[0])
            insights.append(f"{factors_text} {'all have' if len(positive_factors) > 1 else 'has'} a positive relationship with {target_display}.")
            
        if negative_factors:
            factors_text = ", ".join(negative_factors[:-1]) + (" and " + negative_factors[-1] if len(negative_factors) > 1 else negative_factors[0])
            insights.append(f"{factors_text} {'all have' if len(negative_factors) > 1 else 'has'} a negative relationship with {target_display}.")
        
        return insights
    
    def validate_params(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters for correlation analysis
        
        Args:
            query_params: Parameters extracted from the natural language query
            
        Returns:
            Dict containing validated parameters
        """
        # Call parent validation
        validated = super().validate_params(query_params)
        
        # Correlation-specific validation
        if "limit" in validated and isinstance(validated["limit"], int):
            # If user specified a limit for factors, use it
            validated["top_n_factors"] = validated["limit"]
        else:
            validated["top_n_factors"] = 10  # Default top 10 factors
            
        return validated
