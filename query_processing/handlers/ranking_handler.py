#!/usr/bin/env python3
"""
Ranking Query Handler

This module handles ranking-type queries.
"""

import logging
from typing import Dict, Any, List, Optional

from .base_handler import QueryHandler

logger = logging.getLogger("ranking-handler")

class RankingQueryHandler(QueryHandler):
    """
    Handler for ranking queries, e.g., "Which areas have the highest approval rates?"
    """
    
    def prepare_analysis(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the analysis parameters for ranking analysis
        
        Args:
            query_params: Parameters extracted from the natural language query
            
        Returns:
            Dict containing structured parameters for analysis
        """
        # Validate parameters
        validated = self.validate_params(query_params)
        
        # Prepare analysis parameters
        analysis_params = {
            "analysis_type": "ranking",
            "target_variable": validated["target_variable"],
            "demographic_filters": validated.get("demographic_filters", []),
            "limit": validated.get("limit", 10)  # Default to top 10
        }
        
        return analysis_params
        
    def format_results(self, analysis_results: Dict[str, Any], query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format ranking analysis results for presentation
        
        Args:
            analysis_results: Raw results from the analysis
            query_params: Original query parameters
            
        Returns:
            Dict containing formatted results for presentation
        """
        # Extract key elements from results
        target = query_params.get("target_variable", "Mortgage_Approvals")
        ranked_data = analysis_results.get("ranked_data", [])
        limit = query_params.get("limit", 10)
        
        # Format the results for ranking query type
        formatted_results = {
            "query_type": "ranking",
            "target_variable": target,
            "title": f"Areas with highest {target.replace('_', ' ')}",
            "description": f"Top {limit} areas ranked by {target.replace('_', ' ')}",
            "rankings": []
        }
        
        # Add ranked areas
        for i, area_data in enumerate(ranked_data[:limit]):
            area_name = area_data.get("name", f"Area {i+1}")
            value = area_data.get("value", 0)
            
            formatted_results["rankings"].append({
                "rank": i + 1,
                "name": area_name,
                "value": value,
                "additional_metrics": self._extract_additional_metrics(area_data)
            })
        
        # Add natural language insights
        formatted_results["insights"] = self._generate_insights(formatted_results["rankings"], target)
        
        return formatted_results
    
    def _extract_additional_metrics(self, area_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract additional metrics from area data"""
        # Extract relevant metrics that might be useful for context
        metrics = {}
        
        for key, value in area_data.items():
            # Skip the main ranking value and name
            if key in ["name", "value"]:
                continue
            
            # Include relevant metrics
            if key in ["population", "median_income", "homeownership_rate", "rental_rate", 
                      "unemployment_rate", "median_property_value"]:
                # Convert to display format
                display_key = key.replace("_", " ").title()
                metrics[display_key] = value
        
        return metrics
    
    def _generate_insights(self, rankings: List[Dict[str, Any]], target: str) -> List[str]:
        """Generate natural language insights about the rankings"""
        insights = []
        
        if not rankings:
            return ["No ranking data available."]
        
        # Generate insight about the top ranked area
        top_area = rankings[0]
        target_display = target.replace("_", " ")
        
        insights.append(
            f"{top_area['name']} ranks highest in {target_display} with a value of {top_area['value']}."
        )
        
        # Generate insight about the range of values
        if len(rankings) > 1:
            top_value = rankings[0]["value"]
            bottom_value = rankings[-1]["value"]
            
            # Calculate percentage difference
            diff_percentage = ((top_value - bottom_value) / top_value) * 100
            
            insights.append(
                f"There is a {diff_percentage:.1f}% difference between the highest and lowest ranked areas in this list."
            )
        
        # Generate insights about common characteristics if available
        metrics_to_check = ["Median Income", "Population", "Homeownership Rate"]
        for metric in metrics_to_check:
            # Check if at least half the areas have this metric
            areas_with_metric = [r for r in rankings if metric in r["additional_metrics"]]
            
            if len(areas_with_metric) >= len(rankings) / 2:
                # Calculate average of this metric
                avg_value = sum(a["additional_metrics"][metric] for a in areas_with_metric) / len(areas_with_metric)
                
                insights.append(
                    f"The top areas have an average {metric.lower()} of {avg_value:.1f}."
                )
                break
        
        return insights
    
    def validate_params(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters for ranking analysis
        
        Args:
            query_params: Parameters extracted from the natural language query
            
        Returns:
            Dict containing validated parameters
        """
        # Call parent validation
        validated = super().validate_params(query_params)
        
        # Ranking-specific validation
        if "limit" not in validated or not isinstance(validated["limit"], int):
            validated["limit"] = 10
        elif validated["limit"] > 50:
            # Cap the limit at a reasonable number
            validated["limit"] = 50
            
        return validated
