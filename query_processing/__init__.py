"""
Query Processing Package

This package provides functionality for processing natural language queries
into structured analysis parameters for the SHAP microservice.

Components:
- classifier: Query classification and parameter extraction
- handlers: Specialized handlers for different query types
- processor: Main processing pipeline for integrating with Flask endpoints
"""

from enum import Enum

class QueryType(Enum):
    """Enumeration of supported query types"""
    CORRELATION = "correlation"  # What factors influence X?
    RANKING = "ranking"          # Which areas have the highest X?
    COMPARISON = "comparison"    # Compare X between group A and group B
    TREND = "trend"              # How has X changed over time?
    GEOGRAPHIC = "geographic"    # Show X patterns by location
    MIXED = "mixed"              # Combines multiple query types
    UNKNOWN = "unknown"          # Could not classify

# Import key components for external use
from .classifier import QueryClassifier
from .processor import process_query, integrate_with_analysis

__all__ = ['QueryClassifier', 'process_query', 'integrate_with_analysis', 'QueryType']
