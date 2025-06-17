#!/usr/bin/env python3
"""
Query Classifier Module

This module classifies natural language queries into predefined categories
to enable more dynamic processing of SHAP analysis requests.
"""

import re
import logging
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("query-classifier")

class QueryType(Enum):
    """Enumeration of supported query types"""
    CORRELATION = "correlation"  # What factors influence X?
    RANKING = "ranking"          # Which areas have the highest X?
    COMPARISON = "comparison"    # Compare X between group A and group B
    TREND = "trend"              # How has X changed over time?
    GEOGRAPHIC = "geographic"    # Show X patterns by location
    MIXED = "mixed"              # Combines multiple query types
    UNKNOWN = "unknown"          # Could not classify

class QueryClassifier:
    """
    Classifies natural language queries into structured analysis types.
    """
    
    def __init__(self):
        """Initialize the classifier with keyword patterns"""
        # Keyword patterns for each query type - significantly expanded
        self.patterns = {
            QueryType.CORRELATION: [
                r"(?:factors|variables|elements|aspects|features).*(?:affect|influence|impact|relate|correlate|drive|contribute).*",
                r"(?:affect|influence|impact|relate|correlate|drive|contribute).*(?:factors|variables|elements|aspects|features).*",
                r"(?:why|what makes|what causes|reason for|explanation for).*(?:high|low|increase|decrease).*",
                r"(?:significant|important|key|main|primary|critical).*(?:factors|variables|elements|aspects|features).*",
                r"(?:how|what).*(?:affects|influences|impacts|causes|drives).*",
                r"(?:relationship|correlation|connection|link|association).*(?:between|among|with).*",
                r"(?:what).*(?:causes|explains|predicts|leads to|results in).*",
                r"(?:effect|impact|influence).*(?:on|of|to).*",
                r"(?:why).*(?:varies|different|varies|varies across|differs).*",
                r"is there a correlation between.*",
                r"how does.*affect.*"
            ],
            QueryType.RANKING: [
                # Only match explicit numerical rankings
                r"(?:top|bottom|first|last|best|worst)\s+(\d+).*",
                r"(?:highest|lowest)\s+(\d+).*",
                r"show me.*(?:top|bottom)\s+(\d+).*",
                r"list.*(?:top|bottom)\s+(\d+).*",
                # Only match ranking/sorting when explicitly about ordering
                r"(?:rank|ranking|order|sort).*(?:by|of|in terms of).*",
                r"(?:list|show|display).*(?:order|sorted|ranked).*",
            ],
            QueryType.COMPARISON: [
                r"(?:compare|comparison|contrast|difference|versus|vs|vs\.|differences between|similarities between).*",
                r"(?:how does).*(?:compare to|differ from).*",
                r"(?:what is|find|show).*(?:difference|gap|distinction).*(?:between|among).*",
                r"(?:higher|lower|more|less|greater|smaller).*(?:than|compared to|relative to).*",
                r".*vs\.?.*",
                r".*versus.*",
                r".*difference between.*",
                r".*(?:compare|comparing).*",
                r".*(?:relative to|in relation to).*",
                r".*(?:better|worse) than.*",
                r"how do.*(?:compare|contrast|differ|stack up|measure up|match).*",
                r"(?:gap|disparity|variation|discrepancy) between.*"
            ],
            QueryType.TREND: [
                r"(?:change|trend|evolution|development|progression|pattern).*(?:over time|over years|over months|over the past|since).*",
                r"(?:increase|decrease|rise|fall|grow|shrink|improve|worsen).*(?:over time|over years|over months|over the past|since).*",
                r"(?:how has|how have).*(?:changed|evolved|developed|progressed|transformed).*(?:over time|over years|over months|over the past|since).*",
                r"(?:historical|history|past|previous).*(?:data|values|figures|statistics|numbers|performance|results).*",
                r".*(?:increasing|decreasing|growing|shrinking|rising|falling) trend.*",
                r".*(?:trajectory|direction|movement|shift).*(?:of|in|for).*",
                r".*(?:time series|historical data|past performance).*",
                r"has.*(?:increased|decreased|improved|worsened|changed).*(?:since|over|in the past|during).*",
                r".*trend(?:s)?.*(?:since|from|between).*",
                r".*pattern over time.*",
                r"track.*(?:progress|change|evolution|development).*"
            ],
            QueryType.GEOGRAPHIC: [
                r"(?:map|geographic|geographical|spatial|regional|location|area).*(?:distribution|pattern|variation|difference|concentration).*",
                r"(?:where|in what areas|in what regions|in what locations|in what places).*",
                r"(?:show|display|visualize|map).*(?:by region|by area|by location|by geography|by state|by province|by country|by city|by district).*",
                r"(?:distribution|spread|concentration).*(?:across|throughout|in|by).*(?:region|area|location|geography|state|province|country|city|district).*",
                r".*(?:map|maps|mapping|mapped).*",
                r".*(?:geographic|geographical|spatial|regional).*(?:analysis|distribution|pattern|variation).*",
                r".*(?:region|areas|locations|places).*(?:with|having|showing).*",
                r".*(?:spatially|geographically).*(?:distributed|organized|arranged|clustered).*",
                r".*(?:location|position|coordinates|geo|latitude|longitude).*",
                r"where are.*located.*",
                r".*(?:heat map|choropleth|spatial distribution).*"
            ]
        }
        
        # Common target variables in mortgage analysis for easier extraction
        self.target_variables = {
            "mortgage approvals": "Mortgage_Approvals",
            "mortgage approval": "Mortgage_Approvals",
            "approvals": "Mortgage_Approvals",
            "approval rate": "Mortgage_Approvals",
            "approval rates": "Mortgage_Approvals",
            "approved": "Mortgage_Approvals",
            "income": "Median_Income",
            "median income": "Median_Income",
            "household income": "Median_Income",
            "earnings": "Median_Income",
            "salary": "Median_Income",
            "population": "Population",
            "residents": "Population",
            "people": "Population",
            "inhabitants": "Population",
            "demographic": "Population",
            "homeownership": "Homeownership",
            "home ownership": "Homeownership",
            "owning homes": "Homeownership",
            "own their home": "Homeownership",
            "rental": "Rental_Units",
            "rent": "Rental_Units",
            "rentals": "Rental_Units",
            "tenants": "Rental_Units",
            "renters": "Rental_Units",
            "unemployment": "Unemployment_Rate",
            "unemployment rate": "Unemployment_Rate",
            "jobless": "Unemployment_Rate",
            "joblessness": "Unemployment_Rate",
            "employment": "Employment_Rate",
            "employment rate": "Employment_Rate",
            "employed": "Employment_Rate",
            "jobs": "Employment_Rate"
        }
    
    def classify(self, query_text: str) -> Tuple[QueryType, float]:
        """
        Classify a natural language query into a query type
        
        Args:
            query_text: The natural language query to classify
            
        Returns:
            Tuple of (QueryType, confidence_score)
        """
        # Normalize query text (lowercase, remove extra whitespace)
        normalized_query = " ".join(query_text.lower().split())
        
        # Initialize scores for each query type
        scores = {query_type: 0.0 for query_type in QueryType if query_type not in [QueryType.UNKNOWN, QueryType.MIXED]}
        
        # Check each query type's patterns
        for query_type, patterns in self.patterns.items():
            pattern_matches = []
            for i, pattern in enumerate(patterns):
                if re.search(pattern, normalized_query):
                    # Store the index of matched pattern for debugging
                    pattern_matches.append(i)
                    
            # Calculate score as percentage of patterns matched, with weight for multiple matches
            if patterns:
                # Base score is the percentage of unique patterns matched
                base_score = len(pattern_matches) / len(patterns)
                
                # Bonus for multiple matches (diminishing returns)
                match_bonus = min(0.3, len(pattern_matches) * 0.1)  
                
                scores[query_type] = min(1.0, base_score + match_bonus)
        
        # Check for mixed query types (multiple high scores)
        high_scores = [qt for qt, score in scores.items() if score > 0.2]
        if len(high_scores) > 1:
            # Get the two highest scoring query types
            top_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
            
            # If both have significant scores, it's a mixed query
            if top_types[0][1] > 0.2 and top_types[1][1] > 0.15:
                # Average the top two scores for confidence
                mixed_confidence = (top_types[0][1] + top_types[1][1]) / 2
                combined_types = f"{top_types[0][0].value}_{top_types[1][0].value}"
                logger.debug(f"Mixed query type detected: {combined_types} with confidence {mixed_confidence}")
                return QueryType.MIXED, mixed_confidence
        
        # Get the highest scoring query type
        if not scores:
            return QueryType.UNKNOWN, 0.0
            
        best_type = max(scores.items(), key=lambda x: x[1])
        
        # If score is too low, return UNKNOWN
        if best_type[1] < 0.1:  # Minimal threshold
            return QueryType.UNKNOWN, 0.0
            
        logger.debug(f"Classified '{normalized_query}' as {best_type[0]} with score {best_type[1]}")
        return best_type[0], best_type[1]

    def extract_parameters(self, query_text: str, query_type: QueryType) -> Dict[str, Any]:
        """
        Extract relevant parameters from the query based on its type
        
        Args:
            query_text: The natural language query
            query_type: The classified query type
            
        Returns:
            Dict of extracted parameters
        """
        # Normalize query text
        normalized_query = " ".join(query_text.lower().split())
        params = {}
        
        # Extract target variable if present
        target_variable = self._extract_target_variable(normalized_query)
        if target_variable:
            params["target_variable"] = target_variable
        
        # Extract demographic filters if present
        filters = self._extract_filters(normalized_query)
        if filters:
            params["demographic_filters"] = filters
            
        # Extract time period if present for trend analysis
        time_period = self._extract_time_period(normalized_query)
        if time_period and (query_type == QueryType.TREND or query_type == QueryType.MIXED):
            params["time_period"] = time_period
            
        # Add query-type specific extractions
        if query_type == QueryType.RANKING or query_type == QueryType.MIXED:
            limit = self._extract_limit(normalized_query)
            if limit:
                params["limit"] = limit
                
        if query_type == QueryType.COMPARISON or query_type == QueryType.MIXED:
            groups = self._extract_comparison_groups(normalized_query)
            if groups:
                params["comparison_groups"] = groups
                
        if query_type == QueryType.GEOGRAPHIC or query_type == QueryType.MIXED:
            regions = self._extract_regions(normalized_query)
            if regions:
                params["regions"] = regions
                
            # Add map_type parameter for geographic queries
            map_type = self._extract_map_type(normalized_query)
            if map_type:
                params["map_type"] = map_type
                
        if query_type == QueryType.CORRELATION or query_type == QueryType.MIXED:
            # For correlation queries, try to identify the specific relationship being asked about
            relationship = self._extract_correlation_relationship(normalized_query)
            if relationship:
                params["correlation_relationship"] = relationship
        
        return params
    
    def _extract_target_variable(self, query: str) -> Optional[str]:
        """Extract the target variable from the query"""
        # Expanded lookup in target_variables dictionary
        for term, variable in self.target_variables.items():
            if term in query:
                return variable
        
        # Default to Mortgage_Approvals if no target variable found
        return "Mortgage_Approvals"
    
    def _extract_filters(self, query: str) -> List[str]:
        """Extract demographic filters from the query"""
        filters = []
        
        # Check for explicit value comparisons - more robust patterns
        # Pattern: <feature> (greater|higher|more|larger) than <value> - with optional words between
        gt_matches = re.findall(r"(\w+)\s+(?:\w+\s+)*?(greater|higher|more|larger|above|exceeds|over)\s+(?:\w+\s+)*?than\s+([0-9,.]+)", query)
        for match in gt_matches:
            feature, _, value = match
            # Map to standardized field names
            field_mapping = self._map_field_name(feature)
            value = value.replace(",", "")
            filters.append(f"{field_mapping} > {value}")
        
        # Pattern: <feature> (less|lower|smaller) than <value> - with optional words between
        lt_matches = re.findall(r"(\w+)\s+(?:\w+\s+)*?(less|lower|smaller|below|under|beneath)\s+(?:\w+\s+)*?than\s+([0-9,.]+)", query)
        for match in lt_matches:
            feature, _, value = match
            field_mapping = self._map_field_name(feature)
            value = value.replace(",", "")
            filters.append(f"{field_mapping} < {value}")
        
        # Check for demographic group mentions - expanded patterns
        if any(term in query for term in ["high income", "wealthy", "affluent", "rich", "higher income"]):
            filters.append("Median_Income > 75000")
        if any(term in query for term in ["low income", "poor", "lower income", "poverty", "economically disadvantaged"]):
            filters.append("Median_Income < 40000")
        if any(term in query for term in ["urban", "city", "cities", "metropolitan", "metro", "downtown", "densely populated"]):
            filters.append("Population > 100000")
        if any(term in query for term in ["rural", "countryside", "small town", "village", "sparsely populated", "remote"]):
            filters.append("Population < 50000")
        if any(term in query for term in ["young", "younger", "youth", "millennials", "young adults", "young professionals"]):
            filters.append("Young_Adult_Maintainers_Pct > 25")
        if any(term in query for term in ["senior", "older", "elderly", "retirees", "aging population", "retirement age"]):
            filters.append("Senior_Maintainers_Pct > 30")
        
        # Additional demographic filters
        if any(term in query for term in ["high rental", "high renting", "renters", "tenant", "majority rent"]):
            filters.append("Rental_Units_Pct > 50")
        if any(term in query for term in ["high homeownership", "homeowners", "mostly owned", "owner-occupied"]):
            filters.append("Homeownership_Pct > 70")
        if any(term in query for term in ["high unemployment", "jobless", "unemployment crisis"]):
            filters.append("Unemployment_Rate > 8")
        if any(term in query for term in ["low unemployment", "high employment", "job growth"]):
            filters.append("Unemployment_Rate < 4")
            
        # Remove duplicates while preserving order
        unique_filters = []
        for filter_item in filters:
            if filter_item not in unique_filters:
                unique_filters.append(filter_item)
                
        return unique_filters
    
    def _extract_limit(self, query: str) -> Optional[int]:
        """Extract result limit from ranking queries - enhanced to catch more patterns"""
        # Look for patterns like "top 5", "top 10", etc.
        matches = re.findall(r"(?:top|bottom|first|last|best|worst)\s+(\d+)", query)
        if matches:
            return int(matches[0])
            
        # Look for patterns with "highest/lowest N"
        matches = re.findall(r"(?:highest|lowest|best|worst)\s+(\d+)", query)
        if matches:
            return int(matches[0])
        
        # Look for "N highest/lowest"
        matches = re.findall(r"(\d+)\s+(?:highest|lowest|best|worst|top|leading)", query)
        if matches:
            return int(matches[0])
            
        # Default limits for different query types if a ranking term is present but no number
        if any(term in query for term in ["highest", "lowest", "top", "bottom", "best", "worst", "leading"]):
            return 10  # Default to top/bottom 10
            
        return None
    
    def _extract_time_period(self, query: str) -> Optional[Dict[str, str]]:
        """Extract time period for trend analysis"""
        time_period = {}
        
        # Look for start years
        start_year_match = re.search(r"(?:since|from|after)\s+(\d{4})", query)
        if start_year_match:
            time_period["start_year"] = start_year_match.group(1)
        
        # Look for end years
        end_year_match = re.search(r"(?:until|to|through|by)\s+(\d{4})", query)
        if end_year_match:
            time_period["end_year"] = end_year_match.group(1)
            
        # Look for time range descriptions
        if "past year" in query or "last year" in query:
            time_period["period"] = "last_year"
        elif "past 5 years" in query or "last 5 years" in query or "last five years" in query:
            time_period["period"] = "last_5_years"
        elif "past decade" in query or "last decade" in query or "last 10 years" in query:
            time_period["period"] = "last_decade"
            
        return time_period if time_period else None
    
    def _extract_comparison_groups(self, query: str) -> List[Dict[str, Any]]:
        """Extract comparison groups for comparison queries - expanded patterns"""
        groups = []
        
        # Common comparison pairs with more variations
        comparison_pairs = [
            # Urban vs rural comparisons
            (["urban", "city", "metropolitan", "metro"], ["rural", "countryside", "small town", "remote"]),
            
            # Income level comparisons
            (["high income", "wealthy", "affluent", "rich"], ["low income", "poor", "poverty", "economically disadvantaged"]),
            
            # Age-based comparisons
            (["young", "younger", "youth", "millennials"], ["senior", "older", "elderly", "retirees", "aging"]),
            
            # Housing tenure comparisons
            (["homeowner", "owner", "own their home", "owner-occupied"], ["renter", "tenant", "rental", "rent their home"])
        ]
        
        # Check for pairs in the query
        for group1_terms, group2_terms in comparison_pairs:
            if any(term in query for term in group1_terms) and any(term in query for term in group2_terms):
                # Urban vs rural
                if any(term in query for term in group1_terms[0:2]) and any(term in query for term in group2_terms[0:2]):
                    groups = [
                        {"name": "Urban Areas", "filters": ["Population > 100000"]},
                        {"name": "Rural Areas", "filters": ["Population < 50000"]}
                    ]
                # High income vs low income
                elif any(term in query for term in group1_terms[2:4]) and any(term in query for term in group2_terms[2:4]):
                    groups = [
                        {"name": "High Income Areas", "filters": ["Median_Income > 75000"]},
                        {"name": "Low Income Areas", "filters": ["Median_Income < 40000"]}
                    ]
                # Young vs senior
                elif any(term in query for term in group1_terms[4:6]) and any(term in query for term in group2_terms[4:6]):
                    groups = [
                        {"name": "Young Adult Areas", "filters": ["Young_Adult_Maintainers_Pct > 25"]},
                        {"name": "Senior Areas", "filters": ["Senior_Maintainers_Pct > 30"]}
                    ]
                # Homeowner vs renter
                elif any(term in query for term in group1_terms[6:8]) and any(term in query for term in group2_terms[6:8]):
                    groups = [
                        {"name": "Homeowner-Dominant Areas", "filters": ["Homeownership_Pct > 70"]},
                        {"name": "Renter-Dominant Areas", "filters": ["Rental_Units_Pct > 50"]}
                    ]
                break
        
        # Look for more complex comparison specifications - different thresholds
        if not groups and "above" in query and "below" in query:
            # Try to identify what's being compared with thresholds
            for feature, field_name in [
                (["income", "earnings"], "Median_Income"),
                (["population", "residents"], "Population"),
                (["approval", "approvals"], "Mortgage_Approvals")
            ]:
                if any(f in query for f in feature):
                    # Look for threshold values
                    threshold_match = re.search(r"(\d[0-9,.]+)", query)
                    if threshold_match:
                        threshold = threshold_match.group(1).replace(",", "")
                        try:
                            threshold_value = int(threshold)
                            groups = [
                                {"name": f"Above {threshold_value} {field_name}", "filters": [f"{field_name} > {threshold_value}"]},
                                {"name": f"Below {threshold_value} {field_name}", "filters": [f"{field_name} < {threshold_value}"]}
                            ]
                        except ValueError:
                            pass
                    break
        
        return groups
    
    def _extract_regions(self, query: str) -> List[str]:
        """Extract geographic regions from geographic queries - expanded"""
        regions = []
        
        # Canadian provinces and territories
        provinces = {
            "alberta": "Alberta",
            "british columbia": "British Columbia",
            "bc": "British Columbia",
            "manitoba": "Manitoba",
            "new brunswick": "New Brunswick",
            "newfoundland": "Newfoundland and Labrador",
            "newfoundland and labrador": "Newfoundland and Labrador",
            "northwest territories": "Northwest Territories",
            "nova scotia": "Nova Scotia",
            "nunavut": "Nunavut",
            "ontario": "Ontario",
            "prince edward island": "Prince Edward Island",
            "pei": "Prince Edward Island",
            "quebec": "Quebec",
            "saskatchewan": "Saskatchewan",
            "yukon": "Yukon"
        }
        
        # Major Canadian cities
        cities = {
            "toronto": "Toronto",
            "montreal": "Montreal",
            "vancouver": "Vancouver",
            "calgary": "Calgary",
            "edmonton": "Edmonton",
            "ottawa": "Ottawa",
            "winnipeg": "Winnipeg",
            "quebec city": "Quebec City",
            "hamilton": "Hamilton",
            "kitchener": "Kitchener",
            "london": "London",
            "victoria": "Victoria",
            "halifax": "Halifax",
            "oshawa": "Oshawa",
            "windsor": "Windsor",
            "saskatoon": "Saskatoon",
            "regina": "Regina",
            "st. john's": "St. John's",
            "kelowna": "Kelowna",
            "barrie": "Barrie"
        }
        
        # Check for provinces
        for key, value in provinces.items():
            if key in query:
                regions.append(value)
        
        # Check for cities
        for key, value in cities.items():
            if key in query:
                regions.append(value)
        
        # Check for region types
        region_types = {
            "urban": "Urban Areas",
            "rural": "Rural Areas",
            "suburban": "Suburban Areas",
            "metropolitan": "Metropolitan Areas",
            "city center": "City Centers",
            "downtown": "Downtown Areas",
            "coastal": "Coastal Regions"
        }
        
        for key, value in region_types.items():
            if key in query:
                regions.append(value)
                
        # Remove duplicates while preserving order
        unique_regions = []
        for region in regions:
            if region not in unique_regions:
                unique_regions.append(region)
        
        return unique_regions
    
    def _extract_map_type(self, query: str) -> Optional[str]:
        """Extract the type of map visualization requested"""
        if any(term in query for term in ["heat map", "heatmap", "density"]):
            return "heatmap"
        elif any(term in query for term in ["bubble", "circles", "bubble map"]):
            return "bubble"
        elif any(term in query for term in ["choropleth", "colored map", "shaded map"]):
            return "choropleth"
        elif any(term in query for term in ["scatter", "points", "scatter plot"]):
            return "scatter"
        
        # Default to choropleth for geographic queries
        return "choropleth"
    
    def _extract_correlation_relationship(self, query: str) -> Optional[Dict[str, str]]:
        """Extract the specific correlation relationship being asked about"""
        relationship = {}
        
        # Look for "between X and Y" pattern
        between_match = re.search(r"(?:correlation|relationship|connection|link|association)\s+between\s+(\w+(?:\s+\w+)?)\s+and\s+(\w+(?:\s+\w+)?)", query)
        if between_match:
            var1, var2 = between_match.groups()
            relationship["variable1"] = self._map_field_name(var1)
            relationship["variable2"] = self._map_field_name(var2)
            return relationship
            
        # Look for "how does X affect Y" pattern
        affect_match = re.search(r"how\s+(?:does|do)\s+(\w+(?:\s+\w+)?)\s+(?:affect|influence|impact)\s+(\w+(?:\s+\w+)?)", query)
        if affect_match:
            var1, var2 = affect_match.groups()
            relationship["variable1"] = self._map_field_name(var1)
            relationship["variable2"] = self._map_field_name(var2)
            relationship["direction"] = "causal"
            return relationship
        
        # Look for "what factors influence Y" pattern
        factors_match = re.search(r"what\s+(?:factors|variables|elements)\s+(?:affect|influence|impact)\s+(\w+(?:\s+\w+)?)", query)
        if factors_match:
            var = factors_match.group(1)
            relationship["target"] = self._map_field_name(var)
            relationship["type"] = "multi_factor"
            return relationship
            
        return None
    
    def _map_field_name(self, feature_name: str) -> str:
        """Map common terms to field names - expanded mapping"""
        field_mapping = {
            # Income related
            "income": "Median_Income",
            "earnings": "Median_Income",
            "salary": "Median_Income",
            "wealth": "Median_Income",
            "money": "Median_Income",
            
            # Population related
            "population": "Population",
            "people": "Population",
            "residents": "Population",
            "inhabitants": "Population",
            "demographic": "Population",
            
            # Mortgage related
            "approval": "Mortgage_Approvals",
            "approvals": "Mortgage_Approvals",
            "mortgage": "Mortgage_Approvals",
            "mortgages": "Mortgage_Approvals",
            "applications": "Mortgage_Approvals",
            
            # Employment related
            "unemployment": "Unemployment_Rate",
            "jobless": "Unemployment_Rate",
            "employment": "Employment_Rate",
            "jobs": "Employment_Rate",
            "work": "Employment_Rate",
            
            # Housing related
            "homeownership": "Homeownership_Pct",
            "home ownership": "Homeownership_Pct",
            "owners": "Homeownership_Pct",
            "homeowners": "Homeownership_Pct",
            "rental": "Rental_Units_Pct",
            "rent": "Rental_Units_Pct",
            "renting": "Rental_Units_Pct",
            "renters": "Rental_Units_Pct",
            
            # Age related
            "age": "Age",
            "older": "Senior_Maintainers_Pct",
            "seniors": "Senior_Maintainers_Pct",
            "elderly": "Senior_Maintainers_Pct",
            "young": "Young_Adult_Maintainers_Pct",
            "youth": "Young_Adult_Maintainers_Pct",
            "younger": "Young_Adult_Maintainers_Pct"
        }
        
        feature_lower = feature_name.lower()
        
        # Try direct match
        if feature_lower in field_mapping:
            return field_mapping[feature_lower]
        
        # Try partial match (for compound terms)
        for term, mapped_field in field_mapping.items():
            if term in feature_lower:
                return mapped_field
                
        # If no match, title case the input as a fallback
        return feature_name.title()


def process_query(query_text: str) -> Dict[str, Any]:
    """
    Process a natural language query into structured parameters
    
    Args:
        query_text: The natural language query
        
    Returns:
        Dict containing query_type and extracted parameters
    """
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


if __name__ == "__main__":
    # Example usage
    test_queries = [
        "What factors influence mortgage approvals?",
        "Which areas have the highest approval rates?",
        "Compare mortgage approvals between urban and rural areas",
        "How have mortgage approvals changed in high income areas?",
        "Show me the geographic distribution of mortgage approvals"
    ]
    
    for query in test_queries:
        result = process_query(query)
        print(f"\nQuery: {query}")
        print(f"Analysis Type: {result['analysis_type']} (confidence: {result['confidence']:.2f})")
        print(f"Target Variable: {result.get('target_variable', 'Not specified')}")
        print(f"Parameters: {result}")
