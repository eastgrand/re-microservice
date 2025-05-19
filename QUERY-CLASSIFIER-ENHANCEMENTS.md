# Query Classification Enhancements

## Overview
This document summarizes the enhancements made to the query classifier to fix two key issues:
1. Properly classifying previously unclassified natural language queries
2. Adding appropriate error handling for time-related queries that cannot be processed due to data limitations

## Changes Made

### 1. Enhanced Pattern Matching for Unclassified Queries

We improved the regex pattern matching for several query types:

#### Correlation Queries
Added patterns for demographic factor prediction queries:
```python
r"which demographic factors predict.*",
r"what demographic factors affect.*",
r"how do demographics affect.*",
r"what demographics are associated with.*",
r"demographics related to.*",
```
These patterns help classify queries like "Which demographic factors predict high approval rates?"

#### Ranking Queries
Added patterns for complex conditions and area-specific queries:
```python
r"what (?:areas|regions|locations).*(?:high|low).*(?:but|despite|although|yet|and|while).*(?:high|low).*",
r".*(?:areas|regions|locations).*(?:high|low).*(?:and|but|with|while).*(?:low|high).*",
```
These patterns help classify queries like "What areas have high approvals but low income?"

#### Comparison Queries
Added patterns for specific comparative questions about regions:
```python
r"are.*(?:approvals|approval rates|rates).*(?:higher|lower|better|worse).*(?:in|among|across).*",
r"(?:difference|gap) between.*(?:urban|rural|large|small|high|low).*(?:and).*(?:urban|rural|large|small|high|low).*",
```
These patterns help classify queries like "Are mortgage approvals higher in urban or rural regions?"

#### Trend Queries
Added patterns for common trend-related expressions:
```python
r"show the trend.*",
r"historical pattern.*",
```
These patterns help identify time-related queries like "Show the trend in approval rates" or "Historical pattern of approvals"

### 2. Error Handling for Time-Related Queries

We implemented proper error handling for time-related queries when we don't have the necessary historical data:

1. Added a flag to indicate time-series data availability:
```python
# Flag for time series data availability
self.has_time_series_data = False
```

2. Added error handling in the parameter extraction for trend queries:
```python
if query_type == QueryType.TREND or query_type == QueryType.MIXED:
    # Add time period if found
    if time_period:
        params["time_period"] = time_period
    
    # Check if this is a time-related query that we can't process
    if not self.has_time_series_data:
        params["error"] = {
            "code": "TIME_DATA_UNAVAILABLE",
            "message": "We don't have historical time series data available to answer this trend query.",
            "details": "The current dataset only contains the most recent snapshot of mortgage data without historical trends."
        }
```

This ensures that any trend-related query will receive an informative error message rather than returning incomplete or incorrect results.

## Testing Results

The updated classifier now properly classifies all the previously unclassified queries:
- "Are mortgage approvals higher in urban or rural regions?" → comparison (0.49)
- "What areas have high approvals but low income?" → ranking (0.65)
- "Which demographic factors predict high approval rates?" → correlation (0.42)
- "Show the trend in approval rates in high income areas" → trend (0.33) with error
- "Historical pattern of approvals in urban areas" → trend (0.33) with error

All time-related queries now include the appropriate error information to be handled gracefully by the front end.

## Integration Notes

When integrating this updated classifier with the query processor module, ensure that error messages from the classifier are properly propagated to the API response. This will ensure that users receive informative messages when certain queries cannot be answered due to data limitations.

## Future Enhancements

For future development:
1. Add a configuration option to toggle the `has_time_series_data` flag when historical data becomes available
2. Implement a graceful fallback mechanism for trend queries that offers alternative analysis based on the available data
3. Further enhance pattern matching to cover more natural language variations
