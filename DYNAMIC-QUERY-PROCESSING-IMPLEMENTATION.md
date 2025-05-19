# Dynamic Query Processing Implementation

*Date: May 16, 2025*

This document describes the implementation of the dynamic query processing system for the SHAP microservice. The system allows users to submit natural language queries, which are then classified, processed, and responded to with appropriately formatted results.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Module Structure](#module-structure)
4. [Query Types](#query-types)
5. [Integration Points](#integration-points)
6. [Testing](#testing)
7. [Future Improvements](#future-improvements)

## Overview

The dynamic query processing system transforms natural language queries into structured analysis parameters for the SHAP microservice. It supports various query types, extracts relevant parameters, and formats the results based on the query intent.

Key capabilities include:
- Classification of natural language queries into specific types
- Extraction of parameters such as target variables, filters, regions, and time periods
- Specialized handling for different query types
- Enhanced result formatting based on query intent
- Seamless integration with the existing Flask API

## System Architecture

The system follows a modular architecture:

1. **Query Classification**: Natural language queries are classified into specific types using pattern matching.
2. **Parameter Extraction**: Relevant parameters are extracted based on the query type.
3. **Processing**: Structured parameters are used to prepare the analysis.
4. **Analysis**: The existing SHAP analysis engine processes the structured parameters.
5. **Result Formatting**: Raw analysis results are formatted based on the original query type.
6. **Response**: The formatted results are returned to the user.

## Module Structure

The system is organized into the following modules:

```
query_processing/
  ├── __init__.py                # Package initialization and exports
  ├── classifier.py              # Query classification and parameter extraction
  ├── processor.py               # Main processing pipeline
  ├── integration.py             # Flask API integration
  └── handlers/                  # Specialized query handlers
      ├── __init__.py
      ├── base_handler.py        # Base handler class
      ├── correlation_handler.py # Correlation query handler
      ├── ranking_handler.py     # Ranking query handler
      ├── comparison_handler.py  # Comparison query handler
      ├── trend_handler.py       # Trend query handler
      ├── geographic_handler.py  # Geographic query handler
      └── mixed_handler.py       # Mixed query handler
```

## Query Types

The system supports the following query types:

1. **Correlation**: Queries about factors that influence a target variable
   - Example: "What factors influence mortgage approvals?"
   - Parameters: target_variable, correlation_relationship

2. **Ranking**: Queries about areas with highest/lowest values
   - Example: "Which regions have the highest mortgage approval rates?"
   - Parameters: target_variable, limit

3. **Comparison**: Queries comparing different groups
   - Example: "Compare mortgage approvals between urban and rural areas"
   - Parameters: target_variable, comparison_groups

4. **Trend**: Queries about changes over time
   - Example: "How have approval rates changed over the past 5 years?"
   - Parameters: target_variable, time_period

5. **Geographic**: Queries about spatial distribution
   - Example: "Show me a map of mortgage approvals across Canada"
   - Parameters: target_variable, regions, map_type

6. **Mixed**: Queries combining multiple types
   - Example: "What factors drive the difference in approval rates between urban and rural areas?"
   - Parameters: combined parameters from multiple types

## Integration Points

The system integrates with the Flask application at the following points:

1. **POST /analyze endpoint**: Detects natural language queries and processes them
2. **analysis_worker function**: Uses query type for specialized processing and result formatting
3. **job_status endpoint**: Returns formatted results based on the original query type

Integration flow:
1. User submits a natural language query to /analyze
2. The system detects it as a natural language query and processes it
3. The query is classified and parameters are extracted
4. The analysis worker processes the structured parameters
5. Results are formatted based on the query type
6. Formatted results are returned to the user

## Testing

The system includes two test scripts:

1. **test_query_classifier.py**: Tests the classifier with various queries
2. **test_enhanced_queries.py**: Tests the entire system with API calls
3. **test_query_integration.py**: Tests the integration between modules

To run the tests:

```bash
# Test the classifier
python test_query_classifier.py

# Test the API integration
python test_enhanced_queries.py

# Test module integration
python test_query_integration.py
```

## Future Improvements

Planned future improvements include:

1. **Advanced NLP**: Incorporate more sophisticated NLP techniques for query understanding
2. **Personalization**: Adapt to user preferences and query history
3. **Dynamic Visualizations**: Generate visualizations based on query type
4. **Conversational Context**: Maintain context across multiple queries
5. **Feedback Loop**: Learn from user interactions to improve results
6. **Expanded Query Types**: Support more specialized query types
7. **Multi-language Support**: Process queries in multiple languages
