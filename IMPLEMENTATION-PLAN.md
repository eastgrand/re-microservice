# SHAP Dynamic Query Processing Implementation Plan

> Created: May 16, 2025

This document outlines the step-by-step implementation plan for enhancing the SHAP microservice to better process dynamic natural language queries. The plan is organized into manageable phases with specific deliverables.

## Phase 1: Query Classification (Weeks 1-2)

### 1.1 Create a Basic Query Classifier (Week 1)
- **Goal**: Implement a simple rule-based classifier to categorize incoming queries
- **Deliverables**:
  - `query_classifier.py` module with initial classification functions
  - Test cases for different query types
  - Integration with app.py for basic routing

### 1.2 Implement Parameter Extractor (Week 1-2)
- **Goal**: Extract structured parameters from natural language queries
- **Deliverables**:
  - `parameter_extractor.py` module for entity extraction
  - Support for extracting:
    - Target variables (e.g., "mortgage approvals", "income")
    - Comparative values (e.g., "greater than 50,000")
    - Geographic references (e.g., "in Ontario", "urban areas")
    - Demographic groups (e.g., "young adults", "homeowners")

### 1.3 Create Query Templates (Week 2)
- **Goal**: Define templates for different query types to standardize responses
- **Deliverables**:
  - `query_templates.py` with response formats for each query type
  - Integration with job result formatting
  - Documentation for frontend developers

## Phase 2: Enhanced SHAP Analysis (Weeks 3-4)

### 2.1 Implement Dynamic Data Filtering (Week 3)
- **Goal**: Filter data based on extracted query parameters
- **Deliverables**:
  - Enhanced filtering system in `analysis_worker` function
  - Support for complex filter combinations
  - Memory-optimized implementation

### 2.2 Add Context-Aware Result Ranking (Week 3-4)
- **Goal**: Prioritize results based on query intent
- **Deliverables**:
  - Functions to rank and sort results by relevance
  - Option to limit results to most relevant items
  - Weighting system for feature importance

### 2.3 Implement Adaptive Explanations (Week 4)
- **Goal**: Generate natural language explanations tailored to query type
- **Deliverables**:
  - `explanation_generator.py` module
  - Templates for different explanation types
  - Integration with analysis results

## Phase 3: Performance & Integration (Week 5)

### 3.1 Optimize Performance
- **Goal**: Ensure new features don't impact system performance
- **Deliverables**:
  - Memory usage benchmarks for enhanced processing
  - Caching strategy for common queries
  - Optimized implementation for Render deployment

### 3.2 API Enhancement
- **Goal**: Extend API to support new capabilities
- **Deliverables**:
  - Updated API documentation
  - New endpoint for query suggestions
  - Backward compatibility layer

### 3.3 Testing & Documentation
- **Goal**: Comprehensive testing and documentation
- **Deliverables**:
  - Test suite for all new features
  - Updated documentation for frontend developers
  - Sample queries for demonstration

## Implementation Details

### Query Classification Approach

The initial implementation will use a rule-based approach with the following categories:

1. **Correlation Analysis** 
   - Example: "What factors influence mortgage approvals?"
   - Detection: Keywords like "influence", "affect", "impact", "related to"
   - Analysis: Standard SHAP feature importance

2. **Ranking Analysis**
   - Example: "Which areas have the highest mortgage approval rates?"
   - Detection: Keywords like "highest", "lowest", "top", "ranking"
   - Analysis: Sorting by target variable

3. **Comparison Analysis**
   - Example: "Compare mortgage approvals between urban and rural areas"
   - Detection: Keywords like "compare", "difference", "versus", "between"
   - Analysis: Group by category and compare statistics

4. **Trend Analysis**
   - Example: "How have mortgage approvals changed in areas with increasing income?"
   - Detection: Keywords like "change", "trend", "increase", "decrease", "over time"
   - Analysis: Correlation between target and change variables

5. **Geographical Analysis**
   - Example: "Show mortgage approval patterns by location"
   - Detection: Keywords like "map", "region", "location", "geographic"
   - Analysis: Geographic clustering and visualization

### Code Structure

New modules to be created:

```
shap-microservice/
  ├── query_processing/
  │   ├── __init__.py
  │   ├── classifier.py         # Query classification logic
  │   ├── parameter_extractor.py # Extract structured parameters from queries
  │   ├── templates.py          # Response templates
  │   └── explanation.py        # Natural language explanation generation
  │
  ├── analysis/
  │   ├── __init__.py
  │   ├── dynamic_filtering.py  # Enhanced data filtering
  │   ├── result_ranking.py     # Context-aware result sorting
  │   └── specialized_analysis.py # Type-specific analysis functions
```

### Integration with Frontend

The frontend will need minor updates to:

1. Send more detailed query information
2. Properly display enhanced responses
3. Handle type-specific visualizations

These changes will be done in coordination with the frontend team to ensure a smooth transition.

## Timeline

- **Week 1**: Basic query classification and parameter extraction
- **Week 2**: Response templates and initial integration
- **Week 3**: Dynamic filtering and result ranking
- **Week 4**: Adaptive explanations
- **Week 5**: Performance optimization, testing, and documentation
