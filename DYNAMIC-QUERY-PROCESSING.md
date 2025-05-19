# SHAP Microservice Dynamic Query Processing - Reference Guide

> Created: May 16, 2025
> Status: Initial Assessment

## Current System Assessment

### 1. Query Processing Pipeline

#### Findings from Code Analysis:
- [x] The system uses a Flask API with `/analyze` endpoint for processing queries
- [x] Queries are processed asynchronously using Redis Queue (RQ)
- [x] Main API flow: `/analyze` (POST) → Redis Queue → `analysis_worker` → `/job_status/<job_id>` (GET)
- [x] Query parameters include `analysis_type`, `target_variable`, and `demographic_filters` 
- [x] Currently supports two analysis types: 'correlation' and 'ranking'
- [x] Default target variable is 'Mortgage_Approvals'
- [x] Filters can be applied using simple comparison strings (e.g., "Income > 50000")

#### Key Files Examined:
- [x] `/Users/voldeck/code/shap-microservice/app.py` - Main API endpoints and worker function
- [x] `/Users/voldeck/code/shap-microservice/train_model.py` - Model training and SHAP analysis
- [x] `/Users/voldeck/code/shap-microservice/data_versioning.py` - Dataset versioning
- [x] `/Users/voldeck/code/shap-microservice/map_nesto_data.py` - Data field mappings
- [x] `/Users/voldeck/code/shap-microservice/shap_memory_fix.py` - Memory optimizations for SHAP

#### Areas Needing More Investigation:
- [ ] How the frontend interprets and formats the SHAP values returned
- [ ] What additional query types might be valuable
- [ ] How NLP queries are currently transformed into parameters

### 2. NLP Capabilities

#### Findings from Code Analysis:
- [x] Currently no dedicated NLP processing in the backend
- [x] Filter parsing uses simple string operations (split on '<' or '>')
- [x] Limited query interpretation (only 'correlation' and 'ranking' types)
- [x] No semantic analysis or intent recognition

#### Areas Needing More Investigation:
- [ ] Frontend NLP processing (if any)
- [ ] User query patterns and common requests
- [ ] Opportunities for query classification

### 3. SHAP Analysis Structure

#### Findings from Code Analysis:
- [x] Uses XGBoost model with TreeExplainer from SHAP
- [x] Single model used for all analyses
- [x] Memory-optimized processing for large datasets
- [x] Feature importance calculation based on mean absolute SHAP values
- [x] Results include raw SHAP values, feature importance rankings, and a natural language summary
- [x] Data can be filtered but feature selection is static

#### Areas Needing More Investigation:
- [ ] Additional SHAP explainer types that could be used
- [ ] How to dynamically select features based on query
- [ ] Performance impact of more sophisticated SHAP analyses

### 4. Data Source & Filtering

#### Findings from Code Analysis:
- [x] Primary data source is Nesto mortgage application data
- [x] Backup data sources include sample data and legacy sales data
- [x] Extensive field mappings from source data to standardized fields
- [x] Simple filtering based on threshold comparisons
- [x] Results sorted by target variable in descending order
- [x] Memory optimization techniques for large datasets

#### Areas Needing More Investigation:
- [ ] Query-based filtering possibilities
- [ ] Opportunities for dynamic data aggregation
- [ ] Potential additional data sources

## Updated Implementation Plan

### Phase 1: Query Classification System
- [ ] Implement basic query type classifier
  - Correlation analysis (existing)
  - Ranking/comparison analysis (existing)
  - Factor/cause analysis (new)
  - Trend/change analysis (new)
  - Geographical pattern analysis (new)
- [ ] Add structured parameter extraction
  - Target variables
  - Comparison thresholds
  - Geographic regions
  - Demographic segments
- [ ] Create response templates for each query type

### Phase 2: Dynamic Data Filtering
- [ ] Implement query-based filtering system
  - Filter by entity mentions (locations, demographics)
  - Filter by numerical ranges mentioned in query
  - Support for advanced operators (greater than, less than, between)
- [ ] Add dynamic sorting options based on query focus
- [ ] Support for grouping/aggregation based on query keywords

### Phase 3: Enhanced SHAP Visualization & Explanation
- [ ] Implement result formatting based on query type
- [ ] Add natural language explanation generation
  - Factor correlations with target variable
  - Geographic patterns description
  - Comparison between demographic groups
- [ ] Provide confidence metrics for the analysis

### Phase 4: Technical Implementation
- [ ] Modify Flask endpoints to support enhanced query processing
- [ ] Update worker function to handle new query types
- [ ] Add response formatting based on query type
- [ ] Create feedback loop for query improvement

## Integration Points

### Frontend Integration:
- The `/analyze` endpoint should be enhanced to accept more detailed query parameters
- The response format will maintain backward compatibility while adding new fields
- The `/job_status` endpoint will return enhanced results with query-specific formatting

### Data Processing Integration:
- The `analysis_worker` function in app.py will need modifications to handle new query types
- The SHAP analysis can be extended to provide different explanation types

### Model Integration:
- Initial implementation will use the existing XGBoost model
- Future versions can add dynamic model selection based on query type
