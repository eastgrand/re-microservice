# Query Classifier Improvements

## Key Improvements Made

### 1. Enhanced Pattern Recognition
- **Expanded pattern library**: Added 3-4x more patterns for each query type
- **Mixed query detection**: Added capability to identify complex queries spanning multiple types
- **More sophisticated RegEx**: Improved pattern matching for various query structures

### 2. Improved Parameter Extraction
- **Target variable detection**: Expanded dictionary of common mortgage-related terms
- **Time period extraction**: Added ability to identify time ranges and specific years
- **Region identification**: Added support for Canadian provinces and cities
- **Map type detection**: Can now identify different visualization types (heatmap, choropleth, etc.)
- **Correlation relationship extraction**: Identifies specific variables in relationship questions

### 3. Better Demographic Filtering
- **Enhanced filter generation**: Generates appropriate filters based on query context
- **Comparison group detection**: Identifies paired groups for comparative analysis
- **Threshold detection**: Extracts numeric thresholds from queries

### 4. Performance Improvements
- **Reduced unknown classifications**: From 33.3% to 10.9% of queries
- **Increased confidence scores**: Average confidence up from 0.17 to 0.29
- **Better structured results**: More consistent parameter extraction

## Test Results Summary

| Query Type   | Count | Percentage | Avg Confidence |
|--------------|-------|------------|---------------|
| CORRELATION  | 11    | 23.9%      | 0.25          |
| RANKING      | 8     | 17.4%      | 0.35          |
| COMPARISON   | 7     | 15.2%      | 0.39          |
| GEOGRAPHIC   | 7     | 15.2%      | 0.30          |
| TREND        | 6     | 13.0%      | 0.38          |
| MIXED        | 2     | 4.3%       | 0.43          |
| UNKNOWN      | 5     | 10.9%      | 0.00          |

## Integration Roadmap

### 1. Create Query Processing Module
- Create a `query_processing` package structure
- Move improved classifier there with proper imports
- Add unit tests

### 2. Modify Flask API Endpoints
```python
# Integration with app.py's /analyze endpoint
@app.route('/analyze', methods=['POST'])
@require_api_key
def analyze():
    logger.info("/analyze endpoint called (ASYNC)")
    # Get the raw query from request
    raw_query = request.json.get('query', '')
    
    # If a raw natural language query is provided, process it
    if raw_query:
        from query_processing.classifier import process_query
        query_params = process_query(raw_query)
        # Merge with any explicit parameters
        query = {**request.json, **query_params}
        logger.info(f"Processed NL query: {raw_query} → {query_params['analysis_type']}")
    else:
        # Use original query as-is if no raw query
        query = request.json
        
    if not query:
        return jsonify({"error": "No query provided"}), 400
        
    try:
        # Enqueue the analysis job
        job = queue.enqueue(analysis_worker, query, job_timeout=600)
        logger.info(f"Enqueued job {job.id}")
        return jsonify({"job_id": job.id, "status": "queued"}), 202
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"[ANALYZE ENQUEUE ERROR] Exception: {e}\nTraceback:\n{tb}")
        return jsonify({"success": False, "error": str(e), "traceback": tb}), 500
```

### 3. Enhance Analysis Worker Function
```python
# Modified analysis_worker to handle different query types
def analysis_worker(query):
    import time
    logger.info(f"[RQ WORKER] analysis_worker called")
    ensure_model_loaded()
    try:
        # Get query parameters with appropriate defaults
        analysis_type = query.get('analysis_type', DEFAULT_ANALYSIS_TYPE)
        target_variable = query.get('target_variable', query.get('target', DEFAULT_TARGET))
        filters = query.get('demographic_filters', [])
        
        # New parameters from enhanced query processing
        limit = query.get('limit', 10)  # Default to top 10 for ranking
        time_period = query.get('time_period', {})
        regions = query.get('regions', [])
        map_type = query.get('map_type', 'choropleth')
        comparison_groups = query.get('comparison_groups', [])
        
        # Rest of the worker function...
        # [existing code for data filtering and SHAP processing]
        
        # Modify result formatting based on query type
        if analysis_type == 'correlation':
            # Enhanced correlation analysis
            # [implementation]
        elif analysis_type == 'ranking':
            # Limit results based on the limit parameter
            results = results[:limit]
            # [implementation]
        elif analysis_type == 'comparison':
            # Handle comparison between groups
            # [implementation]
        elif analysis_type == 'trend':
            # Handle trend analysis with time period
            # [implementation]
        elif analysis_type == 'geographic':
            # Add region filtering and map type
            # [implementation]
        elif analysis_type == 'mixed':
            # Handle mixed query types
            # [implementation]
        else:
            # Default handling
            # [implementation]
            
        # Return enhanced results
        return {
            "success": True,
            "results": results,
            "summary": summary,
            "feature_importance": feature_importance,
            "shap_values": shap_values_dict,
            "version_info": version_info,
            "query_info": {
                "type": analysis_type,
                "target": target_variable,
                # Include additional query parameters
                "parameters": {
                    "filters": filters,
                    "limit": limit if 'limit' in query else None,
                    "time_period": time_period if time_period else None,
                    "regions": regions if regions else None,
                    "map_type": map_type if 'map_type' in query else None
                }
            }
        }
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"[ANALYSIS JOB ERROR] Exception: {e}\nTraceback:\n{tb}")
        return {"success": False, "error": str(e), "traceback": tb}
```

### 4. Create Module Structure
```
shap-microservice/
  ├── query_processing/
  │   ├── __init__.py
  │   ├── classifier.py     # Improved classifier code
  │   ├── parameters.py     # Parameter extraction functions
  │   └── templates.py      # Result templates by query type
  │
  ├── analysis/
      ├── __init__.py
      └── query_handlers/
          ├── __init__.py
          ├── correlation.py
          ├── ranking.py
          ├── comparison.py
          ├── trend.py
          ├── geographic.py
          └── mixed.py
```

### 5. Testing and Deployment
1. Create a comprehensive test suite with examples of each query type
2. Add integration tests to verify end-to-end flow 
3. Document the new NLP capabilities for frontend developers
4. Deploy the enhanced backend to Render

## Remaining Work
- Create specialized analysis functions for each query type
- Implement result formatting templates
- Add frontend examples for each query type
