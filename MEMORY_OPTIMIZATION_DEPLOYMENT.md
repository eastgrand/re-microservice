
# Memory Optimization Deployment Instructions

## Files Added:
- memory_utils.py - Core memory optimization utilities
- memory_optimization_patch.py - Integration patterns for app.py
- outlier_detection_patch.py - Specific patches for outlier detection
- scenario_analysis_patch.py - Specific patches for scenario analysis  
- feature_interactions_patch.py - Specific patches for feature interactions

## Integration Steps:

### 1. Add imports to app.py (at top after existing imports):
```python
from memory_utils import (
    memory_safe_shap_wrapper, 
    get_endpoint_config,
    batch_shap_calculation,
    force_garbage_collection,
    get_memory_usage,
    ENDPOINT_MEMORY_CONFIGS
)
```

### 2. Add memory monitoring endpoint (add to app.py):
```python
@app.route('/memory-status', methods=['GET'])
def memory_status():
    """Get current memory usage and limits"""
    try:
        current_memory = get_memory_usage()
        
        return safe_jsonify({
            "current_memory_mb": current_memory,
            "memory_configs": ENDPOINT_MEMORY_CONFIGS,
            "status": "healthy" if current_memory < 800 else "high_memory"
        })
    except Exception as e:
        return safe_jsonify({"error": str(e)}, 500)
```

### 3. Replace SHAP calculations in endpoints:

#### Pattern to replace:
```python
shap_values = explainer.shap_values(X_sample)
```

#### Replace with:
```python
from memory_utils import batch_shap_calculation, get_endpoint_config

config = get_endpoint_config(request.path)
shap_values = batch_shap_calculation(
    explainer, X_sample,
    batch_size=config['batch_size'],
    max_memory_mb=config['memory_limit_mb']
)
```

### 4. Add intelligent sampling for large datasets:

#### Pattern to replace:
```python
X_sample = X.sample(n=sample_size, random_state=42)
```

#### Replace with:
```python
from memory_utils import memory_safe_sample_selection, get_endpoint_config

config = get_endpoint_config(request.path)
sampled_df = memory_safe_sample_selection(
    df, target_field,
    config['max_samples'],
    config['sampling_strategy']
)
X_sample = sampled_df[feature_columns]
```

## Memory Limits by Endpoint:
- /analyze: 500 samples, 100 batch size, 800MB limit
- /outlier-detection: 200 samples, 50 batch size, 600MB limit  
- /scenario-analysis: 100 samples, 25 batch size, 500MB limit
- /spatial-clusters: 300 samples, 75 batch size, 700MB limit
- /segment-profiling: 150 samples, 50 batch size, 600MB limit
- /comparative-analysis: 200 samples, 50 batch size, 600MB limit
- /feature-interactions: 800 samples, 150 batch size, 900MB limit

## Expected Results:
- Handle datasets up to 5000 records without memory issues
- Intelligent sampling ensures statistical validity
- Batch processing prevents OOM errors
- 2-5x improvement in maximum dataset size
- Consistent response times under 2 minutes

## Testing:
After deployment, test with progressively larger sample sizes:
```bash
python scripts/test-memory-optimization.py
```

Should show endpoints working with 1000+ samples instead of 500.
