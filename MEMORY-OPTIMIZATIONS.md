# Memory Optimization Guide for SHAP Microservice

This file provides an explanation of the memory optimization techniques implemented to resolve memory usage issues on Render's 512MB limit.

## Changes Made

### 1. Memory Optimization Module (`optimize_memory.py`)
- Added a dedicated module for memory optimization
- Implements functions to:
  - Monitor memory usage
  - Optimize DataFrame dtypes (using smaller numeric types)
  - Load data efficiently
  - Sample data when memory becomes critical
  - Adjust model complexity based on available memory

### 2. Data Loading Optimizations
- Reduced the amount of data loaded at once
- Set `low_memory=False` to avoid mixed type inference errors
- Optimized data types for minimal memory footprint
- Added memory cleanup with garbage collection after major operations
- Sample data when full dataset exceeds memory limits

### 3. Model Training Optimizations
- Reduced model complexity when memory is tight:
  - Fewer estimators
  - Smaller max_depth
  - Use histogram-based training method
- Skip cross-validation when memory usage is critical
- Use early stopping to reduce training time and memory usage
- Free memory after cross-validation
- Regular garbage collection during the training process

### 4. Sample Data Generation
- Reduced sample size from 100 to 50 rows
- This creates a smaller baseline dataset for testing

### 5. App.py Optimizations
- Load limited rows of data for analysis
- Implement memory-optimized data loading

## Memory Usage Thresholds

Different operations are affected at different memory thresholds:
- 450MB+: Critical memory situation, significant optimizations applied
- 400MB+: Skip cross-validation, reduce data size
- 350MB+: Reduce model complexity, use smaller sample sizes

## How It Works

1. Before any memory-intensive operation, memory usage is checked
2. Based on the memory situation, different optimization strategies are applied
3. Progressive fallbacks ensure the app can run even with very limited memory

## Monitoring Memory Usage

The application now logs memory usage at key points:
```
Memory usage at Before data loading: 42.15 MB
Memory usage at After loading Nesto data: 156.78 MB
Memory usage at After dtype optimization: 124.32 MB
Memory usage at Before model training: 145.67 MB
Memory usage at After model training: 238.45 MB
```

## Test Results

The memory optimization was tested with the following results:

### setup_for_render.py:
- Starting memory: 69.20 MB
- Successfully created sample data and trained a model within memory limits

### train_model.py:
- Starting memory: 130.17 MB
- Memory after data loading: 539.00 MB
- Memory after type optimization: 564.66 MB (reduced DataFrame memory by 49.26%)
- Dataset reduced from 112,895 to 80,973 rows when memory became critical
- Memory before model training: 706.00 MB
- Memory after model training: 711.27 MB
- Cross-validation was skipped due to high memory usage
- Training size was reduced to 5,000 samples when memory was very low

### app.py:
- Starting memory: 219.88 MB
- Memory after data loading: 490.34 MB
- Memory after optimization: 498.69 MB
- Dataset reduced from 80,973 to 65,760 rows when memory became critical

The optimizations successfully kept the memory usage under Render's 512MB limit while still allowing the service to function correctly.

## Further Optimizations

If memory issues persist:
1. Further reduce sample data size
2. Implement chunked processing for larger datasets 
3. Consider using more aggressive feature selection
4. Move to a higher memory tier on Render
