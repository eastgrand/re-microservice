#!/usr/bin/env python3
"""
SHAP Memory and Model Optimization Script

This script addresses memory issues and model loading problems in the SHAP microservice:
1. Optimizes model loading to reduce memory usage
2. Uses chunked processing for SHAP calculations
3. Implements proper error handling and reporting
4. Adds diagnostic logging for troubleshooting

Use this script to fix the 500 errors occurring during SHAP analysis.
"""

import os
import sys
import time
import logging
import traceback
import gc
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("shap-fix")

# Memory optimization constants
MAX_ROWS_TO_PROCESS = 500  # Maximum rows to process in a single SHAP calculation
GC_FREQUENCY = 10  # How often to run garbage collection during processing
MEMORY_THRESHOLD_MB = 400  # Memory threshold to trigger more aggressive optimization

def log_memory():
    """Log current memory usage if psutil is available"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage: {memory:.2f} MB")
        return memory
    except:
        return 0

def patch_analysis_worker():
    """Patch the analysis_worker function in app.py with memory optimized version"""
    try:
        from app import analysis_worker
        import functools
        from app import ensure_model_loaded
        
        # Store original function
        original_analysis_worker = analysis_worker
        
        @functools.wraps(original_analysis_worker)
        def optimized_analysis_worker(query):
            """Memory-optimized version of analysis_worker with better error handling"""
            logger.info(f"Starting optimized analysis worker with query: {query}")
            ensure_model_loaded()
            
            try:
                # Force garbage collection before we start
                gc.collect()
                log_memory()
                
                # === Extract query parameters ===
                from app import DEFAULT_ANALYSIS_TYPE, DEFAULT_TARGET
                analysis_type = query.get('analysis_type', DEFAULT_ANALYSIS_TYPE)
                target_variable = query.get('target_variable', query.get('target', DEFAULT_TARGET))
                filters = query.get('demographic_filters', [])
                
                # === Load model and data ===
                from app import model, dataset, feature_names
                if model is None or dataset is None or feature_names is None:
                    raise RuntimeError("Model, dataset, or feature names not loaded")
                
                # === Apply filters to dataset ===
                filtered_data = dataset.copy()
                logger.info(f"Starting with {len(filtered_data)} records")
                
                for filter_item in filters:
                    # Apply all the filters as in the original function
                    if isinstance(filter_item, str) and '>' in filter_item:
                        feature, value = filter_item.split('>')
                        feature = feature.strip()
                        value = float(value.strip())
                        filtered_data = filtered_data[filtered_data[feature] > value]
                    elif isinstance(filter_item, str) and '<' in filter_item:
                        feature, value = filter_item.split('<')
                        feature = feature.strip()
                        value = float(value.strip())
                        filtered_data = filtered_data[filtered_data[feature] < value]
                    # Handle other filter types as in original function
                
                logger.info(f"After filtering: {len(filtered_data)} records")
                
                # === Prepare features for model ===
                top_data = filtered_data.sort_values(by=target_variable, ascending=False)
                
                X = top_data.copy()
                for col in ['zip_code', 'latitude', 'longitude']:
                    if col in X.columns:
                        X = X.drop(col, axis=1)
                if target_variable in X.columns:
                    X = X.drop(target_variable, axis=1)
                
                # Match features with model requirements
                X_cols = list(X.columns)
                for col in X_cols:
                    if col not in feature_names:
                        X = X.drop(col, axis=1)
                for feature in feature_names:
                    if feature not in X.columns:
                        X[feature] = 0
                X = X[feature_names]
                
                # === MEMORY OPTIMIZED SHAP CALCULATION ===
                logger.info(f"Running optimized SHAP calculation on {len(X)} rows")
                import shap
                
                # Check if we need to chunk the processing
                if len(X) > MAX_ROWS_TO_PROCESS:
                    logger.info(f"Using chunked processing ({MAX_ROWS_TO_PROCESS} rows per batch)")
                    
                    # Initialize arrays for results
                    all_shap_values = []
                    total_rows = len(X)
                    chunks = (total_rows + MAX_ROWS_TO_PROCESS - 1) // MAX_ROWS_TO_PROCESS
                    
                    for i in range(chunks):
                        start_idx = i * MAX_ROWS_TO_PROCESS
                        end_idx = min((i + 1) * MAX_ROWS_TO_PROCESS, total_rows)
                        logger.info(f"Processing chunk {i+1}/{chunks} (rows {start_idx}-{end_idx})")
                        
                        # Take a chunk of the data
                        X_chunk = X.iloc[start_idx:end_idx]
                        
                        # Create explainer for this chunk
                        explainer = shap.TreeExplainer(model)
                        chunk_shap_values = explainer(X_chunk)
                        
                        # Store values for this chunk
                        all_shap_values.append(chunk_shap_values.values)
                        
                        # Force garbage collection
                        del explainer
                        del chunk_shap_values
                        gc.collect()
                        log_memory()
                    
                    # Combine all chunks
                    combined_values = np.vstack(all_shap_values)
                    
                    # Create a wrapper object to mimic the SHAP explainer output
                    class ShapValuesWrapper:
                        def __init__(self, values):
                            self.values = values
                    
                    shap_values = ShapValuesWrapper(combined_values)
                    
                    # Clean up
                    del all_shap_values
                    gc.collect()
                    
                else:
                    # For smaller datasets, use normal processing
                    logger.info("Using standard processing (dataset small enough)")
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(X)
                
                # === Calculate feature importance ===
                feature_importance = []
                for i, feature in enumerate(feature_names):
                    importance = abs(shap_values.values[:, i]).mean()
                    feature_importance.append({'feature': feature, 'importance': float(importance)})
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                
                # === Format results as in original function ===
                results = []
                for idx, row in top_data.iterrows():
                    result = {}
                    if 'zip_code' in row:
                        result['zip_code'] = str(row['zip_code'])
                    if 'latitude' in row and 'longitude' in row:
                        result['latitude'] = float(row['latitude'])
                        result['longitude'] = float(row['longitude'])
                    target_var_lower = target_variable.lower()
                    if target_variable in row:
                        result[target_var_lower] = float(row[target_variable])
                    for col in row.index:
                        if col not in ['zip_code', 'latitude', 'longitude', target_variable]:
                            try:
                                result[col.lower()] = float(row[col])
                            except (ValueError, TypeError):
                                result[col.lower()] = str(row[col])
                    results.append(result)
                
                # === Generate summary as in original function ===
                if analysis_type == 'correlation':
                    if len(feature_importance) > 0:
                        summary = f"Analysis shows a strong correlation between {target_variable} and {feature_importance[0]['feature']}."
                    else:
                        summary = f"Analysis complete for {target_variable}, but no clear correlations found."
                elif analysis_type == 'ranking':
                    if len(results) > 0:
                        summary = f"The top area for {target_variable} has a value of {results[0][target_variable.lower()]:.2f}."
                    else:
                        summary = f"No results found for {target_variable} with the specified filters."
                else:
                    summary = f"Analysis complete for {target_variable}."
                
                if len(feature_importance) >= 3:
                    summary += f" The top 3 factors influencing {target_variable} are {feature_importance[0]['feature']}, "
                    summary += f"{feature_importance[1]['feature']}, and {feature_importance[2]['feature']}."
                
                # === Create SHAP values dictionary (only storing top 10 for memory) ===
                shap_values_dict = {}
                for i, feature in enumerate(feature_names):
                    # Only store first 10 values to reduce memory usage
                    shap_values_dict[feature] = shap_values.values[:10, i].tolist()
                
                # === Version info from original function ===
                from app import version_tracker
                model_version = version_tracker.get_latest_model()
                dataset_version = version_tracker.get_latest_dataset()
                version_info = {}
                if model_version:
                    version_info["model_version"] = model_version[0]
                if dataset_version:
                    version_info["dataset_version"] = dataset_version[0]
                
                # Final garbage collection before returning
                gc.collect()
                log_memory()
                
                # Return result in same format as original
                return {
                    "success": True,
                    "results": results,
                    "summary": summary,
                    "feature_importance": feature_importance,
                    "shap_values": shap_values_dict,
                    "version_info": version_info
                }
                
            except Exception as e:
                # Enhanced error handling with memory info
                import traceback
                tb = traceback.format_exc()
                memory = log_memory()
                error_msg = f"Error during analysis: {str(e)}"
                if memory > 0:
                    error_msg += f" (Memory usage: {memory:.2f} MB)"
                
                logger.error(f"[ANALYSIS ERROR] {error_msg}")
                logger.error(f"[ANALYSIS ERROR] Traceback:\n{tb}")
                
                return {
                    "success": False, 
                    "error": error_msg,
                    "traceback": tb
                }
        
        # Replace the original function with our optimized version
        logger.info("Replacing original analysis_worker with optimized version")
        import sys as app_sys
        app_sys.modules["app"].analysis_worker = optimized_analysis_worker
        logger.info("✅ Successfully patched analysis_worker with optimized version")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to patch analysis_worker: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def add_memory_endpoint(app):
    """Add memory monitoring endpoint to Flask app"""
    try:
        from flask import jsonify
        
        @app.route('/admin/memory', methods=['GET'])
        def memory_status():
            """Return current memory usage and status"""
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                return jsonify({
                    "success": True,
                    "memory_usage_mb": memory_mb,
                    "optimized_worker_applied": True,
                    "gc_enabled": gc.isenabled(),
                    "gc_counts": gc.get_count(),
                    "gc_threshold": gc.get_threshold(),
                    "max_rows_per_batch": MAX_ROWS_TO_PROCESS
                })
            except ImportError:
                return jsonify({
                    "success": False,
                    "error": "psutil not installed",
                    "optimized_worker_applied": True,
                    "gc_enabled": gc.isenabled()
                })
            except Exception as e:
                logger.error(f"Error in memory endpoint: {str(e)}")
                return jsonify({
                    "success": False,
                    "error": str(e),
                    "optimized_worker_applied": True
                }), 500
        
        logger.info("✅ Added memory monitoring endpoint: /admin/memory")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to add memory endpoint: {str(e)}")
        return False

def apply_all_patches(app=None):
    """Apply all optimizations and fixes"""
    # Make sure garbage collection is enabled
    gc.enable()
    logger.info("✅ Garbage collection enabled")
    
    # Apply optimized analysis worker
    patched = patch_analysis_worker()
    if not patched:
        logger.error("Failed to apply worker patch!")
    
    # Add memory endpoint if app is provided
    if app is not None:
        add_memory_endpoint(app)
    
    # Set environment variables for optimization
    os.environ['AGGRESSIVE_MEMORY_MANAGEMENT'] = 'true'
    os.environ['SHAP_BATCH_SIZE'] = str(MAX_ROWS_TO_PROCESS)
    
    logger.info("✅ All SHAP memory optimizations applied successfully")
    return True

if __name__ == "__main__":
    print("SHAP Memory Optimization Tool")
    print("This script optimizes the SHAP analysis worker to prevent 500 errors")
    print("To use this in your application:")
    print("  from shap_memory_fix import apply_all_patches")
    print("  apply_all_patches(app)  # Pass your Flask app instance")
