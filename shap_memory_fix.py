#!/usr/bin/env python3
"""
SHAP Memory Optimization

This module provides memory optimization for SHAP analysis,
allowing for the processing of larger datasets without running
out of memory.
"""

import os
import gc
import logging
import numpy as np
from typing import Union, Optional, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shap-memory-fix")

# Configuration for batch processing
MAX_ROWS_TO_PROCESS = int(os.environ.get('SHAP_MAX_BATCH_SIZE', '300'))

class ShapValuesWrapper:
    """Wrapper class to maintain compatibility with SHAP values API"""
    
    def __init__(self, values, base_values=None, data=None, feature_names=None):
        """Initialize with raw SHAP values"""
        self.values = values
        self.base_values = base_values
        self.data = data
        self.feature_names = feature_names
        
    def __len__(self):
        """Return the number of rows in the SHAP values"""
        if hasattr(self.values, '__len__'):
            return len(self.values)
        return 0

def apply_memory_patches(app=None):
    """
    Apply memory optimization patches to the Flask app
    
    Args:
        app: Flask application instance
    """
    import gc
    
    # Enable garbage collection to run more aggressively
    gc.enable()
    logger.info("Garbage collection enabled with thresholds: %s", gc.get_threshold())
    
    # Function to create memory-optimized explainer
    def create_memory_optimized_explainer(model, X, feature_names=None, 
                                         max_rows=MAX_ROWS_TO_PROCESS):
        """
        Creates a memory-optimized explainer by processing data in batches
        
        Args:
            model: The trained model to explain
            X: Input features to explain
            feature_names: Optional list of feature names
            max_rows: Maximum number of rows to process in one batch
            
        Returns:
            ShapValuesWrapper containing the computed SHAP values
        """
        import shap
        
        # Start with garbage collection to ensure we have maximum memory available
        gc.collect()
        
        logger.info(f"Creating memory-optimized explainer for {len(X)} rows")
        logger.info(f"Using max batch size of {max_rows} rows")
        
        # Check if the dataset is small enough to process in one go
        if len(X) <= max_rows:
            logger.info("Dataset small enough for direct processing")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X)
            
            # Return ShapValuesWrapper if needed
            if not hasattr(shap_values, 'values'):
                return ShapValuesWrapper(shap_values)
            return shap_values
        
        # Process in batches for large datasets
        logger.info(f"Processing large dataset in batches")
        all_shap_values = []
        total_rows = len(X)
        chunks = (total_rows + max_rows - 1) // max_rows  # Ceiling division
        
        for i in range(chunks):
            start_idx = i * max_rows
            end_idx = min((i + 1) * max_rows, total_rows)
            
            logger.info(f"Processing batch {i+1}/{chunks} (rows {start_idx}-{end_idx})")
            
            # Extract this chunk of data
            X_chunk = X.iloc[start_idx:end_idx]
            
            # Create explainer and get SHAP values for this chunk
            explainer = shap.TreeExplainer(model)
            chunk_shap_values = explainer(X_chunk)
            
            # Extract values (handle different return types from different SHAP versions)
            if hasattr(chunk_shap_values, 'values'):
                all_shap_values.append(chunk_shap_values.values)
            else:
                all_shap_values.append(chunk_shap_values)
                
            # Force cleanup to free memory
            del explainer
            del chunk_shap_values
            del X_chunk
            gc.collect()
            
        # Combine all chunks
        logger.info("Combining SHAP values from all batches")
        try:
            combined_values = np.vstack(all_shap_values)
            return ShapValuesWrapper(combined_values)
        except:
            logger.warning("Could not combine values using np.vstack, returning list")
            return ShapValuesWrapper(all_shap_values)
    
    # Patch the calculation function
    global calculate_shap_values
    
    # Store original function if it exists
    if 'calculate_shap_values' in globals():
        original_calculate_shap_values = calculate_shap_values
        
        # Define patched function with same signature
        def memory_optimized_calculate_shap_values(model, X, feature_names=None, **kwargs):
            """Memory optimized version of calculate_shap_values"""
            logger.info("Using memory-optimized SHAP calculation")
            return create_memory_optimized_explainer(model, X, feature_names, MAX_ROWS_TO_PROCESS)
        
        # Replace the global function
        calculate_shap_values = memory_optimized_calculate_shap_values
    else:
        # If function doesn't exist yet, create it
        def calculate_shap_values(model, X, feature_names=None, **kwargs):
            """Memory optimized SHAP calculation function"""
            logger.info("Using memory-optimized SHAP calculation")
            return create_memory_optimized_explainer(model, X, feature_names, MAX_ROWS_TO_PROCESS)
    
    # If app is provided, add memory monitoring endpoint
    if app is not None:
        logger.info("Adding memory monitoring endpoint to Flask app")
        try:
            from flask import jsonify
            import psutil
            
            @app.route('/admin/memory', methods=['GET'])
            def memory_status():
                """Return current memory usage and status"""
                try:
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
                except Exception as e:
                    logger.error(f"Error in memory endpoint: {str(e)}")
                    return jsonify({
                        "success": False,
                        "error": str(e)
                    }), 500
        except ImportError:
            logger.warning("Could not add memory endpoint: psutil not installed")
    
    logger.info("Memory optimization patches applied successfully")
    return True

# Export the memory optimization function
__all__ = ['apply_memory_patches', 'MAX_ROWS_TO_PROCESS', 'ShapValuesWrapper']

# If run directly, print info
if __name__ == "__main__":
    print("SHAP Memory Optimization Module")
    print(f"Maximum rows per batch: {MAX_ROWS_TO_PROCESS}")
    print("To use, import this module and call apply_memory_patches()")
