#!/usr/bin/env python3
"""
Memory optimization utilities for the SHAP microservice.
This module provides functions to reduce memory usage during data processing and model training.
"""

import pandas as pd
import numpy as np
import os
import logging
import gc

# Try to import psutil, but provide fallback if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("memory-optimizer")

def get_memory_usage():
    """Get current process memory usage in megabytes."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        return memory_mb
    else:
        # Fallback method if psutil isn't available
        try:
            # Try using /proc/self/status on Linux systems
            if os.path.exists('/proc/self/status'):
                with open('/proc/self/status', 'r') as f:
                    for line in f:
                        if line.startswith('VmRSS:'):
                            return int(line.split()[1]) / 1024  # Convert KB to MB
            
            # If we can't get memory info, return a placeholder
            return 100  # Return placeholder value
        except:
            logger.warning("Could not determine memory usage")
            return 100  # Return placeholder value

def log_memory_usage(step_name):
    """Log the current memory usage."""
    memory_mb = get_memory_usage()
    logger.info(f"Memory usage at {step_name}: {memory_mb:.2f} MB")

def optimize_dtypes(df):
    """
    Optimize DataFrame memory usage by choosing the most memory-efficient data types.
    This can significantly reduce memory consumption for large DataFrames.
    """
    log_memory_usage("Before dtype optimization")
    start_mem = df.memory_usage().sum() / 1024**2
    logger.info(f"DataFrame memory usage before optimization: {start_mem:.2f} MB")
    
    # Process columns by dtype
    for col in df.columns:
        # Numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            # Integer columns
            if pd.api.types.is_integer_dtype(df[col]):
                min_val = df[col].min()
                max_val = df[col].max()
                
                # Choose smallest possible int type that can represent the data
                if min_val >= 0:  # unsigned
                    if max_val < 256:
                        df[col] = df[col].astype(np.uint8)
                    elif max_val < 65536:
                        df[col] = df[col].astype(np.uint16)
                    elif max_val < 4294967296:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:  # signed
                    if min_val > -128 and max_val < 128:
                        df[col] = df[col].astype(np.int8)
                    elif min_val > -32768 and max_val < 32768:
                        df[col] = df[col].astype(np.int16)
                    elif min_val > -2147483648 and max_val < 2147483648:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
                        
            # Float columns - downcast to float32 if possible
            elif pd.api.types.is_float_dtype(df[col]):
                # Check if all values are small enough to be represented as float32
                if np.all(np.abs(df[col].dropna()) < 1e38):
                    df[col] = df[col].astype(np.float32)
                    
        # Categorical or object columns - convert to categorical if few unique values
        elif pd.api.types.is_object_dtype(df[col]):
            unique_count = df[col].nunique()
            total_count = len(df[col])
            
            # If the column has relatively few unique values, convert to categorical
            if unique_count / total_count < 0.5 and unique_count < 100:
                df[col] = df[col].astype('category')
    
    # Force garbage collection
    gc.collect()
    
    # Log memory savings
    end_mem = df.memory_usage().sum() / 1024**2
    logger.info(f"DataFrame memory usage after optimization: {end_mem:.2f} MB")
    logger.info(f"Memory reduced by {100 * (start_mem - end_mem) / start_mem:.2f}%")
    log_memory_usage("After dtype optimization")
    
    return df

def load_and_optimize_data(file_path, low_memory=True, nrows=None):
    """
    Load a CSV file with optimized memory usage.
    
    Args:
        file_path: Path to the CSV file
        low_memory: Whether to use low_memory option
        nrows: Optional number of rows to load (to reduce memory usage)
        
    Returns:
        Optimized pandas DataFrame
    """
    log_memory_usage("Before data loading")
    
    # For extremely large files, read in chunks
    if nrows is not None:
        df = pd.read_csv(file_path, nrows=nrows, low_memory=low_memory)
        logger.info(f"Loaded {nrows} rows from {file_path}")
    else:
        try:
            # First attempt: try to infer data types from a sample
            # This helps pandas choose efficient types from the start
            sample = pd.read_csv(file_path, nrows=1000, low_memory=low_memory)
            dtypes = {col: sample[col].dtype for col in sample.columns}
            
            # Then load the full file with optimized dtypes
            df = pd.read_csv(file_path, dtype=dtypes, low_memory=low_memory)
            
            logger.info(f"Loaded full dataset from {file_path} with inferred types")
        except Exception as e:
            logger.warning(f"Error loading with dtype inference: {e}. Falling back to default loading.")
            df = pd.read_csv(file_path, low_memory=low_memory)
            logger.info(f"Loaded full dataset from {file_path}")
    
    log_memory_usage("After data loading")
    
    # Further optimize the data types
    df = optimize_dtypes(df)
    
    return df

def is_memory_critical(threshold_mb=450):
    """Check if memory usage is approaching the critical threshold."""
    # Get threshold from environment variable if available
    env_threshold = os.environ.get('MAX_MEMORY_MB')
    if env_threshold:
        try:
            threshold_mb = int(env_threshold)
            logger.info(f"Using environment-defined memory threshold: {threshold_mb} MB")
        except ValueError:
            logger.warning(f"Invalid MAX_MEMORY_MB value: {env_threshold}, using default {threshold_mb} MB")
    
    memory_mb = get_memory_usage()
    
    # If in Render's environment, be more aggressive with optimizations
    if 'RENDER' in os.environ:
        logger.info(f"Running in Render environment, assuming memory is limited")
        # In Render, consider memory critical at 80% of threshold
        return memory_mb > (threshold_mb * 0.8)
        
    return memory_mb > threshold_mb

def sample_data_if_needed(df, original_size, max_mb=None):
    """
    Sample the data if memory usage is too high.
    Returns the dataframe with potentially fewer rows.
    """
    # Get threshold from environment variable if available
    if max_mb is None:
        max_mb = int(os.environ.get('MAX_MEMORY_MB', 450))
    
    current_memory = get_memory_usage()
    
    if current_memory < max_mb:
        return df
    
    # Calculate how much we need to reduce
    reduction_factor = max_mb / current_memory
    target_size = int(len(df) * reduction_factor * 0.9)  # 10% safety margin
    
    logger.warning(f"Memory usage critical: {current_memory:.2f} MB. "
                  f"Reducing dataset from {len(df)} to {target_size} rows.")
    
    # Ensure we keep at least some data
    target_size = max(target_size, min(5000, len(df)))
    
    # Sample the data
    if target_size < len(df):
        df = df.sample(n=target_size, random_state=42)
        gc.collect()  # Force garbage collection
        log_memory_usage(f"After sampling to {target_size} rows")
    
    return df

def reduce_model_complexity(model_params, max_mb=None):
    """
    Adjust model parameters based on available memory.
    Returns updated model parameters.
    """
    # Get threshold from environment variable if available
    if max_mb is None:
        max_mb = int(os.environ.get('MAX_MEMORY_MB', 450))
    
    current_memory = get_memory_usage()
    
    # Only adjust if we're getting close to the limit
    if current_memory < max_mb * 0.8:
        return model_params.copy()
    
    logger.warning(f"Memory usage high ({current_memory:.2f} MB), reducing model complexity.")
    
    # Create a copy to modify
    adjusted_params = model_params.copy()
    
    # Reduce number of estimators
    if 'n_estimators' in adjusted_params and adjusted_params['n_estimators'] > 50:
        adjusted_params['n_estimators'] = 50
        logger.info(f"Reduced n_estimators to {adjusted_params['n_estimators']}")
    
    # Reduce max_depth
    if 'max_depth' in adjusted_params and adjusted_params['max_depth'] > 3:
        adjusted_params['max_depth'] = 3
        logger.info(f"Reduced max_depth to {adjusted_params['max_depth']}")
    
    return adjusted_params
