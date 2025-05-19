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
import time

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

# Check for aggressive memory management flag
AGGRESSIVE_MEMORY = os.environ.get('AGGRESSIVE_MEMORY_MANAGEMENT', 'false').lower() == 'true'
if AGGRESSIVE_MEMORY:
    logger.info("AGGRESSIVE memory optimization mode is ENABLED")
    # More aggressive thresholds for Render to start optimizing earlier
    DEFAULT_MAX_MEMORY_MB = 450  # Increased from 400 to allow more data processing
else:
    # Standard memory optimization thresholds
    DEFAULT_MAX_MEMORY_MB = 475  # Increased from 450 to utilize more memory safely

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

def fix_categorical_columns(df):
    """Convert categorical columns to string to avoid serialization issues."""
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            logger.info(f"Converting categorical column to string: {col}")
            df[col] = df[col].astype(str)
    return df

def load_and_optimize_data(file_path, nrows=None, **kwargs):
    """
    Load a CSV file with optimized memory usage and handling for categorical types.
    """
    # First pass to get column dtypes
    dtypes = {}
    try:
        # Sample a few rows to determine types
        sample = pd.read_csv(file_path, nrows=100)
        
        # Convert categorical columns to string
        for col in sample.columns:
            if pd.api.types.is_categorical_dtype(sample[col]):
                dtypes[col] = str
        
        log_memory_usage("After sampling for dtypes")
    except Exception as e:
        logger.warning(f"Error during dtype detection: {e}")
        
    # Read the actual data
    try:
        if nrows:
            df = pd.read_csv(file_path, nrows=nrows, dtype=dtypes, **kwargs)
        else:
            df = pd.read_csv(file_path, dtype=dtypes, **kwargs)
        
        # Fix any remaining categorical columns
        df = fix_categorical_columns(df)
        
        log_memory_usage(f"After loading {file_path} with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Fall back to basic read
        logger.info(f"Falling back to basic read_csv for {file_path}")
        if nrows:
            return pd.read_csv(file_path, nrows=nrows)
        else:
            return pd.read_csv(file_path)

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
    
    # For Render, be more aggressive with memory management
    is_render = 'RENDER' in os.environ
    memory_threshold = max_mb * (0.5 if is_render else 0.9)
    
    if current_memory < memory_threshold:
        return df
    
    # Calculate reduction needed
    # More aggressive for Render environment
    safety_margin = 0.6 if is_render else 0.9
    reduction_factor = memory_threshold / current_memory
    target_size = int(len(df) * reduction_factor * safety_margin)
    
    logger.warning(f"Memory usage critical: {current_memory:.2f} MB. "
                  f"Reducing dataset from {len(df)} to {target_size} rows.")
    
    # Ensure we keep enough data, but be aggressive on Render
    min_rows = 3000 if is_render else 5000
    target_size = max(target_size, min(min_rows, len(df)))
    
    # Sample the data - use stratified sampling if possible for better representation
    if target_size < len(df):
        try:
            # Try stratified sampling on a categorical column if available
            cat_cols = [col for col in df.columns if pd.api.types.is_categorical_dtype(df[col])]
            if cat_cols and len(cat_cols) > 0:
                # Use the first categorical column with not too many unique values
                for col in cat_cols:
                    if df[col].nunique() <= 50:
                        logger.info(f"Using stratified sampling on column {col}")
                        # Calculate fraction needed to get target_size
                        frac = target_size / len(df)
                        df = df.groupby(col, group_keys=False).apply(
                            lambda x: x.sample(frac=frac, random_state=42)
                        )
                        break
                else:
                    # If no good stratification column found, do regular sampling
                    df = df.sample(n=target_size, random_state=42)
            else:
                # Regular random sampling
                df = df.sample(n=target_size, random_state=42)
                
        except Exception as e:
            logger.warning(f"Error in stratified sampling: {e}. Using random sampling.")
            df = df.sample(n=target_size, random_state=42)
            
        # Force garbage collection multiple times
        for _ in range(3):  # Multiple GC calls can help recover more memory
            gc.collect()
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
    is_render = 'RENDER' in os.environ
    
    # For Render, be more aggressive - start optimizing at 60% of max memory
    memory_threshold = max_mb * (0.6 if is_render else 0.8)
    
    # Only adjust if we're getting close to the limit
    if current_memory < memory_threshold:
        return model_params.copy()
    
    logger.warning(f"Memory usage high ({current_memory:.2f} MB), reducing model complexity.")
    
    # Create a copy to modify
    adjusted_params = model_params.copy()
    
    # Set level of complexity reduction based on memory pressure
    memory_ratio = current_memory / max_mb
    
    # Extreme memory pressure - use minimal model  
    if memory_ratio > 0.85:
        # Use histogram-based training for much lower memory usage
        adjusted_params['tree_method'] = 'hist'
        
        # Minimize estimators
        if 'n_estimators' in adjusted_params:
            adjusted_params['n_estimators'] = 30
            logger.info(f"Severely reduced n_estimators to {adjusted_params['n_estimators']}")
        
        # Minimize tree depth
        if 'max_depth' in adjusted_params:
            adjusted_params['max_depth'] = 2
            logger.info(f"Severely reduced max_depth to {adjusted_params['max_depth']}")
            
        # Add leaf-wise growth for smaller trees
        adjusted_params['grow_policy'] = 'lossguide'
        
        # Force more aggressive pruning
        adjusted_params['gamma'] = 1.0
        
    # High memory pressure - use reduced model
    elif memory_ratio > 0.7:
        # Use histogram-based training
        adjusted_params['tree_method'] = 'hist'
        
        # Reduce estimators
        if 'n_estimators' in adjusted_params and adjusted_params['n_estimators'] > 50:
            adjusted_params['n_estimators'] = 50
            logger.info(f"Reduced n_estimators to {adjusted_params['n_estimators']}")
        
        # Reduce tree depth
        if 'max_depth' in adjusted_params and adjusted_params['max_depth'] > 3:
            adjusted_params['max_depth'] = 3
            logger.info(f"Reduced max_depth to {adjusted_params['max_depth']}")
            
        # Add moderate pruning
        adjusted_params['gamma'] = 0.5
    
    # Apply additional memory-saving settings for all cases when needed
    if memory_ratio > 0.6:
        # Reduce complexity of individual trees
        adjusted_params['min_child_weight'] = 5
        
        # Skip some data for training (subsample data)
        adjusted_params['subsample'] = 0.8
        adjusted_params['colsample_bytree'] = 0.8
        
        # Reduce precision to save memory
        adjusted_params['single_precision_histogram'] = True
        
    return adjusted_params


def prune_dataframe_columns(df, target_column=None, max_columns=None, importance_threshold=0.01):
    """
    Reduce DataFrame memory usage by removing only specific legacy fields.
    Preserves all important analytical fields.
    
    Args:
        df: The DataFrame to optimize
        target_column: The target column to keep (won't be removed)
        max_columns: Maximum number of columns to keep (ignored in this implementation)
        importance_threshold: Minimum correlation threshold (ignored in this implementation)
        
    Returns:
        DataFrame with legacy fields removed
    """
    log_memory_usage("Before column pruning")
    original_columns = df.columns.tolist()
    
    # Legacy fields to remove as specified by the user
    legacy_fields = [
        'Single_Status',       # SUM_ECYMARNMCL
        'Single_Family_Homes', # SUM_ECYSTYSING  
        'Married_Population',  # SUM_ECYMARM
        'Aggregate_Income',    # SUM_HSHNIAGG
        'Market_Weight'        # Sum_Weight
    ]
    
    # Identify which legacy fields are actually present
    found_legacy_fields = [field for field in legacy_fields if field in df.columns]
    
    # Keep all columns except legacy fields
    keep_columns = [col for col in df.columns if col not in legacy_fields]
    
    # Keep only non-legacy columns
    df_pruned = df[keep_columns].copy()
    
    # Log results
    columns_removed = len(original_columns) - len(keep_columns)
    logger.info(f"Pruned {columns_removed} legacy columns to save memory. Kept {len(keep_columns)} columns.")
    
    # Log which legacy fields were found and removed
    if found_legacy_fields:
        logger.info(f"Removed legacy fields: {found_legacy_fields}")
    else:
        logger.info("No legacy fields found in the dataset.")
    
    # Force garbage collection
    gc.collect()
    time.sleep(0.1)  # Short delay to let memory be released
    gc.collect()  # Second collection sometimes helps more
    
    log_memory_usage("After column pruning")
    return df_pruned
