"""
Memory management utilities for SHAP microservice
Provides streaming batch processing and memory monitoring to prevent OOM errors
"""

import gc
import psutil
import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0.0

def get_available_memory() -> float:
    """Get available system memory in MB"""
    try:
        return psutil.virtual_memory().available / 1024 / 1024
    except:
        return 1000.0  # Default fallback

def force_garbage_collection():
    """Force garbage collection and return memory freed in MB"""
    before = get_memory_usage()
    gc.collect()
    after = get_memory_usage()
    freed = before - after
    logger.info(f"Garbage collection freed {freed:.1f}MB")
    return freed

def calculate_optimal_batch_size(total_samples: int, max_memory_mb: float = 800) -> int:
    """Calculate optimal batch size based on available memory and sample count"""
    available_memory = get_available_memory()
    current_memory = get_memory_usage()
    
    # Conservative estimate: 10MB per sample for SHAP calculations
    memory_per_sample = 10
    safe_memory_limit = min(max_memory_mb, available_memory * 0.7)
    
    # Calculate batch size that won't exceed memory limit
    max_batch_size = int(safe_memory_limit / memory_per_sample)
    
    # Ensure minimum batch size of 10, maximum of 200
    batch_size = max(10, min(max_batch_size, 200))
    
    # Don't use batches larger than 1/4 of total samples
    if total_samples > 40:
        batch_size = min(batch_size, total_samples // 4)
    
    logger.info(f"Calculated batch size: {batch_size} for {total_samples} samples "
                f"(Available memory: {available_memory:.1f}MB, Current: {current_memory:.1f}MB)")
    
    return batch_size

def batch_shap_calculation(explainer, X: pd.DataFrame, 
                          batch_size: Optional[int] = None,
                          max_memory_mb: float = 800,
                          progress_callback=None) -> np.ndarray:
    """
    Calculate SHAP values in memory-safe batches
    
    Args:
        explainer: SHAP explainer object
        X: Input data for SHAP calculation
        batch_size: Batch size (auto-calculated if None)
        max_memory_mb: Maximum memory to use
        progress_callback: Optional callback for progress updates
    
    Returns:
        numpy array of SHAP values
    """
    total_samples = len(X)
    
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(total_samples, max_memory_mb)
    
    logger.info(f"Starting batch SHAP calculation: {total_samples} samples, "
                f"batch size {batch_size}")
    
    results = []
    
    for i in range(0, total_samples, batch_size):
        start_mem = get_memory_usage()
        
        # Force garbage collection if memory is getting high
        if start_mem > max_memory_mb * 0.8:
            force_garbage_collection()
        
        # Get batch
        end_idx = min(i + batch_size, total_samples)
        batch = X.iloc[i:end_idx]
        
        try:
            # Calculate SHAP values for batch
            batch_shap = explainer.shap_values(batch)
            
            # Convert to list if single output
            if not isinstance(batch_shap, list):
                batch_shap = [batch_shap]
            
            results.append(batch_shap[0])  # Take first output for binary classification
            
            # Clean up immediately
            del batch_shap
            del batch
            
        except Exception as e:
            logger.error(f"Error in batch {i}-{end_idx}: {str(e)}")
            # Try with smaller batch size
            if batch_size > 10:
                smaller_batch_size = max(10, batch_size // 2)
                logger.info(f"Retrying with smaller batch size: {smaller_batch_size}")
                
                # Recursively process this batch with smaller size
                sub_batch_shap = batch_shap_calculation(
                    explainer, batch, smaller_batch_size, max_memory_mb
                )
                results.append(sub_batch_shap)
            else:
                raise e
        
        # Force cleanup after each batch
        force_garbage_collection()
        
        # Progress callback
        if progress_callback:
            progress = (end_idx / total_samples) * 100
            progress_callback(progress)
        
        end_mem = get_memory_usage()
        logger.info(f"Batch {i//batch_size + 1}/{(total_samples-1)//batch_size + 1} complete. "
                   f"Memory: {start_mem:.1f}MB → {end_mem:.1f}MB")
    
    # Combine all results
    if not results:
        raise ValueError("No SHAP values calculated - all batches failed")
    
    combined_shap = np.concatenate(results, axis=0)
    
    # Final cleanup
    del results
    force_garbage_collection()
    
    logger.info(f"Batch SHAP calculation complete. Final shape: {combined_shap.shape}")
    return combined_shap

def memory_safe_sample_selection(df: pd.DataFrame, 
                                target_field: str,
                                max_samples: int,
                                strategy: str = 'balanced') -> pd.DataFrame:
    """
    Select samples in a memory-safe way for SHAP analysis
    
    Args:
        df: Input dataframe
        target_field: Target variable field name
        max_samples: Maximum number of samples to select
        strategy: Sampling strategy ('balanced', 'random', 'extremes')
    
    Returns:
        Sampled dataframe
    """
    if len(df) <= max_samples:
        return df
    
    logger.info(f"Sampling {max_samples} from {len(df)} records using '{strategy}' strategy")
    
    if strategy == 'balanced':
        # Sample equally from different quantiles of target variable
        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        samples_per_quantile = max_samples // (len(quantiles) - 1)
        
        sampled_dfs = []
        for i in range(len(quantiles) - 1):
            q_low = df[target_field].quantile(quantiles[i])
            q_high = df[target_field].quantile(quantiles[i + 1])
            
            mask = (df[target_field] >= q_low) & (df[target_field] <= q_high)
            quantile_df = df[mask]
            
            if len(quantile_df) > 0:
                n_samples = min(samples_per_quantile, len(quantile_df))
                sampled = quantile_df.sample(n=n_samples, random_state=42)
                sampled_dfs.append(sampled)
        
        result = pd.concat(sampled_dfs, ignore_index=True)
        
    elif strategy == 'extremes':
        # Focus on outliers and extreme values
        q_low = df[target_field].quantile(0.05)
        q_high = df[target_field].quantile(0.95)
        
        low_extremes = df[df[target_field] <= q_low]
        high_extremes = df[df[target_field] >= q_high]
        middle = df[(df[target_field] > q_low) & (df[target_field] < q_high)]
        
        # 40% low extremes, 40% high extremes, 20% middle
        n_low = min(int(max_samples * 0.4), len(low_extremes))
        n_high = min(int(max_samples * 0.4), len(high_extremes))
        n_middle = max_samples - n_low - n_high
        
        sampled_parts = []
        if n_low > 0:
            sampled_parts.append(low_extremes.sample(n=n_low, random_state=42))
        if n_high > 0:
            sampled_parts.append(high_extremes.sample(n=n_high, random_state=42))
        if n_middle > 0 and len(middle) > 0:
            sampled_parts.append(middle.sample(n=min(n_middle, len(middle)), random_state=42))
        
        result = pd.concat(sampled_parts, ignore_index=True)
        
    else:  # random
        result = df.sample(n=max_samples, random_state=42)
    
    logger.info(f"Selected {len(result)} samples for analysis")
    return result

# Endpoint-specific configurations
ENDPOINT_MEMORY_CONFIGS = {
    '/analyze': {
        'max_samples': 500,
        'batch_size': 100,
        'memory_limit_mb': 800,
        'sampling_strategy': 'random'
    },
    '/outlier-detection': {
        'max_samples': 200,
        'batch_size': 50,
        'memory_limit_mb': 600,
        'sampling_strategy': 'extremes'
    },
    '/scenario-analysis': {
        'max_samples': 100,
        'batch_size': 25,
        'memory_limit_mb': 500,
        'sampling_strategy': 'balanced'
    },
    '/spatial-clusters': {
        'max_samples': 300,
        'batch_size': 75,
        'memory_limit_mb': 700,
        'sampling_strategy': 'balanced'
    },
    '/segment-profiling': {
        'max_samples': 150,
        'batch_size': 50,
        'memory_limit_mb': 600,
        'sampling_strategy': 'balanced'
    },
    '/comparative-analysis': {
        'max_samples': 200,
        'batch_size': 50,
        'memory_limit_mb': 600,
        'sampling_strategy': 'balanced'
    },
    '/feature-interactions': {
        'max_samples': 800,
        'batch_size': 150,
        'memory_limit_mb': 900,
        'sampling_strategy': 'random'
    }
}

def get_endpoint_config(endpoint_path: str) -> dict:
    """Get memory configuration for specific endpoint"""
    return ENDPOINT_MEMORY_CONFIGS.get(endpoint_path, {
        'max_samples': 100,
        'batch_size': 25,
        'memory_limit_mb': 500,
        'sampling_strategy': 'random'
    })

def memory_safe_shap_wrapper(explainer, df: pd.DataFrame, 
                           target_field: str, endpoint_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Complete memory-safe wrapper for SHAP calculations
    
    Returns:
        Tuple of (shap_values, sampled_dataframe)
    """
    config = get_endpoint_config(endpoint_path)
    
    # Sample data if needed
    sampled_df = memory_safe_sample_selection(
        df, target_field, 
        config['max_samples'], 
        config['sampling_strategy']
    )
    
    # Prepare features (exclude target)
    feature_cols = [col for col in sampled_df.columns if col != target_field]
    X = sampled_df[feature_cols]
    
    # Calculate SHAP values in batches
    shap_values = batch_shap_calculation(
        explainer, X,
        batch_size=config['batch_size'],
        max_memory_mb=config['memory_limit_mb']
    )
    
    return shap_values, sampled_df 