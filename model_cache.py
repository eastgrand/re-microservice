import os
import pickle
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import xgboost as xgb
import shap

logger = logging.getLogger(__name__)

class ModelCache:
    """
    Caches fitted XGBoost models and SHAP explainers to avoid retraining on every request.
    Models are cached based on target variable and relevant feature combinations.
    """
    
    def __init__(self, cache_dir: str = "models/cache", max_cache_size: int = 10):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.cache_index_file = os.path.join(cache_dir, "cache_index.json")
        self.cache_index = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache index
        self._load_cache_index()
    
    def _load_cache_index(self):
        """Load the cache index from disk."""
        try:
            if os.path.exists(self.cache_index_file):
                with open(self.cache_index_file, 'r') as f:
                    self.cache_index = json.load(f)
                logger.info(f"Loaded cache index with {len(self.cache_index)} entries")
        except Exception as e:
            logger.error(f"Failed to load cache index: {e}")
            self.cache_index = {}
    
    def _save_cache_index(self):
        """Save the cache index to disk."""
        try:
            with open(self.cache_index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def _generate_cache_key(self, target_variable: str, features: list, analysis_type: str) -> str:
        """Generate a unique cache key for the model configuration."""
        # Sort features for consistent hashing
        sorted_features = sorted(features)
        
        # Create hash from target, features, and analysis type
        key_data = f"{target_variable}:{','.join(sorted_features)}:{analysis_type}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_model(self, target_variable: str, features: list, analysis_type: str) -> Optional[Tuple[Any, Any]]:
        """
        Retrieve a cached model and SHAP explainer if available.
        Returns (model, explainer) tuple or None if not cached.
        """
        cache_key = self._generate_cache_key(target_variable, features, analysis_type)
        
        if cache_key not in self.cache_index:
            return None
        
        cache_entry = self.cache_index[cache_key]
        model_path = os.path.join(self.cache_dir, f"{cache_key}_model.pkl")
        explainer_path = os.path.join(self.cache_dir, f"{cache_key}_explainer.pkl")
        
        # Check if files exist and are not expired
        if not os.path.exists(model_path) or not os.path.exists(explainer_path):
            # Remove invalid cache entry
            del self.cache_index[cache_key]
            self._save_cache_index()
            return None
        
        # Check expiration (models expire after 24 hours)
        created_time = datetime.fromisoformat(cache_entry['created_at'])
        if datetime.now() - created_time > timedelta(hours=24):
            logger.info(f"Cache entry {cache_key} expired, removing")
            self._remove_cache_entry(cache_key)
            return None
        
        try:
            # Load model and explainer
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            with open(explainer_path, 'rb') as f:
                explainer = pickle.load(f)
            
            # Update access time
            cache_entry['last_accessed'] = datetime.now().isoformat()
            cache_entry['access_count'] = cache_entry.get('access_count', 0) + 1
            self._save_cache_index()
            
            logger.info(f"Retrieved cached model for {target_variable} with {len(features)} features")
            return model, explainer
            
        except Exception as e:
            logger.error(f"Failed to load cached model {cache_key}: {e}")
            self._remove_cache_entry(cache_key)
            return None
    
    def cache_model(self, target_variable: str, features: list, analysis_type: str, 
                   model: Any, explainer: Any, training_metrics: Dict[str, Any] = None):
        """
        Cache a trained model and SHAP explainer.
        """
        cache_key = self._generate_cache_key(target_variable, features, analysis_type)
        
        # Ensure cache doesn't exceed max size
        self._enforce_cache_size_limit()
        
        model_path = os.path.join(self.cache_dir, f"{cache_key}_model.pkl")
        explainer_path = os.path.join(self.cache_dir, f"{cache_key}_explainer.pkl")
        
        try:
            # Save model and explainer
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            with open(explainer_path, 'wb') as f:
                pickle.dump(explainer, f)
            
            # Update cache index
            self.cache_index[cache_key] = {
                'target_variable': target_variable,
                'features': features,
                'analysis_type': analysis_type,
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'access_count': 1,
                'training_metrics': training_metrics or {},
                'model_size_mb': os.path.getsize(model_path) / (1024 * 1024),
                'explainer_size_mb': os.path.getsize(explainer_path) / (1024 * 1024)
            }
            
            self._save_cache_index()
            logger.info(f"Cached model for {target_variable} with {len(features)} features")
            
        except Exception as e:
            logger.error(f"Failed to cache model {cache_key}: {e}")
            # Clean up partial files
            for path in [model_path, explainer_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    def _enforce_cache_size_limit(self):
        """Remove oldest/least accessed cache entries if cache is full."""
        if len(self.cache_index) < self.max_cache_size:
            return
        
        # Sort by last accessed time and access count
        sorted_entries = sorted(
            self.cache_index.items(),
            key=lambda x: (x[1]['last_accessed'], x[1]['access_count'])
        )
        
        # Remove oldest entries
        entries_to_remove = len(self.cache_index) - self.max_cache_size + 1
        for i in range(entries_to_remove):
            cache_key, _ = sorted_entries[i]
            self._remove_cache_entry(cache_key)
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove a cache entry and its associated files."""
        if cache_key in self.cache_index:
            del self.cache_index[cache_key]
        
        # Remove files
        for suffix in ['_model.pkl', '_explainer.pkl']:
            file_path = os.path.join(self.cache_dir, f"{cache_key}{suffix}")
            if os.path.exists(file_path):
                os.remove(file_path)
        
        self._save_cache_index()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size_mb = 0
        total_access_count = 0
        
        for entry in self.cache_index.values():
            total_size_mb += entry.get('model_size_mb', 0) + entry.get('explainer_size_mb', 0)
            total_access_count += entry.get('access_count', 0)
        
        return {
            'total_entries': len(self.cache_index),
            'max_entries': self.max_cache_size,
            'total_size_mb': round(total_size_mb, 2),
            'total_access_count': total_access_count,
            'entries': [
                {
                    'target_variable': entry['target_variable'],
                    'feature_count': len(entry['features']),
                    'analysis_type': entry['analysis_type'],
                    'created_at': entry['created_at'],
                    'access_count': entry['access_count'],
                    'size_mb': round(entry.get('model_size_mb', 0) + entry.get('explainer_size_mb', 0), 2)
                }
                for entry in self.cache_index.values()
            ]
        }
    
    def clear_cache(self):
        """Clear all cached models."""
        for cache_key in list(self.cache_index.keys()):
            self._remove_cache_entry(cache_key)
        logger.info("Cleared all cached models")
    
    def clear_expired_cache(self):
        """Clear expired cache entries."""
        now = datetime.now()
        expired_keys = []
        
        for cache_key, entry in self.cache_index.items():
            created_time = datetime.fromisoformat(entry['created_at'])
            if now - created_time > timedelta(hours=24):
                expired_keys.append(cache_key)
        
        for cache_key in expired_keys:
            self._remove_cache_entry(cache_key)
        
        logger.info(f"Cleared {len(expired_keys)} expired cache entries")

# Global cache instance
model_cache = ModelCache()

def get_or_train_model(target_variable: str, features: list, analysis_type: str, 
                      X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, Any]:
    """
    Get a cached model or train a new one if not cached.
    Returns (model, explainer) tuple.
    """
    # Try to get from cache first
    cached_result = model_cache.get_cached_model(target_variable, features, analysis_type)
    if cached_result is not None:
        logger.info(f"Using cached model for {target_variable}")
        return cached_result
    
    # Train new model
    logger.info(f"Training new model for {target_variable} with {len(features)} features")
    
    # Configure XGBoost parameters for faster training
    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,  # Reduced for faster training
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1  # Use all available cores
    }
    
    # Train model
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate training metrics
    train_score = model.score(X_train, y_train)
    training_metrics = {
        'r2_score': train_score,
        'feature_count': len(features),
        'training_samples': len(X_train)
    }
    
    # Cache the model
    model_cache.cache_model(target_variable, features, analysis_type, model, explainer, training_metrics)
    
    return model, explainer 