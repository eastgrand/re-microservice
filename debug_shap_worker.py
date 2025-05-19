#!/usr/bin/env python3
"""
SHAP Worker Process Debug Script

This script diagnoses issues with the SHAP analysis worker function directly
by executing the same code path that the worker would run, but with detailed logging
and error reporting to identify the exact cause of failures.
"""

import os
import sys
import time
import json
import logging
import traceback
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("shap-debug")

def log_memory_usage(label):
    """Log current memory usage"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"Memory usage at {label}: {memory_mb:.2f} MB")
    except ImportError:
        logger.warning("psutil not available, skipping memory logging")
    except Exception as e:
        logger.warning(f"Error logging memory: {str(e)}")

def test_shap_imports():
    """Test SHAP imports to ensure all dependencies are available"""
    logger.info("--- Testing SHAP imports ---")
    
    try:
        import shap
        logger.info(f"✅ Successfully imported shap version {shap.__version__}")
        
        # Test specific SHAP components
        logger.info("Testing TreeExplainer import...")
        from shap import TreeExplainer
        logger.info("✅ TreeExplainer imported successfully")
        
        # Test the numpy compatibility
        logger.info("Testing NumPy compatibility...")
        import numpy as np
        logger.info(f"✅ NumPy version: {np.__version__}")
        
        # Test xgboost
        logger.info("Testing XGBoost import...")
        import xgboost as xgb
        logger.info(f"✅ XGBoost version: {xgb.__version__}")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Import error: {str(e)}")
        logger.error(f"IMPORT TRACEBACK: {traceback.format_exc()}")
        return False
    except Exception as e:
        logger.error(f"❌ Error testing imports: {str(e)}")
        logger.error(f"ERROR TRACEBACK: {traceback.format_exc()}")
        return False

def load_model_direct():
    """Attempt to load model directly without using the app.py functions"""
    logger.info("--- Loading Model Directly ---")
    
    try:
        log_memory_usage("before model load")
        
        # Standard paths from app.py
        MODEL_PATH = "models/xgboost_model.pkl"
        FEATURE_NAMES_PATH = "models/feature_names.txt"
        DATASET_PATH = "data/cleaned_data.csv"
        
        # Check if files exist
        logger.info(f"Checking for model at {MODEL_PATH}...")
        if not os.path.exists(MODEL_PATH):
            logger.error(f"❌ Model file not found: {MODEL_PATH}")
            return None, None, None
        
        logger.info(f"Checking for feature names at {FEATURE_NAMES_PATH}...")
        if not os.path.exists(FEATURE_NAMES_PATH):
            logger.error(f"❌ Feature names file not found: {FEATURE_NAMES_PATH}")
            return None, None, None
            
        logger.info(f"Checking for dataset at {DATASET_PATH}...")
        uses_fallback = False
        if not os.path.exists(DATASET_PATH):
            logger.warning(f"⚠️ Dataset not found at {DATASET_PATH}, checking fallback...")
            if os.path.exists("data/cleaned_data.csv"):
                DATASET_PATH = "data/cleaned_data.csv"
                logger.info(f"Using fallback dataset: {DATASET_PATH}")
                uses_fallback = True
            else:
                logger.error(f"❌ Dataset not found at any expected location")
                return None, None, None
        
        # Load model
        logger.info("Loading model...")
        import pickle
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"✅ Model loaded: {type(model)}")
        
        # Load feature names
        logger.info("Loading feature names...")
        with open(FEATURE_NAMES_PATH, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        logger.info(f"✅ Feature names loaded: {len(feature_names)} features")
        
        # Load dataset (with memory optimization)
        logger.info("Loading dataset...")
        import pandas as pd
        dataset = pd.read_csv(DATASET_PATH, nrows=1000)  # Load fewer rows for testing
        logger.info(f"✅ Dataset loaded: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
        
        log_memory_usage("after model load")
        
        return model, dataset, feature_names
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        logger.error(f"ERROR TRACEBACK: {traceback.format_exc()}")
        return None, None, None

def run_test_analysis(model, dataset, feature_names):
    """Run a simple test analysis to check if SHAP works"""
    logger.info("--- Running Test Analysis ---")
    
    try:
        log_memory_usage("before analysis")
        
        # Create a simple test query
        test_query = {
            "analysis_type": "correlation",
            "target_variable": "Mortgage_Approvals",
            "demographic_filters": ["Income > 50000"]
        }
        
        logger.info(f"Test query: {test_query}")
        
        # Run analysis (similar to the analysis_worker function, but simplified)
        analysis_type = test_query.get('analysis_type', 'correlation')
        target_variable = test_query.get('target_variable')
        filters = test_query.get('demographic_filters', [])
        
        # Filter dataset
        filtered_data = dataset.copy()
        logger.info(f"Starting with {len(filtered_data)} records")
        
        for filter_item in filters:
            if isinstance(filter_item, str) and '>' in filter_item:
                feature, value = filter_item.split('>')
                feature = feature.strip()
                value = float(value.strip())
                
                if feature in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data[feature] > value]
                    logger.info(f"Applied filter {feature} > {value}: {len(filtered_data)} records remaining")
                else:
                    logger.warning(f"⚠️ Feature '{feature}' not found in dataset, skipping filter")
        
        # Prepare data for SHAP analysis
        X = filtered_data.copy()
        
        # Remove non-feature columns
        for col in ['zip_code', 'latitude', 'longitude']:
            if col in X.columns:
                X = X.drop(col, axis=1)
                
        if target_variable in X.columns:
            X = X.drop(target_variable, axis=1)
        
        # Align columns with model features
        X_cols = list(X.columns)
        for col in X_cols:
            if col not in feature_names:
                X = X.drop(col, axis=1)
                
        for feature in feature_names:
            if feature not in X.columns:
                X[feature] = 0
        
        X = X[feature_names]
        logger.info(f"Prepared features tensor shape: {X.shape}")
        
        # Run SHAP analysis
        logger.info("Creating TreeExplainer...")
        import shap
        explainer = shap.TreeExplainer(model)
        logger.info("✅ Created TreeExplainer")
        
        logger.info("Calculating SHAP values...")
        shap_values = explainer(X)
        logger.info(f"✅ Calculated SHAP values shape: {shap_values.values.shape}")
        
        # Calculate feature importance
        logger.info("Calculating feature importance...")
        feature_importance = []
        for i, feature in enumerate(feature_names):
            importance = abs(shap_values.values[:, i]).mean()
            feature_importance.append({
                'feature': feature, 
                'importance': float(importance)
            })
        
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        logger.info(f"Top 3 features: {feature_importance[:3]}")
        
        log_memory_usage("after analysis")
        
        # Force garbage collection
        gc.collect()
        log_memory_usage("after gc")
        
        return {
            "success": True,
            "top_features": feature_importance[:5],
            "data_shape": X.shape
        }
        
    except Exception as e:
        logger.error(f"❌ Error running analysis: {str(e)}")
        logger.error(f"ERROR TRACEBACK: {traceback.format_exc()}")
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

def main():
    """Main function to run all tests"""
    logger.info("=== SHAP Worker Debug Tool ===")
    
    # Test SHAP imports
    if not test_shap_imports():
        logger.error("❌ SHAP import tests failed, cannot proceed")
        return 1
        
    # Load model
    model, dataset, feature_names = load_model_direct()
    if model is None or dataset is None or feature_names is None:
        logger.error("❌ Model loading failed, cannot proceed")
        return 1
    
    # Run test analysis
    result = run_test_analysis(model, dataset, feature_names)
    
    if result["success"]:
        logger.info("✅ Test analysis completed successfully!")
        logger.info(f"Top features: {json.dumps(result['top_features'], indent=2)}")
    else:
        logger.error("❌ Test analysis failed!")
        
    # Print summary
    logger.info("\n=== DEBUG SUMMARY ===")
    if result["success"]:
        logger.info("✅ SHAP analysis works correctly")
        logger.info("The issue is likely not with SHAP computation itself.")
        logger.info("\nPossible remaining issues:")
        logger.info("1. Worker process is not running or is crashing")
        logger.info("2. Redis connection issues between app and worker")
        logger.info("3. Memory limits being exceeded in production environment")
    else:
        logger.info("❌ SHAP analysis failed")
        logger.info("This is likely why worker jobs are not completing.")
        logger.info("\nRecommended actions:")
        logger.info("1. Fix the specific error shown in the traceback")
        logger.info("2. Check memory usage during analysis")
        logger.info("3. Verify model and data are compatible")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
