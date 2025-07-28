#!/usr/bin/env python3
"""
Debug script to test SHAP initialization locally
This will help identify what's causing the SHAP calculation to fail
"""

import os
import sys
import logging
import traceback
import pickle
import pandas as pd
import numpy as np
import shap
import xgboost as xgb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if model loads correctly"""
    logger.info("=== Testing Model Loading ===")
    
    model_path = "models/xgboost_model.pkl"
    feature_names_path = "models/feature_names.txt"
    
    try:
        # Load model
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"✅ Model loaded successfully: {type(model)}")
            logger.info(f"📊 Model info: {model}")
            
            # Check if it's XGBoost model
            if hasattr(model, 'get_params'):
                logger.info(f"🔧 Model params: {model.get_params()}")
            if hasattr(model, 'n_features_in_'):
                logger.info(f"🎯 Model expects {model.n_features_in_} features")
        else:
            logger.error(f"❌ Model file not found: {model_path}")
            return None, None
        
        # Load feature names
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            logger.info(f"✅ Feature names loaded: {len(feature_names)} features")
            logger.info(f"📋 First 5 features: {feature_names[:5]}")
            logger.info(f"📋 Last 5 features: {feature_names[-5:]}")
        else:
            logger.error(f"❌ Feature names file not found: {feature_names_path}")
            return model, None
            
        return model, feature_names
        
    except Exception as e:
        logger.error(f"❌ Error loading model/features: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def test_data_loading():
    """Test if training data loads correctly"""
    logger.info("=== Testing Data Loading ===")
    
    data_path = "data/nesto_merge_0.csv"
    
    try:
        if os.path.exists(data_path):
            # Load first few rows to check structure
            df = pd.read_csv(data_path, nrows=10)
            logger.info(f"✅ Data loaded successfully: {df.shape}")
            logger.info(f"📊 Columns ({len(df.columns)}): {list(df.columns)[:10]}...")
            logger.info(f"🎯 Sample data types: {df.dtypes.head()}")
            
            # Load full data for further testing
            full_df = pd.read_csv(data_path)
            logger.info(f"📈 Full dataset shape: {full_df.shape}")
            
            return full_df
        else:
            logger.error(f"❌ Data file not found: {data_path}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error loading data: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def test_feature_alignment(model, feature_names, data):
    """Test if model features align with data"""
    logger.info("=== Testing Feature Alignment ===")
    
    if model is None or feature_names is None or data is None:
        logger.error("❌ Missing model, features, or data - cannot test alignment")
        return False
    
    try:
        # Check if model expects specific number of features
        expected_features = len(feature_names)
        if hasattr(model, 'n_features_in_'):
            model_expected = model.n_features_in_
            logger.info(f"🎯 Model expects {model_expected} features, we have {expected_features}")
            if model_expected != expected_features:
                logger.warning(f"⚠️ Feature count mismatch! Model: {model_expected}, Features: {expected_features}")
        
        # Check if features exist in data
        data_columns = set(data.columns)
        missing_features = []
        available_features = []
        
        for feature in feature_names:
            if feature in data_columns:
                available_features.append(feature)
            else:
                missing_features.append(feature)
        
        logger.info(f"✅ Available features: {len(available_features)}/{len(feature_names)}")
        
        if missing_features:
            logger.warning(f"⚠️ Missing features ({len(missing_features)}): {missing_features[:10]}...")
            if len(missing_features) > 100:
                logger.error(f"❌ Too many missing features ({len(missing_features)}) - this will cause SHAP to fail")
                return False
        
        # Try to prepare a small sample for model prediction
        logger.info("🧪 Testing model prediction with sample data...")
        sample_data = data.head(5).copy()
        
        # Fill missing features with 0
        for feature in feature_names:
            if feature not in sample_data.columns:
                sample_data[feature] = 0
        
        # Select only model features in correct order
        model_data = sample_data[feature_names].fillna(0)
        model_data = model_data.replace([np.inf, -np.inf], 0)
        
        logger.info(f"🎯 Model input shape: {model_data.shape}")
        logger.info(f"🎯 Model input sample:\n{model_data.head(2)}")
        
        # Test prediction
        predictions = model.predict(model_data)
        logger.info(f"✅ Model prediction successful: {predictions[:3]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature alignment test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_shap_initialization(model, feature_names, data):
    """Test SHAP explainer initialization"""
    logger.info("=== Testing SHAP Initialization ===")
    
    if model is None or feature_names is None or data is None:
        logger.error("❌ Missing model, features, or data - cannot test SHAP")
        return None
    
    try:
        logger.info("🧠 Creating SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        logger.info(f"✅ SHAP explainer created successfully: {type(explainer)}")
        
        # Test with small sample
        logger.info("🧪 Testing SHAP calculation with 2 samples...")
        sample_data = data.head(2).copy()
        
        # Prepare features
        for feature in feature_names:
            if feature not in sample_data.columns:
                sample_data[feature] = 0
        
        model_data = sample_data[feature_names].fillna(0)
        model_data = model_data.replace([np.inf, -np.inf], 0)
        
        logger.info(f"🎯 SHAP input shape: {model_data.shape}")
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(model_data)
        logger.info(f"✅ SHAP calculation successful!")
        logger.info(f"📊 SHAP values shape: {shap_values.shape}")
        logger.info(f"🎯 Sample SHAP values: {shap_values[0][:5]}")
        
        # Check for non-zero values
        non_zero_count = np.count_nonzero(shap_values)
        total_values = shap_values.size
        logger.info(f"📈 Non-zero SHAP values: {non_zero_count}/{total_values} ({non_zero_count/total_values*100:.1f}%)")
        
        return explainer
        
    except Exception as e:
        logger.error(f"❌ SHAP initialization failed: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def test_shap_versions():
    """Test SHAP and XGBoost versions"""
    logger.info("=== Testing Package Versions ===")
    
    try:
        import shap
        import xgboost
        import sklearn
        import pandas
        import numpy
        
        logger.info(f"📦 SHAP version: {shap.__version__}")
        logger.info(f"📦 XGBoost version: {xgboost.__version__}")
        logger.info(f"📦 Scikit-learn version: {sklearn.__version__}")
        logger.info(f"📦 Pandas version: {pandas.__version__}")
        logger.info(f"📦 NumPy version: {numpy.__version__}")
        
    except Exception as e:
        logger.error(f"❌ Error checking versions: {str(e)}")

def main():
    """Run all debug tests"""
    logger.info("🚀 Starting SHAP Microservice Debug Tests")
    logger.info("=" * 60)
    
    # Test package versions
    test_shap_versions()
    logger.info("")
    
    # Test model loading
    model, feature_names = test_model_loading()
    logger.info("")
    
    # Test data loading
    data = test_data_loading()
    logger.info("")
    
    # Test feature alignment
    alignment_ok = test_feature_alignment(model, feature_names, data)
    logger.info("")
    
    # Test SHAP initialization
    if alignment_ok:
        explainer = test_shap_initialization(model, feature_names, data)
        logger.info("")
        
        if explainer:
            logger.info("🎉 All tests passed! SHAP should work correctly.")
        else:
            logger.error("❌ SHAP initialization failed - this is the root cause!")
    else:
        logger.error("❌ Feature alignment failed - SHAP cannot work with misaligned features!")
    
    logger.info("=" * 60)
    logger.info("✅ Debug tests complete")

if __name__ == "__main__":
    main()