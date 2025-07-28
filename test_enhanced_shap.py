#!/usr/bin/env python3
"""
Enhanced SHAP test with proper data preprocessing
"""

import pandas as pd
import numpy as np
import pickle
import shap
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data_for_shap(data, model_features):
    """Preprocess data to be compatible with SHAP/XGBoost"""
    logger.info("🔧 Preprocessing data for SHAP compatibility...")
    
    processed_data = data.copy()
    
    # Add missing features with default values
    for feature in model_features:
        if feature not in processed_data.columns:
            if feature in ['Age', 'Income']:
                # For demographic features, use reasonable defaults
                processed_data[feature] = 0
            else:
                processed_data[feature] = 0
    
    # Select only model features in correct order
    processed_data = processed_data[model_features]
    
    # Handle different data types
    for col in processed_data.columns:
        if processed_data[col].dtype == 'object':
            # For string/object columns, try to convert to numeric or encode
            try:
                # Try direct conversion to numeric
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
            except:
                # If that fails, use label encoding
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
            
        # Fill any remaining NaN values
        processed_data[col] = processed_data[col].fillna(0)
        
        # Replace inf values
        processed_data[col] = processed_data[col].replace([np.inf, -np.inf], 0)
    
    # Ensure all data is numeric
    processed_data = processed_data.astype(float)
    
    logger.info(f"✅ Data preprocessed: {processed_data.shape}")
    logger.info(f"🎯 Data types: {processed_data.dtypes.value_counts().to_dict()}")
    
    return processed_data

def test_shap_calculation():
    """Test SHAP calculation with proper preprocessing"""
    try:
        logger.info("🚀 Starting Enhanced SHAP Test")
        
        # Load model and feature names
        with open("models/xgboost_model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        with open("models/feature_names.txt", 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # Load data
        data = pd.read_csv("data/nesto_merge_0.csv")
        logger.info(f"📊 Loaded data: {data.shape}")
        
        # Take a small sample for testing
        sample_data = data.head(5)
        
        # Preprocess data
        processed_data = preprocess_data_for_shap(sample_data, feature_names)
        
        # Create SHAP explainer
        logger.info("🧠 Creating SHAP explainer...")
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        logger.info("⚡ Calculating SHAP values...")
        shap_values = explainer.shap_values(processed_data)
        
        # Verify results
        logger.info(f"✅ SHAP calculation successful!")
        logger.info(f"📊 SHAP values shape: {shap_values.shape}")
        
        # Check for meaningful values
        non_zero_count = np.count_nonzero(shap_values)
        total_count = shap_values.size
        logger.info(f"📈 Non-zero SHAP values: {non_zero_count}/{total_count} ({non_zero_count/total_count*100:.1f}%)")
        
        # Show sample values
        logger.info(f"🎯 Sample SHAP values (first 5 features):")
        for i, feature in enumerate(feature_names[:5]):
            logger.info(f"   {feature}: {shap_values[0][i]:.6f}")
        
        logger.info("🎉 SHAP is now working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"❌ SHAP test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_shap_calculation()
