#!/usr/bin/env python3
"""
Debug script to identify the exact feature mismatch between model and feature names
"""

import os
import pickle
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Debug the feature mismatch"""
    logger.info("🚀 Starting Feature Mismatch Debug")
    logger.info("=" * 60)
    
    # Load model and extract features from booster
    logger.info("=== Loading Model Features ===")
    with open("models/xgboost_model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    booster = model.get_booster()
    model_features = booster.feature_names
    logger.info(f"✅ Model features: {len(model_features)}")
    
    # Load feature names from file
    logger.info("=== Loading File Features ===")
    with open("models/feature_names.txt", 'r') as f:
        file_features = [line.strip() for line in f.readlines()]
    logger.info(f"✅ File features: {len(file_features)}")
    
    # Find the differences
    logger.info("=== Analyzing Differences ===")
    model_set = set(model_features)
    file_set = set(file_features)
    
    # Features in model but not in file
    missing_from_file = model_set - file_set
    logger.info(f"🔍 Features in model but NOT in file ({len(missing_from_file)}):")
    for feature in sorted(missing_from_file):
        logger.info(f"   - {feature}")
    
    # Features in file but not in model
    extra_in_file = file_set - model_set
    logger.info(f"🔍 Features in file but NOT in model ({len(extra_in_file)}):")
    for feature in sorted(extra_in_file):
        logger.info(f"   - {feature}")
    
    # Check order differences
    logger.info("=== Checking Feature Order ===")
    if len(model_features) == len(file_features):
        order_mismatches = []
        for i, (model_feat, file_feat) in enumerate(zip(model_features, file_features)):
            if model_feat != file_feat:
                order_mismatches.append((i, model_feat, file_feat))
        
        if order_mismatches:
            logger.warning(f"⚠️ Feature order mismatches found ({len(order_mismatches)}):")
            for i, model_feat, file_feat in order_mismatches[:10]:  # Show first 10
                logger.warning(f"   Position {i}: Model='{model_feat}' vs File='{file_feat}'")
        else:
            logger.info("✅ Feature order matches perfectly")
    
    # Load data and check which features are available
    logger.info("=== Checking Data Availability ===")
    data = pd.read_csv("data/nesto_merge_0.csv")
    data_columns = set(data.columns)
    logger.info(f"✅ Data columns: {len(data_columns)}")
    
    # Check model features against data
    model_missing_from_data = model_set - data_columns
    logger.info(f"🔍 Model features missing from data ({len(model_missing_from_data)}):")
    for feature in sorted(model_missing_from_data):
        logger.info(f"   - {feature}")
    
    # Check file features against data
    file_missing_from_data = file_set - data_columns
    logger.info(f"🔍 File features missing from data ({len(file_missing_from_data)}):")
    for feature in sorted(file_missing_from_data):
        logger.info(f"   - {feature}")
    
    # Generate corrected feature names file
    logger.info("=== Generating Solutions ===")
    
    if missing_from_file:
        logger.info("🔧 Solution 1: Update feature_names.txt to match model exactly")
        corrected_features = model_features.copy()
        
        # Write corrected feature names
        with open("models/feature_names_corrected.txt", 'w') as f:
            for feature in corrected_features:
                f.write(f"{feature}\n")
        logger.info(f"✅ Written corrected feature names to models/feature_names_corrected.txt")
        
        # Show what needs to be added
        logger.info("📝 Features that need to be added to feature_names.txt:")
        for feature in sorted(missing_from_file):
            logger.info(f"   + {feature}")
    
    if extra_in_file:
        logger.info("🔧 Solution 2: Remove extra features from feature_names.txt")
        logger.info("📝 Features that should be removed from feature_names.txt:")
        for feature in sorted(extra_in_file):
            logger.info(f"   - {feature}")
    
    # Test SHAP with corrected features
    logger.info("=== Testing SHAP with Corrected Features ===")
    try:
        import shap
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        logger.info("✅ SHAP explainer created successfully")
        
        # Prepare test data with model features
        sample_data = data.head(2).copy()
        
        # Add missing features with 0
        for feature in model_features:
            if feature not in sample_data.columns:
                sample_data[feature] = 0
        
        # Select only model features in correct order
        model_data = sample_data[model_features].fillna(0)
        model_data = model_data.replace([np.inf, -np.inf], 0)
        
        logger.info(f"🎯 Test data shape: {model_data.shape}")
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(model_data)
        logger.info(f"🎉 SHAP calculation successful! Shape: {shap_values.shape}")
        
        # Check for meaningful values
        non_zero_count = np.count_nonzero(shap_values)
        total_count = shap_values.size
        logger.info(f"📊 Non-zero SHAP values: {non_zero_count}/{total_count} ({non_zero_count/total_count*100:.1f}%)")
        
        logger.info("✅ SHAP is working correctly with corrected features!")
        
    except Exception as e:
        logger.error(f"❌ SHAP test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("=" * 60)
    logger.info("🏁 Feature mismatch debug complete")

if __name__ == "__main__":
    main()