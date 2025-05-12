#!/usr/bin/env python3
"""
Comprehensive verification of SHAP with XGBoost models
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import os
import sys
import pickle
import traceback
from pathlib import Path

def verify_with_xgboost_model():
    """Test SHAP with an actual XGBoost model from the microservice"""
    print("Testing SHAP with actual XGBoost model...")
    
    # Path to the model
    model_path = Path("/Users/voldeck/code/shap-microservice/models/xgboost_model.pkl")
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Creating a simple test model instead.")
        
        # Create a simple dataset and model
        X = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100),
        })
        y = X['feature1'] * 2 + X['feature2'] + np.random.randn(100) * 0.1
        
        model = xgb.XGBRegressor(n_estimators=10)
        model.fit(X, y)
    else:
        # Load the existing model
        print(f"Loading model from {model_path}")
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded successfully")
            
            # Try to load feature names and generate some sample data
            feature_names_path = Path("/Users/voldeck/code/shap-microservice/models/feature_names.txt")
            if feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    feature_names = [line.strip() for line in f.readlines()]
                print(f"Loaded {len(feature_names)} feature names")
                
                # Generate sample data matching the features
                X = pd.DataFrame(np.random.rand(10, len(feature_names)), columns=feature_names)
            else:
                print("Feature names not found, using test data")
                X = pd.DataFrame(np.random.rand(10, 5), columns=[f'f{i}' for i in range(5)])
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    # Create SHAP explainer
    try:
        print("\nCreating SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        print("TreeExplainer created successfully!")
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer(X[:5])
        print("SHAP values calculated successfully!")
        print(f"SHAP values shape: {shap_values.values.shape}")
        
        # Calculate and print feature importance
        print("\nFeature importance:")
        feature_names = X.columns
        importance_vals = np.abs(shap_values.values).mean(0)
        
        for i, (name, importance) in enumerate(zip(feature_names, importance_vals)):
            print(f"{i+1}. {name}: {importance:.6f}")
            
        print("\nSHAP with XGBoost verification PASSED!")
        return True
    except Exception as e:
        print(f"\nError during SHAP calculation: {e}")
        traceback.print_exc()
        print("\nSHAP with XGBoost verification FAILED!")
        return False

if __name__ == "__main__":
    # Verify SHAP with XGBoost
    verify_with_xgboost_model()
