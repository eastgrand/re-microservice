#!/usr/bin/env python3
"""
Simple test to verify SHAP is working correctly
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import os
import sys

def test_shap():
    print("Testing SHAP functionality...")
    
    # Create a simple dataset
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100),
    })
    y = X['feature1'] * 2 + X['feature2'] + np.random.randn(100) * 0.1
    
    # Train a simple XGBoost model
    model = xgb.XGBRegressor(n_estimators=10)
    model.fit(X, y)
    
    # Try to create a SHAP explainer
    try:
        print("Creating SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        print("TreeExplainer created successfully!")
        
        # Try to calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer(X[:5])
        print("SHAP values calculated successfully!")
        print(f"SHAP values shape: {shap_values.values.shape}")
        
        # Check feature importance
        feature_importance = []
        for i, feature in enumerate(X.columns):
            importance = abs(shap_values.values[:, i]).mean()
            feature_importance.append({
                'feature': feature,
                'importance': float(importance)
            })
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        print("Feature importance calculated:")
        for feature in feature_importance:
            print(f"  {feature['feature']}: {feature['importance']}")
        
        print("\nSHAP test completed successfully!")
        return True
    except Exception as e:
        print(f"Error in SHAP test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_shap()
    print("\nTest completed.")
