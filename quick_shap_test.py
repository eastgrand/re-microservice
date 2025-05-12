#!/usr/bin/env python3
"""
Minimal test script to verify SHAP is working properly after patching
"""
import shap
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_breast_cancer

# Print version information
print(f"SHAP version: {shap.__version__}")
print(f"XGBoost version: {xgb.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# Create a simple dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Simple model
model = xgb.XGBClassifier(n_estimators=10)
model.fit(X, y)

# Try the SHAP TreeExplainer
try:
    print("\nCreating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    print("TreeExplainer created successfully!")
    
    # Try calculating SHAP values 
    print("Calculating SHAP values...")
    shap_values = explainer(X[:5])
    print("SHAP values calculated successfully!")
    print(f"SHAP values shape: {shap_values.values.shape}")
    
    # This is the most important part that verifies our patches work
    print("\nSUCCESS: SHAP is working properly with the applied patches!")
except Exception as e:
    print(f"\nERROR: SHAP test failed: {e}")
    import traceback
    traceback.print_exc()
