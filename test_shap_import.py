#!/usr/bin/env python3
"""
Simple script to verify SHAP imports are working correctly
"""
import sys
import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
print("Importing SHAP...")
import shap
print(f"SHAP version: {shap.__version__}")

# Check for important modules
try:
    from shap.explainers import Tree as TreeExplainer
    print("TreeExplainer imported successfully")
except Exception as e:
    print(f"Error importing TreeExplainer: {e}")

# Try loading our model
try:
    model_path = os.path.join('models', 'xgboost_model.pkl')
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

print("Import test completed")
