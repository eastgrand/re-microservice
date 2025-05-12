"""
A simplified version of the SHAP module for use with the microservice.
"""

import pandas as pd
import numpy as np

class TreeExplainer:
    """
    A simplified version of the SHAP TreeExplainer that doesn't rely on np.int
    """
    def __init__(self, model):
        self.model = model
        
    def __call__(self, X):
        """
        Calculate feature importance by using the XGBoost feature importance directly
        """
        # Get feature importances from the model
        importance = self.model.feature_importances_
        
        # Create a simple structure that mimics the SHAP values output
        class ShapValues:
            def __init__(self, values, feature_names):
                self.values = values
                self.feature_names = feature_names
                self.base_values = np.zeros(len(X))
                
        # Create a matrix of values where each row is the same importance vector
        # This is a simplification as real SHAP values would be different for each instance
        values = np.tile(importance, (len(X), 1))
        
        return ShapValues(values, X.columns)
