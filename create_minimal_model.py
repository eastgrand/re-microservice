#!/usr/bin/env python3
"""
Creates a minimal XGBoost model for fallback purposes.
This is used when the full model can't be loaded or created on Render.
Updated for more reliable deployment - May 15, 2025
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
import pickle
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("minimal-model-creator")

# Create standard directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
logger.info("Ensured models and data directories exist")

def create_minimal_model():
    """Create a minimal XGBoost model that can be used as fallback."""
    logger.info("Creating minimal XGBoost model for fallback purposes")
    
    try:
        # Import XGBoost here to handle import errors gracefully
        import xgboost as xgb
        logger.info("Successfully imported XGBoost")
    except ImportError:
        logger.warning("XGBoost not available, creating fake model files instead")
        return create_fake_model_files()
    
    # Create a minimal dataset with key columns (typical mortgage data)
    data = {
        "Income": np.random.normal(60000, 15000, 100),
        "Age": np.random.normal(40, 10, 100),
        "Credit_Score": np.random.normal(700, 50, 100),
        "Homeownership_Pct": np.random.normal(70, 15, 100),
        "Employment_Years": np.random.normal(8, 4, 100),
        "Debt_To_Income": np.random.normal(0.3, 0.1, 100),
        "Mortgage_Approvals": np.random.normal(50, 10, 100)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Split into features and target
    X = df.drop("Mortgage_Approvals", axis=1)
    y = df["Mortgage_Approvals"]
    
    # Create feature names
    feature_names = list(X.columns)
    logger.info(f"Using features: {feature_names}")
    
    # Create and train minimal model
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
    
    # Train model with minimal iterations
    model = xgb.train(params, dtrain, num_boost_round=10)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    minimal_model_path = 'models/xgboost_minimal.pkl'
    pickle.dump(model, open(minimal_model_path, 'wb'))
    logger.info(f"Saved minimal model to {minimal_model_path}")
    
    # Save feature names
    minimal_features_path = 'models/minimal_feature_names.txt'
    with open(minimal_features_path, 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    logger.info(f"Saved minimal feature names to {minimal_features_path}")
    
    # Create minimal dataset
    os.makedirs('data', exist_ok=True)
    minimal_data_path = 'data/minimal_dataset.csv'
    df.to_csv(minimal_data_path, index=False)
    logger.info(f"Saved minimal dataset to {minimal_data_path}")
    
    return model, feature_names

if __name__ == "__main__":
    model, features = create_minimal_model()
    print(f"Successfully created minimal model with {len(features)} features")
    sys.exit(0)
