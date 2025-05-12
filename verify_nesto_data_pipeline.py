#!/usr/bin/env python3
"""
Script to verify the Nesto data pipeline, from raw data with descriptive field names
to processed data and model training/prediction.
"""

import os
import pandas as pd
import logging
import pickle
import sys
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("verify-pipeline")

def verify_nesto_data_pipeline():
    """Verifies the entire Nesto data pipeline."""
    logger.info("Starting Nesto data pipeline verification")
    
    # Step 1: Check if the raw data file exists
    raw_data_path = 'data/nesto_merge_0.csv'
    if not os.path.exists(raw_data_path):
        logger.error(f"Raw data file not found at {raw_data_path}")
        return False
    
    logger.info(f"Found raw data file: {raw_data_path}")
    logger.info("Pipeline now uses nesto_merge_0.csv directly with field mapping during loading")
    
    # Step 2: Validate the field mappings
    try:
        from map_nesto_data import FIELD_MAPPINGS, TARGET_VARIABLE
        logger.info(f"Field mappings validated: {len(FIELD_MAPPINGS)} mappings defined")
        logger.info(f"Target variable: {TARGET_VARIABLE}")
    except Exception as e:
        logger.error(f"Failed to validate field mappings: {e}")
        return False
    
    # Step 3: Run the model training process
    try:
        logger.info("Running model training process")
        result = subprocess.run(['python', 'train_model.py'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Model training completed successfully")
            # Get the last 200 characters of the output to show completion info
            if result.stdout:
                logger.info(f"Training output highlights: {result.stdout[-200:]}")
        else:
            logger.error(f"Model training failed with code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output: {result.stderr[-500:]}")  # Last 500 chars of error
            return False
    except Exception as e:
        logger.error(f"Exception during model training: {e}")
        return False
    
    # Step 4: Verify that the model was created
    model_path = 'models/xgboost_model.pkl'
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return False
    
    logger.info(f"Model file found at {model_path}")
    
    # Step 5: Verify that feature names were saved
    feature_names_path = 'models/feature_names.txt'
    if not os.path.exists(feature_names_path):
        logger.error(f"Feature names file not found at {feature_names_path}")
        return False
    
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    logger.info(f"Feature names file found with {len(feature_names)} features")
    
    # Step 6: Load the model and verify it can make predictions
    try:
        model = pickle.load(open(model_path, 'rb'))
        
        # Create a small sample for prediction - using a more efficient approach
        # Create a dictionary of all features with default values
        feature_dict = {feature: [0.0] for feature in feature_names}
        
        # Create the DataFrame in one operation to avoid fragmentation
        X_sample = pd.DataFrame(feature_dict)
        
        # Make a prediction
        prediction = model.predict(X_sample)
        logger.info(f"Made a test prediction: {prediction}")
    except Exception as e:
        logger.error(f"Failed to load model or make prediction: {e}")
        return False
    
    logger.info("VERIFICATION SUCCESSFUL: The Nesto data pipeline is working correctly")
    return True

if __name__ == "__main__":
    success = verify_nesto_data_pipeline()
    sys.exit(0 if success else 1)
