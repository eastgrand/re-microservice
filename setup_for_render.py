#!/usr/bin/env python3
"""
Setup script to prepare the environment for deployment to Render.com.
This script ensures all data files and models are correctly prepared.
"""

# SKIP TRAINING CHECK - Added May 15, 2025
import os
SKIP_TRAINING = os.path.exists(".skip_training") or os.environ.get("SKIP_MODEL_TRAINING") == "true"
if SKIP_TRAINING:
    print("⚡ SKIP TRAINING FLAG DETECTED - MODEL TRAINING WILL BE BYPASSED")
    # Set environment variable to ensure other scripts know about this too
    os.environ["SKIP_MODEL_TRAINING"] = "true"

import sys
import logging
import subprocess
import pandas as pd
import numpy as np
import shutil
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("setup-for-render")

# Check for memory optimization mode
MEMORY_OPTIMIZATION = os.environ.get('MEMORY_OPTIMIZATION', 'false').lower() == 'true'
if MEMORY_OPTIMIZATION:
    logger.info("Memory optimization mode is ENABLED for setup")
    try:
        from optimize_memory import log_memory_usage, get_memory_usage
        log_memory_usage("Setup script start")
    except ImportError:
        logger.warning("Could not import optimize_memory module, continuing without memory tracking")
        
        # Simple implementation in case the module isn't available yet
        def log_memory_usage(step_name):
            logger.info(f"Memory usage tracking not available for: {step_name}")
else:
    logger.info("Running in standard mode (memory optimization disabled)")

def setup_environment():
    """Prepare the environment for deployment."""
    logger.info("Setting up environment for deployment to Render.com")
    
    # Step 1: Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    logger.info("Created necessary directories")
    
    # Step 2: Check if we have the Nesto data file
    nesto_data_path = 'data/nesto_merge_0.csv'
    if not os.path.exists(nesto_data_path):
        logger.error(f"❌ Required Nesto data file not found at {nesto_data_path}.")
        logger.error("Deployment cannot proceed without this file. Please upload the real data file and retry.")
        return False
    else:
        logger.info(f"Found Nesto data file at {nesto_data_path}")
    
    # Step 3: Run the data mapping script
    try:
        logger.info("Mapping data fields...")
        from map_nesto_data import load_and_preprocess_data
        # The function now reads its config internally, so no args are needed.
        load_and_preprocess_data()
        logger.info("Data mapping completed successfully")

        # --- AUTOMATION: Rename CONVERSION_RATE to CONVERSIONRATE in cleaned_data.csv ---
        cleaned_path = 'data/cleaned_data.csv'
        df = pd.read_csv(cleaned_path)
        if 'CONVERSION_RATE' in df.columns:
            logger.info("Renaming 'CONVERSION_RATE' column to 'CONVERSIONRATE' for blob export compatibility.")
            df.rename(columns={'CONVERSION_RATE': 'CONVERSIONRATE'}, inplace=True)
            df.to_csv(cleaned_path, index=False)
            logger.info("Column rename complete. Saved updated cleaned_data.csv.")
        else:
            logger.info("No 'CONVERSION_RATE' column found to rename.")
        # --- END AUTOMATION ---
    except Exception as e:
        logger.error(f"Error during data mapping: {e}")
        return False
    
    # Step 4: Run the model training script (unless skipped)
    if SKIP_TRAINING:
        logger.info("⚡ SKIPPING MODEL TRAINING due to skip_training flag")
        logger.info("Using existing model files from repository")
        
        # Verify model files exist
        if os.path.exists("models/xgboost_model.pkl") and os.path.exists("models/feature_names.txt"):
            logger.info("✅ Model files found - proceeding without training")
            return True
        else:
            logger.warning("⚠️ Model files not found but skip_training is enabled")
            logger.info("Creating minimal model instead...")
            try:
                result = subprocess.run([sys.executable, 'create_minimal_model.py'], 
                                  capture_output=True, text=True, check=True)
                logger.info("Created minimal model successfully")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Error creating minimal model: {e}")
                logger.error(f"Error output: {e.stderr}")
                return False
    else:
        try:
            logger.info("Training model...")
            result = subprocess.run([sys.executable, 'train_model.py'], 
                                  capture_output=True, text=True, check=True)
            logger.info("Model training completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during model training: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during model training: {e}")
            return False
    
    # Step 5: Verify that the model file exists
    model_path = 'models/xgboost_model.pkl'
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        return False
    logger.info(f"Model file created at {model_path}")
    
    # Step 6: Verify that feature names file exists
    feature_names_path = 'models/feature_names.txt'
    if not os.path.exists(feature_names_path):
        logger.error(f"Feature names file not found at {feature_names_path}")
        return False
    logger.info(f"Feature names file created at {feature_names_path}")
    
    logger.info("Environment setup completed successfully!")
    return True

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)
