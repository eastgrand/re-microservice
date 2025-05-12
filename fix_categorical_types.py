#!/usr/bin/env python3
"""
Fix for handling categorical data types in the microservice.
This script is used during the Render deployment process.
"""

import pandas as pd
import numpy as np
import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix-categorical")

def fix_categorical_types():
    """
    Fix categorical data types in the dataset by forcing conversion to string.
    This ensures proper JSON serialization for API responses.
    """
    # Check if the dataset exists
    dataset_paths = [
        'data/cleaned_data.csv',
        'data/nesto_merge_0.csv',
        'data/minimal_dataset.csv'
    ]
    
    fixed = False
    for path in dataset_paths:
        if os.path.exists(path):
            logger.info(f"Processing dataset: {path}")
            
            try:
                # Load the dataset
                df = pd.read_csv(path)
                original_shape = df.shape
                logger.info(f"Dataset shape: {original_shape}")
                
                # Convert categorical columns to string
                for col in df.columns:
                    if df[col].dtype.name == 'category':
                        logger.info(f"Converting categorical column to string: {col}")
                        df[col] = df[col].astype(str)
                
                # Save with a new filename
                fixed_path = path.replace('.csv', '_fixed.csv')
                df.to_csv(fixed_path, index=False)
                logger.info(f"Saved fixed dataset to: {fixed_path}")
                
                # Update environment variable to use fixed dataset
                os.environ['DATASET_PATH'] = fixed_path
                
                # Try to update the .env file for persistence
                try:
                    with open('.env', 'a') as f:
                        f.write(f"\nDATASET_PATH={fixed_path}")
                except Exception as e:
                    logger.warning(f"Could not update .env file: {e}")
                
                fixed = True
                
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                continue
    
    if not fixed:
        logger.warning("No datasets found to fix")
    
    return fixed

if __name__ == "__main__":
    try:
        success = fix_categorical_types()
        if success:
            print("Successfully fixed categorical data types")
            sys.exit(0)
        else:
            print("No datasets required fixing")
            sys.exit(0)
    except Exception as e:
        print(f"Error fixing categorical data types: {e}")
        sys.exit(1)
