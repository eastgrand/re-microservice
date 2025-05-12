#!/usr/bin/env python3
"""
Patch for handling categorical data types in the microservice.
This script fixes issues with categorical data types in the dataset.
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("categorical-fix")

def fix_categorical_data():
    """Fix categorical data types in the dataset."""
    # Check if dataset exists
    dataset_paths = [
        'data/cleaned_data.csv',
        'data/nesto_merge_0.csv',
        'data/minimal_dataset.csv'
    ]
    
    fixed_any = False
    
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            logger.info(f"Fixing categorical data in {dataset_path}")
            
            try:
                # Load the dataset
                dataset = pd.read_csv(dataset_path)
                
                # Check for categorical columns
                cat_columns = dataset.select_dtypes(include=['category']).columns.tolist()
                if cat_columns:
                    logger.info(f"Found {len(cat_columns)} categorical columns: {cat_columns}")
                    
                    # Convert categorical columns to string
                    for col in cat_columns:
                        dataset[col] = dataset[col].astype(str)
                    
                    # Save the fixed dataset
                    fixed_path = dataset_path.replace('.csv', '_fixed.csv')
                    dataset.to_csv(fixed_path, index=False)
                    logger.info(f"Saved fixed dataset to {fixed_path}")
                    
                    # Update environment variable to use the fixed dataset
                    if 'cleaned_data.csv' in dataset_path:
                        os.environ['DATASET_PATH'] = fixed_path
                        with open('.env', 'a') as f:
                            f.write(f"\nDATASET_PATH={fixed_path}")
                        logger.info(f"Updated DATASET_PATH to {fixed_path}")
                    
                    fixed_any = True
                else:
                    logger.info(f"No categorical columns found in {dataset_path}")
            except Exception as e:
                logger.error(f"Error processing {dataset_path}: {e}")
    
    if not fixed_any:
        logger.info("No categorical data needed fixing")
    
    return fixed_any

if __name__ == "__main__":
    try:
        if fix_categorical_data():
            print("Successfully fixed categorical data issues")
            sys.exit(0)
        else:
            print("No categorical data issues found")
            sys.exit(0)
    except Exception as e:
        print(f"Error fixing categorical data: {e}")
        sys.exit(1)
