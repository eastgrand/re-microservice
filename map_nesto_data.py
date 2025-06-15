#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maps Nesto data from nesto_merge_0.csv to the expected format for the model.
This utility handles the transformation from descriptive field names to the
standardized field names expected by the model.
"""

import pandas as pd
import yaml
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

# --- MASTER SCHEMA DEFINITION ---
# This is the single source of truth for all data fields, their mappings, and aliases.
# The schema endpoint will expose this to the frontend.
MASTER_SCHEMA = {
    'FREQUENCY': {
        'canonical_name': 'FREQUENCY',
        'raw_mapping': 'FREQUENCY',
        'aliases': ['frequency', 'applications', 'application', 'numberOfApplications', 'number_of_applications', 'mortgage_applications'],
        'description': 'Total number of mortgage applications.'
    },
    'mortgage_approvals': {
        'canonical_name': 'mortgage_approvals',
        'raw_mapping': 'SUM_FUNDED',
        'aliases': ['approvals', 'approval', 'funded_applications', 'funded', 'SUM_FUNDED'],
        'description': 'Total number of funded mortgage applications.'
    },
    'conversion_rate': {
        'canonical_name': 'conversion_rate',
        'raw_mapping': 'CONVERSION_RATE',
        'aliases': ['conversion rate', 'conversionrate'],
        'description': 'The ratio of funded applications to total applications.'
    },
    'median_income': {
        'canonical_name': 'median_income',
        'raw_mapping': '2024 Household Average Income (Current Year $)',
        'aliases': ['income', 'average income', 'household income'],
        'description': 'The median household income in the area.'
    },
    'disposable_income': {
        'canonical_name': 'disposable_income',
        'raw_mapping': '2024 Household Discretionary Aggregate Income',
        'aliases': ['discretionary income', 'householdDiscretionaryIncome', 'household_discretionary_income', 'householddiscretionaryincome'],
        'description': 'The aggregate discretionary income for households.'
    },
    'condo_ownership_pct': {
        'canonical_name': 'condo_ownership_pct',
        'raw_mapping': '2024 Condominium Status - In Condo (%)',
        'aliases': ['condo ownership', 'condominium', 'condo_ownership'],
        'description': 'Percentage of households that are condominiums.'
    },
    'visible_minority_population_pct': {
        'canonical_name': 'visible_minority_population_pct',
        'raw_mapping': '2024 Visible Minority Total Population (%)',
        'aliases': ['visible minority', 'visibleMinorityPopulationPct', 'visible_minority_population'],
        'description': 'Percentage of the population identified as a visible minority.'
    }
}

# --- Dynamically Generated Mappings (from MASTER_SCHEMA) ---

# FIELD_MAPPINGS: Connects raw CSV headers to the canonical names used in the service.
FIELD_MAPPINGS = {
    details['raw_mapping']: details['canonical_name']
    for _, details in MASTER_SCHEMA.items()
    if 'raw_mapping' in details
}

# NUMERIC_COLS: A list of all canonical names that should be treated as numeric.
NUMERIC_COLS = [details['canonical_name'] for _, details in MASTER_SCHEMA.items()]

# The target variable for the model
TARGET_VARIABLE = 'conversion_rate'

def load_and_preprocess_data(config_path='config/dataset.yaml'):
    """
    Loads the dataset and configuration, then preprocesses the data by renaming
    columns and converting types based on the centrally defined schema.
    """
    logging.info(f"Loading dataset configuration from: {config_path}")
    
    # Correctly locate the config file relative to the script's directory
    script_dir = os.path.dirname(__file__)
    abs_config_path = os.path.join(script_dir, config_path)
    
    with open(abs_config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Define paths relative to the script's location
    base_path = script_dir
    raw_data_path = os.path.join(base_path, config['raw_data_path'])
    cleaned_data_path = os.path.join(base_path, config['cleaned_data_path'])

    logging.info(f"Loading raw data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    
    # Rename columns based on the dynamic FIELD_MAPPINGS
    df.rename(columns=FIELD_MAPPINGS, inplace=True)
    logging.info("Renamed columns based on master schema.")
    
    # Ensure all columns required by the schema exist, otherwise log a warning
    final_columns = list(FIELD_MAPPINGS.values())
    for col in final_columns:
        if col not in df.columns:
            logging.warning(f"Column '{col}' not found in DataFrame after renaming. It will be missing from the output.")
    
    # Select only the columns defined in our canonical schema
    df = df[[col for col in final_columns if col in df.columns]]

    # Convert specified columns to numeric, coercing errors
    logging.info(f"Converting numeric columns: {NUMERIC_COLS}")
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logging.warning(f"Numeric column '{col}' not found for type conversion.")
            
    # Fill NaN values with the median of the column
    df.fillna(df.median(numeric_only=True), inplace=True)
    logging.info("Filled NaN values with column medians.")

    # Save the cleaned data
    os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
    df.to_csv(cleaned_data_path, index=False)
    logging.info(f"Cleaned data saved to: {cleaned_data_path}")

    return df

if __name__ == '__main__':
    # This allows the script to be run directly to regenerate the cleaned data
    load_and_preprocess_data()