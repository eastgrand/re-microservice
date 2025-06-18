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
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

# --- CORE FIELD MAPPINGS ---
# For ArcGIS demographic data, we'll keep the original field names
# This eliminates confusion in field resolution
CORE_FIELD_MAPPINGS = {
    # No field renaming - keep original ArcGIS field names as they are clean and consistent
}

# The target variable for the model
TARGET_VARIABLE = 'TOTPOP_CY'

def generate_field_alias(field_name):
    """Generate common aliases for a field name"""
    aliases = [field_name.lower()]
    
    # Convert to snake_case
    snake_case = re.sub(r'[^\w\s]', '', field_name.lower())
    snake_case = re.sub(r'\s+', '_', snake_case.strip())
    aliases.append(snake_case)
    
    # Add variations without common words
    clean_name = re.sub(r'\b(2024|2023|2022|2021|census|current|year|\$|%)\b', '', field_name.lower())
    clean_name = re.sub(r'[^\w\s]', '', clean_name)
    clean_name = re.sub(r'\s+', '_', clean_name.strip())
    if clean_name and clean_name != snake_case:
        aliases.append(clean_name)
    
    # Add percentage field aliases first
    if '(%)' in field_name:
        base_name = field_name.replace('(%)', '').strip()
        base_snake = re.sub(r'[^\w\s]', '', base_name.lower())
        base_snake = re.sub(r'\s+', '_', base_snake.strip())
        aliases.append(base_snake + '_pct')
        
        # Remove year prefixes for cleaner aliases
        clean_base = re.sub(r'\b(2024|2023|2022|2021)\b', '', base_name).strip()
        if clean_base:
            clean_snake = re.sub(r'[^\w\s]', '', clean_base.lower())
            clean_snake = re.sub(r'\s+', '_', clean_snake.strip())
            aliases.append(clean_snake + '_pct')
    
    # Add specific aliases for common patterns
    if 'single-detached house' in field_name.lower():
        aliases.extend(['single_detached_house', 'detached_house', 'single_family_house', 'single_detached_house_pct'])
    elif 'semi-detached house' in field_name.lower():
        aliases.extend(['semi_detached_house', 'semi_detached_house_pct'])
    elif 'apartment' in field_name.lower():
        aliases.extend(['apartment', 'apartments'])
    elif 'condominium' in field_name.lower():
        aliases.extend(['condo', 'condominium'])
    elif 'visible minority' in field_name.lower():
        aliases.extend(['visible_minority', 'minority', 'diversity'])
    elif 'income' in field_name.lower():
        aliases.extend(['income', 'household_income'])
    
    # Special case for the specific field we need
    if field_name == '2024 Structure Type Single-Detached House (%)':
        aliases.extend(['single_detached_house_pct', 'detached_house_pct', 'single_family_house_pct'])
    
    return list(set(aliases))  # Remove duplicates

def generate_dynamic_schema(df):
    """Generate schema for all fields in the dataset"""
    schema = {}
    
    for col in df.columns:
        # Check if this field has a core mapping
        canonical_name = CORE_FIELD_MAPPINGS.get(col, col)
        
        # Generate description
        if col in CORE_FIELD_MAPPINGS:
            # Use predefined descriptions for core demographic fields
            descriptions = {
                'total_population': 'Total population in the area.',
                'asian_population': 'Asian population count.',
                'black_population': 'Black population count.',
                'white_population': 'White population count.',
                'american_indian_population': 'American Indian population count.',
                'pacific_islander_population': 'Pacific Islander population count.',
                'other_race_population': 'Other race population count.',
                'multiracial_population': 'Multiracial population count.',
                'median_income': 'Median household income in the area.',
                'millennial_population': 'Millennial generation population count.',
                'genz_population': 'Generation Z population count.',
                'gen_alpha_population': 'Generation Alpha population count.',
                'household_population': 'Total household population.',
                'family_population': 'Total family population.',
                'hispanic_white_population': 'Hispanic white population count.',
                'hispanic_black_population': 'Hispanic black population count.',
                'hispanic_american_indian_population': 'Hispanic American Indian population count.',
                'hispanic_pacific_islander_population': 'Hispanic Pacific Islander population count.',
                'hispanic_other_population': 'Hispanic other race population count.',
                'diversity_index': 'Diversity index score for the area.',
                'wealth_index': 'Wealth index score for the area.',
            }
            description = descriptions.get(canonical_name, f'Data field: {col}')
        else:
            description = f'Data field: {col}'
        
        # Determine data type
        data_type = 'numeric' if df[col].dtype in ['int64', 'float64'] else 'string'
        
        schema[canonical_name] = {
            'canonical_name': canonical_name,
            'raw_mapping': col,
            'aliases': generate_field_alias(col),
            'description': description,
            'type': data_type
        }
    
    return schema

# Global variables that will be set when data is loaded
MASTER_SCHEMA = {}
FIELD_MAPPINGS = {}
NUMERIC_COLS = []

def initialize_schema(df):
    """Initialize the global schema variables from the loaded dataframe"""
    global MASTER_SCHEMA, FIELD_MAPPINGS, NUMERIC_COLS
    
    MASTER_SCHEMA = generate_dynamic_schema(df)
    
    # Generate field mappings (raw name -> canonical name)
    FIELD_MAPPINGS = {
        details['raw_mapping']: details['canonical_name']
        for _, details in MASTER_SCHEMA.items()
        if 'raw_mapping' in details
    }

    # All numeric columns
    NUMERIC_COLS = [
        details['canonical_name'] 
        for _, details in MASTER_SCHEMA.items() 
        if details.get('type') == 'numeric'
    ]
    
    logging.info(f"Initialized dynamic schema with {len(MASTER_SCHEMA)} fields")

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
    raw_data_path = os.path.join(base_path, config['raw_csv'])
    # Ensure the cleaned data path is always 'data/cleaned_data.csv'
    cleaned_data_path = os.path.join(base_path, 'data', 'cleaned_data.csv')

    logging.info(f"Loading raw data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    logging.info(f"Initial columns from raw CSV: {df.columns.tolist()}")
    
    # Apply core field mappings only
    df.rename(columns=CORE_FIELD_MAPPINGS, inplace=True)
    logging.info("Applied core field mappings.")
    logging.info(f"Columns after renaming: {df.columns.tolist()}")
    
    # Keep ALL columns, not just the ones in MASTER_SCHEMA
    # This preserves all 137 fields while applying canonical mappings where they exist
    logging.info(f"Preserving all {len(df.columns)} columns from the dataset")
    
    # Convert all numeric-looking columns to numeric, not just the ones in MASTER_SCHEMA
    numeric_converted = 0
    for col in df.columns:
        # Skip obvious non-numeric columns
        if col.upper() in ['OBJECTID', 'ID', 'FORWARD SORTATION AREA', 'PROVINCE_CODE']:
            continue
            
        # Try to convert to numeric
        original_dtype = df[col].dtype
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check if conversion was successful (dtype changed)
        if df[col].dtype != original_dtype:
            numeric_converted += 1
    
    logging.info(f"Converted {numeric_converted} columns to numeric type")
        
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