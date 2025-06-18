#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-generated field mappings from Project Configuration Manager
DO NOT EDIT MANUALLY - Use Project Configuration Manager to update
"""

import pandas as pd
import yaml
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

# --- CORE FIELD MAPPINGS ---
# Auto-generated from Project Configuration Manager
CORE_FIELD_MAPPINGS = {

}

# The target variable for the model
TARGET_VARIABLE = 'conversion_rate'

def generate_field_alias(field_name):
    """Generate aliases for a field name"""
    aliases = [field_name.lower()]
    
    # Add snake_case version
    snake_case = re.sub(r'[^\w\s]', '', field_name.lower())
    snake_case = re.sub(r'\s+', '_', snake_case)
    if snake_case != field_name.lower():
        aliases.append(snake_case)
    
    # Add version without year prefixes
    no_year = re.sub(r'\b(2024|2023|2022|2021)\b', '', field_name).strip()
    if no_year and no_year != field_name:
        aliases.append(no_year.lower())
        aliases.append(re.sub(r'\s+', '_', re.sub(r'[^\w\s]', '', no_year.lower())))
    
    return list(set(aliases))

def generate_dynamic_schema(df):
    """Generate schema for all fields in the dataset"""
    schema = {}
    
    for col in df.columns:
        # Check if this field has a core mapping
        canonical_name = CORE_FIELD_MAPPINGS.get(col, col)
        
        # Generate description
        if canonical_name in FIELD_DESCRIPTIONS:
            description = FIELD_DESCRIPTIONS[canonical_name]
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

# Field descriptions for auto-generated mappings
FIELD_DESCRIPTIONS = {

}

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

def load_and_preprocess_data():
    """Load and preprocess the Nesto data"""
    # Implementation would go here
    pass