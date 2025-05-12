#!/usr/bin/env python3
"""
Maps Nesto data from nesto_merge_0.csv to the expected format for the model.
This utility handles the transformation from descriptive field names to the
standardized field names expected by the model.
"""

import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nesto-data-mapper")

# Define the field mappings according to NESTO_FIELD_MAPPING.md
FIELD_MAPPINGS = {
    'Forward Sortation Area': 'zip_code',
    '2024 Total Population': 'Population',
    '2024 Household Average Income (Current Year $)': 'Income',
    '2024 Maintainers - Median Age': 'Age',
    '2024 Tenure: Owned': 'Homeownership',
    '2024 Tenure: Rented': 'Rental_Units',
    '2024 Visible Minority Black': 'African_American_Population',
    '2024 Visible Minority Chinese': 'Asian_Population',
    '2024 Visible Minority Latin American': 'Latin_American_Population',
    '2024 Visible Minority Total Population': 'Total_Minority_Population',
    '2024 Labour Force - Labour Participation Rate': 'Labor_Participation_Rate',
    '2024 Labour Force - Labour Employment Rate': 'Employment_Rate',
    '2024 Tenure: Total Households': 'Households',
    'Funded Applications': 'Mortgage_Approvals'
}

# Target variable for the model
TARGET_VARIABLE = 'Mortgage_Approvals'


def map_nesto_data(input_path='data/nesto_merge_0.csv', output_path='data/cleaned_data.csv'):
    """
    Maps data from nesto_merge_0.csv with descriptive field names to the format
    expected by the model with standardized field names.
    
    Args:
        input_path (str): Path to the input CSV file with descriptive field names
        output_path (str): Path where the mapped data will be saved
        
    Returns:
        pd.DataFrame: The mapped DataFrame
    """
    logger.info(f"Loading Nesto data from {input_path}")
    
    try:
        # Read the input CSV file
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Check if required columns exist
        missing_columns = [col for col in FIELD_MAPPINGS.keys() if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns in input data: {missing_columns}")
            
        # Create a new DataFrame with mapped column names
        mapped_df = pd.DataFrame()
        
        # Apply mappings
        for source_col, target_col in FIELD_MAPPINGS.items():
            if source_col in df.columns:
                mapped_df[target_col] = df[source_col]
                logger.info(f"Mapped '{source_col}' to '{target_col}'")
            else:
                logger.warning(f"Column '{source_col}' not found in input data")
                
        # Handle location data if available (latitude/longitude)
        if 'Shape__Area' in df.columns and 'Shape__Length' in df.columns:
            # Use these as proxy values for positioning if real coordinates aren't available
            # This is just for visualization purposes
            mapped_df['latitude'] = df['Shape__Area'].rank(pct=True) * 10
            mapped_df['longitude'] = df['Shape__Length'].rank(pct=True) * 10
            logger.info("Added proxy latitude/longitude based on Shape area/length")
        
        # Check if we have the required target variable
        target_source = next((k for k, v in FIELD_MAPPINGS.items() if v == TARGET_VARIABLE), None)
        if target_source and target_source in df.columns:
            logger.info(f"Target variable '{target_source}' mapped to '{TARGET_VARIABLE}'")
        else:
            logger.error(f"Target variable not found in the source data")
            
        # Basic validation - check that we have the expected columns
        expected_columns = ['zip_code', TARGET_VARIABLE]  # Minimal required columns
        missing_expected = [col for col in expected_columns if col not in mapped_df.columns]
        if missing_expected:
            logger.error(f"Critical columns missing after mapping: {missing_expected}")
        
        # Save the mapped data
        mapped_df.to_csv(output_path, index=False)
        logger.info(f"Saved mapped data to {output_path} with {len(mapped_df)} records and {len(mapped_df.columns)} columns")
        
        return mapped_df
        
    except Exception as e:
        logger.error(f"Error mapping Nesto data: {str(e)}")
        raise


if __name__ == "__main__":
    # Create the data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Default paths
    input_path = 'data/nesto_merge_0.csv'
    output_path = 'data/cleaned_data.csv'
    
    # Map the data
    map_nesto_data(input_path, output_path)