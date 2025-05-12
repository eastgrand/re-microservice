#!/usr/bin/env python3
"""
Test script to verify that percentage fields are properly mapped.
"""

import pandas as pd
import os
import logging
from map_nesto_data import map_nesto_data, FIELD_MAPPINGS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test-percentage-fields")

def test_percentage_fields():
    """Test the mapping of percentage fields."""
    logger.info("Testing percentage field mappings")
    
    # Create a test dataset with percentage fields
    test_data = {
        'Forward Sortation Area': ['A1A', 'B2B', 'C3C'],
        '2024 Total Population': [10000, 20000, 30000],
        '2024 Household Average Income (Current Year $)': [75000, 85000, 95000],
        '2024 Tenure: Owned': [2000, 3000, 4000],
        '2024 Tenure: Owned (%)': [70.0, 60.0, 80.0],
        '2024 Tenure: Rented': [857, 2000, 1000],
        '2024 Tenure: Rented (%)': [30.0, 40.0, 20.0],
        '2024 Visible Minority Black': [500, 1000, 1500],
        '2024 Visible Minority Black (%)': [5.0, 5.0, 5.0],
        '2024 Tenure: Total Households': [2857, 5000, 5000],
        '2024 Labour Force - Labour Employment Rate': [95.0, 92.0, 94.0],
        '2024 Condominium Status - In Condo (%)': [25.0, 30.0, 15.0],
        '2024 Female Household Population (%)': [51.2, 50.8, 50.5],
        'Funded Applications': [150, 200, 250]
    }
    
    test_df = pd.DataFrame(test_data)
    
    # Save the test data to a temporary file
    temp_input_path = 'data/temp_test_input.csv'
    temp_output_path = 'data/temp_test_output.csv'
    test_df.to_csv(temp_input_path, index=False)
    
    # Map the data using our mapping function
    logger.info("Mapping test data...")
    mapped_df = map_nesto_data(temp_input_path, temp_output_path)
    
    # Check if the percentage fields are properly mapped
    expected_mappings = {
        'zip_code': 'Forward Sortation Area',
        'Homeownership': '2024 Tenure: Owned',
        'Homeownership_Pct': '2024 Tenure: Owned (%)',
        'Rental_Units': '2024 Tenure: Rented',
        'Rental_Units_Pct': '2024 Tenure: Rented (%)',
        'African_American_Population': '2024 Visible Minority Black',
        'African_American_Population_Pct': '2024 Visible Minority Black (%)',
        'Households': '2024 Tenure: Total Households',
        'Employment_Rate': '2024 Labour Force - Labour Employment Rate',
        'Condo_Ownership_Pct': '2024 Condominium Status - In Condo (%)',
        'Female_Population_Pct': '2024 Female Household Population (%)',
        'Mortgage_Approvals': 'Funded Applications'
    }
    
    logger.info(f"Mapped columns: {mapped_df.columns.tolist()}")
    
    # Verify each mapping
    success = True
    for target, source in expected_mappings.items():
        if target not in mapped_df.columns:
            logger.error(f"Missing expected column: {target} (should be mapped from {source})")
            success = False
        else:
            logger.info(f"Successfully mapped {source} to {target}")
    
    # Clean up temporary files
    os.remove(temp_input_path)
    os.remove(temp_output_path)
    
    if success:
        logger.info("SUCCESS: All percentage fields were properly mapped")
    else:
        logger.error("FAILURE: Some percentage fields were not properly mapped")
    
    return success

if __name__ == "__main__":
    test_percentage_fields()
