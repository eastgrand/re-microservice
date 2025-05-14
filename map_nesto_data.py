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
    # Geographic fields
    # 'Forward Sortation Area' is the FSA (Canadian postal code prefix) and is mapped to 'zip_code'.
    # 'ID' is a province code, not an FSA, and is mapped to 'Province_Code'.
    # The following fields are mapped for reference but are always excluded from model features:
    #   'Object ID' -> 'Object_ID', 'Shape__Area' -> 'Geographic_Area', 'Shape__Length' -> 'Geographic_Length'
    # See train_model.py for enforced exclusion.
    'Forward Sortation Area': 'zip_code',
    'ID': 'Province_Code',
    'Object ID': 'Object_ID',
    'Shape__Area': 'Geographic_Area',
    'Shape__Length': 'Geographic_Length',
    
    # Basic demographic fields
    '2024 Total Population': 'Population',
    '2024 Household Average Income (Current Year $)': 'Income',
    '2024 Household Median Income (Current Year $)': 'Median_Income',
    '2024 Maintainers - Median Age': 'Age',
    '2024 Female Household Population': 'Female_Population',
    '2024 Female Household Population (%)': 'Female_Population_Pct',
    '2024 Male Household Population': 'Male_Population',
    '2024 Male Household Population (%)': 'Male_Population_Pct',
    '2024 Household Type - Total Households': 'Total_Household_Types',
    
    # Age groups and household maintainers
    '2024 Maintainers - 25 to 34': 'Young_Adult_Maintainers',
    '2024 Maintainers - 25 to 34 (%)': 'Young_Adult_Maintainers_Pct',
    '2024 Maintainers - 35 to 44': 'Middle_Age_Maintainers',
    '2024 Maintainers - 35 to 44 (%)': 'Middle_Age_Maintainers_Pct',
    '2024 Maintainers - 45 to 54': 'Mature_Maintainers',
    '2024 Maintainers - 45 to 54 (%)': 'Mature_Maintainers_Pct',
    '2024 Maintainers - 55 to 64': 'Senior_Maintainers',
    '2024 Maintainers - 55 to 64 (%)': 'Senior_Maintainers_Pct',
    
    # Housing tenure fields
    '2024 Tenure: Owned': 'Homeownership',
    '2024 Tenure: Owned (%)': 'Homeownership_Pct',
    '2024 Tenure: Rented': 'Rental_Units',
    '2024 Tenure: Rented (%)': 'Rental_Units_Pct',
    '2024 Tenure: Band Housing': 'Band_Housing',
    '2024 Tenure: Band Housing (%)': 'Band_Housing_Pct',
    '2024 Tenure: Total Households': 'Households',
    
    # Visible minority demographics
    '2024 Visible Minority Black': 'African_American_Population',
    '2024 Visible Minority Black (%)': 'African_American_Population_Pct',
    '2024 Visible Minority Chinese': 'Asian_Population',
    '2024 Visible Minority Chinese (%)': 'Asian_Population_Pct',
    '2024 Visible Minority Latin American': 'Latin_American_Population',
    '2024 Visible Minority Latin American (%)': 'Latin_American_Population_Pct',
    '2024 Visible Minority Arab': 'Arab_Population',
    '2024 Visible Minority Arab (%)': 'Arab_Population_Pct',
    '2024 Visible Minority Filipino': 'Filipino_Population',
    '2024 Visible Minority Filipino (%)': 'Filipino_Population_Pct',
    '2024 Visible Minority South Asian': 'South_Asian_Population',
    '2024 Visible Minority South Asian (%)': 'South_Asian_Population_Pct',
    '2024 Visible Minority Southeast Asian': 'Southeast_Asian_Population',
    '2024 Visible Minority Southeast Asian (%)': 'Southeast_Asian_Population_Pct',
    '2024 Visible Minority Japanese': 'Japanese_Population',
    '2024 Visible Minority Japanese (%)': 'Japanese_Population_Pct',
    '2024 Visible Minority Korean': 'Korean_Population',
    '2024 Visible Minority Korean (%)': 'Korean_Population_Pct',
    '2024 Visible Minority West Asian': 'West_Asian_Population',
    '2024 Visible Minority West Asian (%)': 'West_Asian_Population_Pct',
    '2024 Visible Minority Total Population': 'Total_Minority_Population',
    '2024 Visible Minority Total Population (%)': 'Total_Minority_Population_Pct',
    
    # Labor market indicators
    '2024 Labour Force - Labour Participation Rate': 'Labor_Participation_Rate',
    '2024 Labour Force - Labour Employment Rate': 'Employment_Rate',
    '2024 Labour Force - Labour Unemployment Rate': 'Unemployment_Rate',
    
    # Housing type fields
    '2024 Condominium Status - In Condo': 'Condo_Units',
    '2024 Condominium Status - In Condo (%)': 'Condo_Ownership_Pct',
    '2024 Condominium Status - Not In Condo': 'Non_Condo_Units',
    '2024 Condominium Status - Not In Condo (%)': 'Non_Condo_Pct',
    '2024 Condominium Status - Total Households': 'Total_Condo_Status_Households',
    '2024 Structure Type Single-Detached House': 'Single_Detached_Houses',
    '2024 Structure Type Single-Detached House (%)': 'Single_Detached_House_Pct',
    '2024 Structure Type Semi-Detached House': 'Semi_Detached_Houses',
    '2024 Structure Type Semi-Detached House (%)': 'Semi_Detached_House_Pct',
    '2024 Structure Type Row House': 'Row_Houses',
    '2024 Structure Type Row House (%)': 'Row_House_Pct',
    '2024 Structure Type Apartment, Building Five or More Story': 'Large_Apartments',
    '2024 Structure Type Apartment, Building Five or More Story (%)': 'Large_Apartment_Pct',
    '2024 Structure Type Apartment, Building Fewer Than Five Story': 'Small_Apartments',
    '2024 Structure Type Apartment, Building Fewer Than Five Story (%)': 'Small_Apartment_Pct',
    '2024 Structure Type Movable Dwelling': 'Movable_Dwellings',
    '2024 Structure Type Movable Dwelling (%)': 'Movable_Dwelling_Pct',
    '2024 Structure Type Other Single-Attached House': 'Other_Single_Attached_Houses',
    '2024 Structure Type Other Single-Attached House (%)': 'Other_Attached_House_Pct',
    '2021 Housing: Apartment or Flat in Duplex (Census)': 'Duplex_Apartments',
    '2021 Housing: Apartment or Flat in Duplex (Census) (%)': 'Duplex_Apt_Pct',
    
    # Housing age/period fields - percentages
    '2021 Period of Construction - 1960 or Before (Census) (%)': 'Old_Housing_Pct',
    '2021 Period of Construction - 1961 to 1980 (Census) (%)': 'Housing_1961_1980_Pct',
    '2021 Period of Construction - 1981 to 1990 (Census) (%)': 'Housing_1981_1990_Pct',
    '2021 Period of Construction - 1991 to 2000 (Census) (%)': 'Housing_1991_2000_Pct',
    '2021 Period of Construction - 2001 to 2005 (Census) (%)': 'Housing_2001_2005_Pct',
    '2021 Period of Construction - 2006 to 2010 (Census) (%)': 'Housing_2006_2010_Pct',
    '2021 Period of Construction - 2011 to 2016 (Census) (%)': 'Housing_2011_2016_Pct',
    '2021 Period of Construction - 2016 to 2021 (Census) (%)': 'Housing_2016_2021_Pct',
    
    # Housing age/period fields - absolutes
    '2021 Period of Construction - 1960 or Before (Census)': 'Old_Housing',
    '2021 Period of Construction - 1961 to 1980 (Census)': 'Housing_1961_1980',
    '2021 Period of Construction - 1981 to 1990 (Census)': 'Housing_1981_1990',
    '2021 Period of Construction - 1991 to 2000 (Census)': 'Housing_1991_2000',
    '2021 Period of Construction - 2001 to 2005 (Census)': 'Housing_2001_2005',
    '2021 Period of Construction - 2006 to 2010 (Census)': 'Housing_2006_2010',
    '2021 Period of Construction - 2011 to 2016 (Census)': 'Housing_2011_2016',
    '2021 Period of Construction - 2016 to 2021 (Census)': 'Housing_2016_2021',
    
    # Economic change indicators - recent
    '2023-2024 Total Population % Change': 'Recent_Population_Change',
    '2023-2024 Current$ Household Average Income % Change': 'Recent_Income_Change',
    '2022-2023 Total Population % Change': 'Prior_Year_Population_Change',
    '2022-2023 Current$ Household Average Income % Change': 'Prior_Year_Income_Change',
    '2021-2022 Total Population % Change': 'Two_Year_Prior_Population_Change',
    
    # Economic change indicators - future projections
    '2024-2025 Total Population % Change': 'Next_Year_Population_Change_Projection',
    '2024-2025 Current$ Household Average Income % Change': 'Next_Year_Income_Change_Projection',
    '2025-2026 Total Population % Change': 'Two_Year_Population_Change_Projection',
    '2025-2026 Current$ Household Average Income % Change': 'Two_Year_Income_Change_Projection',
    '2026-2027 Total Population % Change': 'Three_Year_Population_Change_Projection',
    '2026-2027 Current$ Household Average Income % Change': 'Three_Year_Income_Change_Projection',
    
    # Marital status demographics
    '2024 Pop 15+: Married (And Not Separated)': 'Married_Population',
    '2024 Pop 15+: Married (And Not Separated) (%)': 'Married_Population_Pct',
    '2024 Pop 15+: Single (Never Legally Married)': 'Single_Population',
    '2024 Pop 15+: Single (Never Legally Married) (%)': 'Single_Population_Pct',
    '2024 Pop 15+: Divorced': 'Divorced_Population',
    '2024 Pop 15+: Divorced (%)': 'Divorced_Population_Pct',
    '2024 Pop 15+: Separated': 'Separated_Population',
    '2024 Pop 15+: Separated (%)': 'Separated_Population_Pct',
    '2024 Pop 15+: Widowed': 'Widowed_Population',
    '2024 Pop 15+: Widowed (%)': 'Widowed_Population_Pct',
    '2024 Pop 15+: Living Common Law': 'Common_Law_Population',
    '2024 Pop 15+: Living Common Law (%)': 'Common_Law_Population_Pct',
    '2024 Pop 15+: Married or Living Common-Law': 'Combined_Married_Population',
    '2024 Pop 15+: Married or Living Common-Law (%)': 'Combined_Married_Population_Pct',
    '2024 Pop 15+: Not Married or Common-Law': 'Not_Married_Population',
    '2024 Pop 15+: Not Married or Common-Law (%)': 'Not_Married_Population_Pct',
    
    # Financial metrics - housing costs
    '2024 Property Taxes (Shelter)': 'Property_Tax_Total',
    '2024 Property Taxes (Shelter) (Avg)': 'Avg_Property_Tax',
    '2024 Regular Mortgage Payments (Shelter)': 'Mortgage_Payments_Total',
    '2024 Regular Mortgage Payments (Shelter) (Avg)': 'Avg_Mortgage_Payment',
    '2024 Condominium Charges (Shelter)': 'Condo_Fees_Total',
    '2024 Condominium Charges (Shelter) (Avg)': 'Avg_Condo_Fees',
    
    # Financial metrics - income and assets
    '2024 Household Aggregate Income': 'Total_Income',
    '2024 Household Aggregate Income (Current Year $)': 'Current_Year_Total_Income',
    '2024 Household Discretionary Aggregate Income': 'Discretionary_Income',
    '2024 Household Disposable Aggregate Income': 'Disposable_Income',
    
    # Banking and financial services
    '2024 Financial Services': 'Financial_Services_Total',
    '2024 Financial Services (Avg)': 'Avg_Financial_Services',
    '2024 Service Charges for Banks, Other Financial Institutions': 'Bank_Service_Charges_Total',
    '2024 Service Charges for Banks, Other Financial Institutions (Avg)': 'Avg_Bank_Service_Charges',
    
    # Mortgage data
    'FREQUENCY': 'Mortgage_Applications',
    'SUM_FUNDED': 'Mortgage_Approvals'
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
        # Check if the input file exists
        if not os.path.exists(input_path):
            logger.error(f"Input file not found at {input_path}")
            raise FileNotFoundError(f"Input file not found at {input_path}")
        
        # Read the input CSV file
        df = pd.read_csv(input_path, low_memory=False)  # Added low_memory=False to avoid dtype warning
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