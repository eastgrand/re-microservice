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

# The keys in this dictionary MUST BE LOWERCASE to match the logic in app.py
FIELD_MAPPINGS = {
    'conversion_rate': 'CONVERSION_RATE',
    'zip_code': 'Forward Sortation Area',
    'province_code': 'ID',
    'object_id': 'Object ID',
    'geographic_area': 'Shape__Area',
    'geographic_length': 'Shape__Length',
    'area_size': 'LANDAREA',
    'population': '2024 Total Population',
    'total_household_types': '2024 Household Type - Total Households',
    'age': '2024 Maintainers - Median Age',
    'female_population': '2024 Female Household Population',
    'female_population_pct': '2024 Female Household Population (%)',
    'male_population': '2024 Male Household Population',
    'male_population_pct': '2024 Male Household Population (%)',
    'young_adult_maintainers': '2024 Maintainers - 25 to 34',
    'young_adult_maintainers_pct': '2024 Maintainers - 25 to 34 (%)',
    'middle_age_maintainers': '2024 Maintainers - 35 to 44',
    'middle_age_maintainers_pct': '2024 Maintainers - 35 to 44 (%)',
    'mature_maintainers': '2024 Maintainers - 45 to 54',
    'mature_maintainers_pct': '2024 Maintainers - 45 to 54 (%)',
    'senior_maintainers': '2024 Maintainers - 55 to 64',
    'senior_maintainers_pct': '2024 Maintainers - 55 to 64 (%)',
    'homeownership': '2024 Tenure: Owned',
    'homeownership_pct': '2024 Tenure: Owned (%)',
    'rental_units': '2024 Tenure: Rented',
    'rental_units_pct': '2024 Tenure: Rented (%)',
    'band_housing': '2024 Tenure: Band Housing',
    'band_housing_pct': '2024 Tenure: Band Housing (%)',
    'households': '2024 Tenure: Total Households',
    'african_american_population': '2024 Visible Minority Black',
    'african_american_population_pct': '2024 Visible Minority Black (%)',
    'asian_population': '2024 Visible Minority Chinese',
    'asian_population_pct': '2024 Visible Minority Chinese (%)',
    'latin_american_population': '2024 Visible Minority Latin American',
    'latin_american_population_pct': '2024 Visible Minority Latin American (%)',
    'arab_population': '2024 Visible Minority Arab',
    'arab_population_pct': '2024 Visible Minority Arab (%)',
    'filipino_population': '2024 Visible Minority Filipino',
    'filipino_population_pct': '2024 Visible Minority Filipino (%)',
    'south_asian_population': '2024 Visible Minority South Asian',
    'south_asian_population_pct': '2024 Visible Minority South Asian (%)',
    'southeast_asian_population': '2024 Visible Minority Southeast Asian',
    'southeast_asian_population_pct': '2024 Visible Minority Southeast Asian (%)',
    'japanese_population': '2024 Visible Minority Japanese',
    'japanese_population_pct': '2024 Visible Minority Japanese (%)',
    'korean_population': '2024 Visible Minority Korean',
    'korean_population_pct': '2024 Visible Minority Korean (%)',
    'west_asian_population': '2024 Visible Minority West Asian',
    'west_asian_population_pct': '2024 Visible Minority West Asian (%)',
    'total_minority_population': '2024 Visible Minority Total Population',
    'total_minority_population_pct': '2024 Visible Minority Total Population (%)',
    'labor_participation_rate': '2024 Labour Force - Labour Participation Rate',
    'employment_rate': '2024 Labour Force - Labour Employment Rate',
    'unemployment_rate': '2024 Labour Force - Labour Unemployment Rate',
    'condo_units': '2024 Condominium Status - In Condo',
    'condo_ownership_pct': '2024 Condominium Status - In Condo (%)',
    'non_condo_units': '2024 Condominium Status - Not In Condo',
    'non_condo_pct': '2024 Condominium Status - Not In Condo (%)',
    'total_condo_status_households': '2024 Condominium Status - Total Households',
    'single_detached_houses': '2024 Structure Type Single-Detached House',
    'single_detached_house_pct': '2024 Structure Type Single-Detached House (%)',
    'semi_detached_houses': '2024 Structure Type Semi-Detached House',
    'semi_detached_house_pct': '2024 Structure Type Semi-Detached House (%)',
    'row_houses': '2024 Structure Type Row House',
    'row_house_pct': '2024 Structure Type Row House (%)',
    'large_apartments': '2024 Structure Type Apartment, Building Five or More Story',
    'large_apartment_pct': '2024 Structure Type Apartment, Building Five or More Story (%)',
    'small_apartments': '2024 Structure Type Apartment, Building Fewer Than Five Story',
    'small_apartment_pct': '2024 Structure Type Apartment, Building Fewer Than Five Story (%)',
    'movable_dwellings': '2024 Structure Type Movable Dwelling',
    'movable_dwelling_pct': '2024 Structure Type Movable Dwelling (%)',
    'other_single_attached_houses': '2024 Structure Type Other Single-Attached House',
    'other_attached_house_pct': '2024 Structure Type Other Single-Attached House (%)',
    'duplex_apartments': '2021 Housing: Apartment or Flat in Duplex (Census)',
    'duplex_apt_pct': '2021 Housing: Apartment or Flat in Duplex (Census) (%)',
    'old_housing': '2021 Period of Construction - 1960 or Before (Census)',
    'old_housing_pct': '2021 Period of Construction - 1960 or Before (Census) (%)',
    'housing_1961_1980': '2021 Period of Construction - 1961 to 1980 (Census)',
    'housing_1961_1980_pct': '2021 Period of Construction - 1961 to 1980 (Census) (%)',
    'housing_1981_1990': '2021 Period of Construction - 1981 to 1990 (Census)',
    'housing_1981_1990_pct': '2021 Period of Construction - 1981 to 1990 (Census) (%)',
    'housing_1991_2000': '2021 Period of Construction - 1991 to 2000 (Census)',
    'housing_1991_2000_pct': '2021 Period of Construction - 1991 to 2000 (Census) (%)',
    'housing_2001_2005': '2021 Period of Construction - 2001 to 2005 (Census)',
    'housing_2001_2005_pct': '2021 Period of Construction - 2001 to 2005 (Census) (%)',
    'housing_2006_2010': '2021 Period of Construction - 2006 to 2010 (Census)',
    'housing_2006_2010_pct': '2021 Period of Construction - 2006 to 2010 (Census) (%)',
    'housing_2011_2016': '2021 Period of Construction - 2011 to 2016 (Census)',
    'housing_2011_2016_pct': '2021 Period of Construction - 2011 to 2016 (Census) (%)',
    'housing_2016_2021': '2021 Period of Construction - 2016 to 2021 (Census)',
    'housing_2016_2021_pct': '2021 Period of Construction - 2016 to 2021 (Census) (%)',
    'recent_population_change': '2023-2024 Total Population % Change',
    'recent_income_change': '2023-2024 Current$ Household Average Income % Change',
    'prior_year_population_change': '2022-2023 Total Population % Change',
    'prior_year_income_change': '2022-2023 Current$ Household Average Income % Change',
    'two_year_prior_population_change': '2021-2022 Total Population % Change',
    'next_year_population_change_projection': '2024-2025 Total Population % Change',
    'next_year_income_change_projection': '2024-2025 Current$ Household Average Income % Change',
    'two_year_population_change_projection': '2025-2026 Total Population % Change',
    'two_year_income_change_projection': '2025-2026 Current$ Household Average Income % Change',
    'three_year_population_change_projection': '2026-2027 Total Population % Change',
    'three_year_income_change_projection': '2026-2027 Current$ Household Average Income % Change',
    'married_population': '2024 Pop 15+: Married (And Not Separated)',
    'married_population_pct': '2024 Pop 15+: Married (And Not Separated) (%)',
    'single_population': '2024 Pop 15+: Single (Never Legally Married)',
    'single_population_pct': '2024 Pop 15+: Single (Never Legally Married) (%)',
    'divorced_population': '2024 Pop 15+: Divorced',
    'divorced_population_pct': '2024 Pop 15+: Divorced (%)',
    'separated_population': '2024 Pop 15+: Separated',
    'separated_population_pct': '2024 Pop 15+: Separated (%)',
    'widowed_population': '2024 Pop 15+: Widowed',
    'widowed_population_pct': '2024 Pop 15+: Widowed (%)',
    'common_law_population': '2024 Pop 15+: Living Common Law',
    'common_law_population_pct': '2024 Pop 15+: Living Common Law (%)',
    'combined_married_population': '2024 Pop 15+: Married or Living Common-Law',
    'combined_married_population_pct': '2024 Pop 15+: Married or Living Common-Law (%)',
    'not_married_population': '2024 Pop 15+: Not Married or Common-Law',
    'not_married_population_pct': '2024 Pop 15+: Not Married or Common-Law (%)',
    'income': '2024 Household Average Income (Current Year $)',
    'median_income': '2024 Household Median Income (Current Year $)',
    'property_tax_total': '2024 Property Taxes (Shelter)',
    'avg_property_tax': '2024 Property Taxes (Shelter) (Avg)',
    'mortgage_payments_total': '2024 Regular Mortgage Payments (Shelter)',
    'avg_mortgage_payment': '2024 Regular Mortgage Payments (Shelter) (Avg)',
    'condo_fees_total': '2024 Condominium Charges (Shelter)',
    'avg_condo_fees': '2024 Condominium Charges (Shelter) (Avg)',
    'total_income': '2024 Household Aggregate Income',
    'current_year_total_income': '2024 Household Aggregate Income (Current Year $)',
    'discretionary_income': '2024 Household Discretionary Aggregate Income',
    'disposable_income': '2024 Household Disposable Aggregate Income',
    'financial_services_total': '2024 Financial Services',
    'avg_financial_services': '2024 Financial Services (Avg)',
    'bank_service_charges_total': '2024 Service Charges for Banks, Other Financial Institutions',
    'avg_bank_service_charges': '2024 Service Charges for Banks, Other Financial Institutions (Avg)',
    'mortgage_applications': 'Mortgage Applicationns',
    'mortgage_approvals': 'Funded Applications',
    'single_status': 'SUM_ECYMARNMCL',
    'single_family_homes': 'SUM_ECYSTYSING',
    'aggregate_income': 'SUM_HSHNIAGG',
    'market_weight': 'Sum_Weight'
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