#!/usr/bin/env python3
"""
Setup script to prepare the environment for deployment to Render.com.
This script ensures all data files and models are correctly prepared.
"""

import os
import sys
import logging
import subprocess
import pandas as pd
import numpy as np
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("setup-for-render")

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
        logger.warning(f"Nesto data file not found at {nesto_data_path}")
        
        # Create a comprehensive sample data file for deployment with all mapped fields
        logger.info("Creating comprehensive sample data file with all mapped fields")
        
        # Base demographic values
        total_pop = np.random.randint(1000, 50000, 100)
        female_pop = np.random.randint(500, 25000, 100)
        male_pop = total_pop - female_pop
        
        # Housing tenure values
        owned_dwellings = np.random.randint(100, 5000, 100)
        rented_dwellings = np.random.randint(50, 3000, 100)
        band_housing = np.random.randint(0, 200, 100)
        total_households = owned_dwellings + rented_dwellings + band_housing
        
        # Minority populations
        total_minority = np.random.randint(0, 5000, 100)
        black_minority = np.random.randint(0, 1000, 100)
        chinese_minority = np.random.randint(0, 1000, 100)
        latin_minority = np.random.randint(0, 1000, 100)
        arab_minority = np.random.randint(0, 800, 100)
        filipino_minority = np.random.randint(0, 600, 100)
        south_asian_minority = np.random.randint(0, 1200, 100)
        southeast_asian_minority = np.random.randint(0, 500, 100)
        japanese_minority = np.random.randint(0, 200, 100)
        korean_minority = np.random.randint(0, 300, 100)
        west_asian_minority = np.random.randint(0, 400, 100)
        
        # Housing types
        condo_units = np.random.randint(50, 2000, 100)
        non_condo_units = total_households - condo_units
        single_detached = np.random.randint(200, 3000, 100)
        semi_detached = np.random.randint(50, 1000, 100)
        row_houses = np.random.randint(50, 1000, 100)
        large_apartments = np.random.randint(50, 2000, 100)
        small_apartments = np.random.randint(100, 2000, 100)
        movable_dwellings = np.random.randint(0, 200, 100)
        other_attached_houses = np.random.randint(0, 500, 100)
        duplex_apartments = np.random.randint(50, 1000, 100)
        
        # Housing age/period values
        old_housing = np.random.randint(50, 1500, 100)
        housing_1961_1980 = np.random.randint(100, 2000, 100)
        housing_1981_1990 = np.random.randint(50, 1000, 100)
        housing_1991_2000 = np.random.randint(50, 1000, 100)
        housing_2001_2005 = np.random.randint(20, 800, 100)
        housing_2006_2010 = np.random.randint(20, 800, 100)
        housing_2011_2016 = np.random.randint(20, 600, 100)
        housing_2016_2021 = np.random.randint(10, 500, 100)
        total_housing = (old_housing + housing_1961_1980 + housing_1981_1990 + 
                        housing_1991_2000 + housing_2001_2005 + housing_2006_2010 + 
                        housing_2011_2016 + housing_2016_2021)
        
        # Marital status demographics
        married_pop = np.random.randint(500, 20000, 100)
        common_law_pop = np.random.randint(100, 5000, 100)
        single_pop = np.random.randint(300, 15000, 100)
        divorced_pop = np.random.randint(50, 3000, 100)
        separated_pop = np.random.randint(20, 2000, 100)
        widowed_pop = np.random.randint(20, 3000, 100)
        combined_married = married_pop + common_law_pop
        not_married = single_pop + divorced_pop + separated_pop + widowed_pop
        adult_pop = combined_married + not_married  # Population 15+
        
        # Age group values for household maintainers
        young_adult_maintainers = np.random.randint(100, 1000, 100)
        middle_age_maintainers = np.random.randint(200, 1500, 100)
        mature_maintainers = np.random.randint(150, 1200, 100)
        senior_maintainers = np.random.randint(150, 1200, 100)
        total_maintainers = young_adult_maintainers + middle_age_maintainers + mature_maintainers + senior_maintainers
        
        # Financial metrics
        property_tax_total = np.random.uniform(1e6, 1e7, 100)
        mortgage_payments_total = np.random.uniform(5e6, 2e7, 100)
        condo_fees_total = np.random.uniform(2e5, 2e6, 100)
        financial_services_total = np.random.uniform(1e6, 5e6, 100)
        bank_charges_total = np.random.uniform(2e5, 1e6, 100)
        
        # Income metrics
        avg_income = np.random.normal(75000, 15000, 100)
        median_income = np.random.normal(68000, 12000, 100)
        total_income = avg_income * total_households
        discretionary_income = total_income * np.random.uniform(0.3, 0.5, 100)
        disposable_income = total_income * np.random.uniform(0.6, 0.8, 100)
        
        # Calculate percentage fields based on absolute values
        sample_data = pd.DataFrame({
            # Geographic fields
            'Object ID': np.arange(1001, 1101),
            'Forward Sortation Area': [f'A{i}A' for i in range(100)],
            'ID': [f'AB{i:02d}' for i in range(100)],
            'Shape__Area': np.random.uniform(1000000, 10000000, 100),
            'Shape__Length': np.random.uniform(10000, 100000, 100),
            
            # Basic demographic fields
            '2024 Total Population': total_pop,
            '2024 Household Type - Total Households': total_households,
            '2024 Household Average Income (Current Year $)': avg_income,
            '2024 Household Median Income (Current Year $)': median_income,
            '2024 Maintainers - Median Age': np.random.normal(42, 8, 100),
            '2024 Female Household Population': female_pop,
            '2024 Female Household Population (%)': (female_pop / total_pop * 100).round(2),
            '2024 Male Household Population': male_pop,
            '2024 Male Household Population (%)': (male_pop / total_pop * 100).round(2),
            
            # Age groups and household maintainers
            '2024 Maintainers - 25 to 34': young_adult_maintainers,
            '2024 Maintainers - 25 to 34 (%)': (young_adult_maintainers / total_maintainers * 100).round(2),
            '2024 Maintainers - 35 to 44': middle_age_maintainers,
            '2024 Maintainers - 35 to 44 (%)': (middle_age_maintainers / total_maintainers * 100).round(2),
            '2024 Maintainers - 45 to 54': mature_maintainers,
            '2024 Maintainers - 45 to 54 (%)': (mature_maintainers / total_maintainers * 100).round(2),
            '2024 Maintainers - 55 to 64': senior_maintainers,
            '2024 Maintainers - 55 to 64 (%)': (senior_maintainers / total_maintainers * 100).round(2),
            
            # Housing tenure fields
            '2024 Tenure: Owned': owned_dwellings,
            '2024 Tenure: Owned (%)': (owned_dwellings / total_households * 100).round(2),
            '2024 Tenure: Rented': rented_dwellings,
            '2024 Tenure: Rented (%)': (rented_dwellings / total_households * 100).round(2),
            '2024 Tenure: Band Housing': band_housing,
            '2024 Tenure: Band Housing (%)': (band_housing / total_households * 100).round(2),
            '2024 Tenure: Total Households': total_households,
            
            # Visible minority demographics
            '2024 Visible Minority Black': black_minority,
            '2024 Visible Minority Black (%)': (black_minority / total_pop * 100).round(2),
            '2024 Visible Minority Chinese': chinese_minority,
            '2024 Visible Minority Chinese (%)': (chinese_minority / total_pop * 100).round(2),
            '2024 Visible Minority Latin American': latin_minority,
            '2024 Visible Minority Latin American (%)': (latin_minority / total_pop * 100).round(2),
            '2024 Visible Minority Arab': arab_minority,
            '2024 Visible Minority Arab (%)': (arab_minority / total_pop * 100).round(2),
            '2024 Visible Minority Filipino': filipino_minority,
            '2024 Visible Minority Filipino (%)': (filipino_minority / total_pop * 100).round(2),
            '2024 Visible Minority South Asian': south_asian_minority,
            '2024 Visible Minority South Asian (%)': (south_asian_minority / total_pop * 100).round(2),
            '2024 Visible Minority Southeast Asian': southeast_asian_minority,
            '2024 Visible Minority Southeast Asian (%)': (southeast_asian_minority / total_pop * 100).round(2),
            '2024 Visible Minority Japanese': japanese_minority,
            '2024 Visible Minority Japanese (%)': (japanese_minority / total_pop * 100).round(2),
            '2024 Visible Minority Korean': korean_minority,
            '2024 Visible Minority Korean (%)': (korean_minority / total_pop * 100).round(2),
            '2024 Visible Minority West Asian': west_asian_minority,
            '2024 Visible Minority West Asian (%)': (west_asian_minority / total_pop * 100).round(2),
            '2024 Visible Minority Total Population': total_minority,
            '2024 Visible Minority Total Population (%)': (total_minority / total_pop * 100).round(2),
            
            # Labor market indicators
            '2024 Labour Force - Labour Participation Rate': np.random.normal(65, 10, 100),
            '2024 Labour Force - Labour Employment Rate': np.random.normal(60, 10, 100),
            '2024 Labour Force - Labour Unemployment Rate': np.random.normal(5, 2, 100),
            
            # Housing type fields
            '2024 Condominium Status - In Condo': condo_units,
            '2024 Condominium Status - In Condo (%)': (condo_units / total_households * 100).round(2),
            '2024 Condominium Status - Not In Condo': non_condo_units,
            '2024 Condominium Status - Not In Condo (%)': (non_condo_units / total_households * 100).round(2),
            '2024 Condominium Status - Total Households': total_households,
            '2024 Structure Type Single-Detached House': single_detached,
            '2024 Structure Type Single-Detached House (%)': (single_detached / total_households * 100).round(2),
            '2024 Structure Type Semi-Detached House': semi_detached,
            '2024 Structure Type Semi-Detached House (%)': (semi_detached / total_households * 100).round(2),
            '2024 Structure Type Row House': row_houses,
            '2024 Structure Type Row House (%)': (row_houses / total_households * 100).round(2),
            '2024 Structure Type Apartment, Building Five or More Story': large_apartments,
            '2024 Structure Type Apartment, Building Five or More Story (%)': (large_apartments / total_households * 100).round(2),
            '2024 Structure Type Apartment, Building Fewer Than Five Story': small_apartments,
            '2024 Structure Type Apartment, Building Fewer Than Five Story (%)': (small_apartments / total_households * 100).round(2),
            '2024 Structure Type Movable Dwelling': movable_dwellings,
            '2024 Structure Type Movable Dwelling (%)': (movable_dwellings / total_households * 100).round(2),
            '2024 Structure Type Other Single-Attached House': other_attached_houses,
            '2024 Structure Type Other Single-Attached House (%)': (other_attached_houses / total_households * 100).round(2),
            '2021 Housing: Apartment or Flat in Duplex (Census)': duplex_apartments,
            '2021 Housing: Apartment or Flat in Duplex (Census) (%)': (duplex_apartments / total_households * 100).round(2),
            
            # Housing age/period fields - percentages and absolutes
            '2021 Period of Construction - 1960 or Before (Census)': old_housing,
            '2021 Period of Construction - 1960 or Before (Census) (%)': (old_housing / total_housing * 100).round(2),
            '2021 Period of Construction - 1961 to 1980 (Census)': housing_1961_1980,
            '2021 Period of Construction - 1961 to 1980 (Census) (%)': (housing_1961_1980 / total_housing * 100).round(2),
            '2021 Period of Construction - 1981 to 1990 (Census)': housing_1981_1990,
            '2021 Period of Construction - 1981 to 1990 (Census) (%)': (housing_1981_1990 / total_housing * 100).round(2),
            '2021 Period of Construction - 1991 to 2000 (Census)': housing_1991_2000,
            '2021 Period of Construction - 1991 to 2000 (Census) (%)': (housing_1991_2000 / total_housing * 100).round(2),
            '2021 Period of Construction - 2001 to 2005 (Census)': housing_2001_2005,
            '2021 Period of Construction - 2001 to 2005 (Census) (%)': (housing_2001_2005 / total_housing * 100).round(2),
            '2021 Period of Construction - 2006 to 2010 (Census)': housing_2006_2010,
            '2021 Period of Construction - 2006 to 2010 (Census) (%)': (housing_2006_2010 / total_housing * 100).round(2),
            '2021 Period of Construction - 2011 to 2016 (Census)': housing_2011_2016,
            '2021 Period of Construction - 2011 to 2016 (Census) (%)': (housing_2011_2016 / total_housing * 100).round(2),
            '2021 Period of Construction - 2016 to 2021 (Census)': housing_2016_2021,
            '2021 Period of Construction - 2016 to 2021 (Census) (%)': (housing_2016_2021 / total_housing * 100).round(2),
            
            # Economic change indicators - recent and projected
            '2021-2022 Total Population % Change': np.random.uniform(-3, 4, 100).round(2),
            '2022-2023 Total Population % Change': np.random.uniform(-2.5, 4.5, 100).round(2),
            '2022-2023 Current$ Household Average Income % Change': np.random.uniform(-1.5, 7.5, 100).round(2),
            '2023-2024 Total Population % Change': np.random.uniform(-2, 5, 100).round(2),
            '2023-2024 Current$ Household Average Income % Change': np.random.uniform(-1, 8, 100).round(2),
            '2024-2025 Total Population % Change': np.random.uniform(-2, 5.5, 100).round(2),
            '2024-2025 Current$ Household Average Income % Change': np.random.uniform(-0.5, 7, 100).round(2),
            '2025-2026 Total Population % Change': np.random.uniform(-1.5, 6, 100).round(2),
            '2025-2026 Current$ Household Average Income % Change': np.random.uniform(-0.5, 7.5, 100).round(2),
            '2026-2027 Total Population % Change': np.random.uniform(-1, 6.5, 100).round(2),
            '2026-2027 Current$ Household Average Income % Change': np.random.uniform(0, 8, 100).round(2),
            
            # Marital status demographics
            '2024 Pop 15+: Married (And Not Separated)': married_pop,
            '2024 Pop 15+: Married (And Not Separated) (%)': (married_pop / adult_pop * 100).round(2),
            '2024 Pop 15+: Single (Never Legally Married)': single_pop,
            '2024 Pop 15+: Single (Never Legally Married) (%)': (single_pop / adult_pop * 100).round(2),
            '2024 Pop 15+: Divorced': divorced_pop,
            '2024 Pop 15+: Divorced (%)': (divorced_pop / adult_pop * 100).round(2),
            '2024 Pop 15+: Separated': separated_pop,
            '2024 Pop 15+: Separated (%)': (separated_pop / adult_pop * 100).round(2),
            '2024 Pop 15+: Widowed': widowed_pop,
            '2024 Pop 15+: Widowed (%)': (widowed_pop / adult_pop * 100).round(2),
            '2024 Pop 15+: Living Common Law': common_law_pop,
            '2024 Pop 15+: Living Common Law (%)': (common_law_pop / adult_pop * 100).round(2),
            '2024 Pop 15+: Married or Living Common-Law': combined_married,
            '2024 Pop 15+: Married or Living Common-Law (%)': (combined_married / adult_pop * 100).round(2),
            '2024 Pop 15+: Not Married or Common-Law': not_married,
            '2024 Pop 15+: Not Married or Common-Law (%)': (not_married / adult_pop * 100).round(2),
            
            # Financial metrics - housing costs
            '2024 Property Taxes (Shelter)': property_tax_total,
            '2024 Property Taxes (Shelter) (Avg)': (property_tax_total / total_households).round(2),
            '2024 Regular Mortgage Payments (Shelter)': mortgage_payments_total,
            '2024 Regular Mortgage Payments (Shelter) (Avg)': (mortgage_payments_total / owned_dwellings).round(2),
            '2024 Condominium Charges (Shelter)': condo_fees_total,
            '2024 Condominium Charges (Shelter) (Avg)': (condo_fees_total / condo_units).round(2),
            
            # Financial metrics - income and assets
            '2024 Household Aggregate Income': total_income,
            '2024 Household Aggregate Income (Current Year $)': total_income,
            '2024 Household Discretionary Aggregate Income': discretionary_income,
            '2024 Household Disposable Aggregate Income': disposable_income,
            
            # Banking and financial services
            '2024 Financial Services': financial_services_total,
            '2024 Financial Services (Avg)': (financial_services_total / total_households).round(2),
            '2024 Service Charges for Banks, Other Financial Institutions': bank_charges_total,
            '2024 Service Charges for Banks, Other Financial Institutions (Avg)': (bank_charges_total / total_households).round(2),
            
            # Mortgage data
            'Mortgage Applicationns': np.random.randint(20, 300, 100),
            'Funded Applications': np.random.randint(10, 200, 100)
        })
        
        sample_data.to_csv(nesto_data_path, index=False)
        logger.info(f"Created comprehensive sample data with all mapped fields at {nesto_data_path}")
    else:
        logger.info(f"Found Nesto data file at {nesto_data_path}")
    
    # Step 3: Run the data mapping script
    try:
        logger.info("Mapping data fields...")
        from map_nesto_data import map_nesto_data
        mapped_data = map_nesto_data(nesto_data_path, 'data/cleaned_data.csv')
        logger.info("Data mapping completed successfully")
    except Exception as e:
        logger.error(f"Error during data mapping: {e}")
        return False
    
    # Step 4: Run the model training script
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
