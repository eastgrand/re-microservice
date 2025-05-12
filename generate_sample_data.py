# filepath: /Users/voldeck/code/shap-microservice/generate_sample_data.py
import pandas as pd
import numpy as np
import os

def generate_sample_data(output_file='data/sample_data.csv', n_records=1000):
    """Generate sample sales data and save to CSV file"""
    print("Generating sample sales data for SHAP microservice...")
    
    # Set random seed for reproducibility
    np.random.seed(42)

    # Make sure the data directory exists
    os.makedirs('data', exist_ok=True)

    # Create zip codes
    zip_codes = np.random.randint(10000, 99999, size=n_records)

    # Create demographic data with correlations
    # Income influences Nike sales positively
    income = np.random.normal(75000, 25000, n_records)

    # Age influences Nike sales - higher sales in younger demographics
    age = np.random.normal(35, 10, n_records)

    # Population demographics
    hispanic_population = np.random.normal(15000, 5000, n_records)
    african_american_population = np.random.normal(12000, 4000, n_records)
    asian_population = np.random.normal(9000, 3000, n_records)

    # Geographic coordinates (simple approximation for US locations)
    latitude = np.random.uniform(24, 49, n_records)  # Continental US latitudes
    longitude = np.random.uniform(-125, -66, n_records)  # Continental US longitudes

    # Create Nike sales with correlations to demographics
    # Base value
    nike_sales = np.random.normal(100000, 30000, n_records)

    # Apply correlations
    # Income positively affects sales
    nike_sales += 0.3 * (income - 75000) / 10000
    # Younger people buy more Nike
    nike_sales += -0.2 * (age - 35) * 1000
    # Higher Hispanic population increases sales (as an example correlation)
    nike_sales += 0.15 * (hispanic_population - 15000) / 1000
    # Higher African American population increases sales (as an example correlation)
    nike_sales += 0.1 * (african_american_population - 12000) / 1000

    # Make sure sales are positive
    nike_sales = np.maximum(nike_sales, 10000)

    # Create a DataFrame
    df = pd.DataFrame({
        'zip_code': zip_codes,
        'Income': income,
        'Age': age,
        'Hispanic_Population': hispanic_population,
        'African_American_Population': african_american_population,
        'Asian_Population': asian_population,
        'latitude': latitude,
        'longitude': longitude,
        'Nike_Sales': nike_sales
    })

    # Round numeric columns for readability
    df = df.round({
        'Income': 0,
        'Age': 1,
        'Hispanic_Population': 0,
        'African_American_Population': 0,
        'Asian_Population': 0,
        'latitude': 4,
        'longitude': 4,
        'Nike_Sales': 0
    })

    # Convert zip code to string format
    df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)

    # Save to CSV
    df.to_csv(output_file, index=False)

    print(f"Sample data generated and saved to {output_file}")
    print(f"Generated {n_records} records with the following columns:")
    for column in df.columns:
        print(f"- {column}")
    print("\nSample data preview:")
    print(df.head(5))
    
    return df

# Execute when run directly
if __name__ == "__main__":
    generate_sample_data()
