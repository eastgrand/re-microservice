import pandas as pd
import requests
import json
import logging
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_arcgis_data(url: str) -> List[Dict]:
    """
    Fetch data from an ArcGIS feature service.
    
    Args:
        url (str): The URL of the ArcGIS feature service
        
    Returns:
        List[Dict]: List of features from the service
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get('features', [])
    except Exception as e:
        logger.error(f"Error fetching data from {url}: {str(e)}")
        return []

def process_features(features: List[Dict]) -> pd.DataFrame:
    """
    Convert ArcGIS features to a pandas DataFrame.
    
    Args:
        features (List[Dict]): List of features from ArcGIS service
        
    Returns:
        pd.DataFrame: Processed data as a DataFrame
    """
    if not features:
        return pd.DataFrame()
    
    # Extract attributes from features
    data = [feature.get('attributes', {}) for feature in features]
    return pd.DataFrame(data)

def main():
    # Read layer configurations
    try:
        layers_df = pd.read_csv('nesto.csv')
    except Exception as e:
        logger.error(f"Error reading nesto.csv: {str(e)}")
        return
    
    # Process each layer
    all_data = []
    for _, row in layers_df.iterrows():
        layer_name = row['Layer Name']
        url = row['URL']
        
        logger.info(f"Processing layer: {layer_name}")
        features = fetch_arcgis_data(url)
        df = process_features(features)
        
        if not df.empty:
            # Validate ID field exists
            if 'ID' not in df.columns:
                logger.error(f"Layer {layer_name} missing required ID field")
                continue
                
            # Validate ID field is unique
            if df['ID'].duplicated().any():
                logger.error(f"Layer {layer_name} has duplicate IDs")
                continue
                
            # Add layer name as a prefix to columns except ID
            rename_dict = {col: f"{layer_name}_{col}" for col in df.columns if col != 'ID'}
            df = df.rename(columns=rename_dict)
            all_data.append(df)
    
    # Merge all data using ID as join key
    if all_data:
        # Start with first dataset
        final_df = all_data[0]
        logger.info(f"Starting merge with {len(final_df)} rows from first layer")
        
        # Join subsequent datasets
        for i, df in enumerate(all_data[1:], 1):
            logger.info(f"Merging layer {i+1} with {len(df)} rows")
            final_df = final_df.merge(df, on='ID', how='inner')
            logger.info(f"After merge: {len(final_df)} rows")
            
            # Validate merge
            if len(final_df) == 0:
                logger.error("Merge resulted in empty dataset - no matching IDs")
                return
                
        # Validate final dataset
        logger.info(f"Final dataset has {len(final_df)} rows and {len(final_df.columns)} columns")
        if len(final_df) > 2000:  # Sanity check
            logger.warning(f"Final dataset has more rows than expected: {len(final_df)}")
            
        # Save merged data
        output_path = 'data/nesto_merge_0.csv'
        final_df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
    else:
        logger.warning("No data was collected from any layer")

if __name__ == "__main__":
    main() 