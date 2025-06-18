#!/usr/bin/env python3
"""
ArcGIS Feature Service Data Extraction and Joining Script

This script automates the process of:
1. Extracting data from multiple ArcGIS feature service layers
2. Joining them by the 'ID' field (zip codes)
3. Converting the result to CSV format for the SHAP microservice

Usage:
    python extract_arcgis_data.py --config config/feature_services.json
    python extract_arcgis_data.py --url "https://your-service.com/rest/services" --layers "layer1,layer2"
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import requests
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("arcgis-extractor")

class ArcGISDataExtractor:
    """Extract and join data from ArcGIS feature services"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ArcGIS-Data-Extractor/1.0'
        })
    
    def get_layer_info(self, layer_id: str) -> Dict[str, Any]:
        """Get metadata about a specific layer"""
        url = f"{self.base_url}/{layer_id}"
        params = {'f': 'json'}
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get layer info for {layer_id}: {e}")
            raise
    
    def extract_layer_data(self, layer_id: str, where_clause: str = "1=1", 
                          max_records: int = 2000) -> pd.DataFrame:
        """Extract all data from a feature service layer"""
        logger.info(f"🔄 Extracting data from layer {layer_id}...")
        
        url = f"{self.base_url}/{layer_id}/query"
        
        # Get total record count first
        count_params = {
            'where': where_clause,
            'returnCountOnly': 'true',
            'f': 'json'
        }
        
        try:
            count_response = self.session.get(url, params=count_params, timeout=self.timeout)
            count_response.raise_for_status()
            total_records = count_response.json().get('count', 0)
            logger.info(f"📊 Layer {layer_id} has {total_records} records")
            
            all_features = []
            offset = 0
            
            while offset < total_records:
                params = {
                    'where': where_clause,
                    'outFields': '*',
                    'f': 'json',
                    'resultOffset': offset,
                    'resultRecordCount': max_records,
                    'returnGeometry': 'false'  # We don't need geometry for CSV
                }
                
                logger.info(f"📥 Fetching records {offset} to {offset + max_records}")
                
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                
                if 'features' not in data:
                    logger.warning(f"No features found in response for layer {layer_id}")
                    break
                
                features = data['features']
                if not features:
                    break
                
                # Extract attributes from each feature
                for feature in features:
                    if 'attributes' in feature:
                        all_features.append(feature['attributes'])
                
                offset += len(features)
                
                # Rate limiting
                time.sleep(0.1)
            
            if not all_features:
                logger.warning(f"No data extracted from layer {layer_id}")
                return pd.DataFrame()
            
            df = pd.DataFrame(all_features)
            logger.info(f"✅ Extracted {len(df)} records from layer {layer_id}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to extract data from layer {layer_id}: {e}")
            raise
    
    def join_layers_by_id(self, layer_dataframes: Dict[str, pd.DataFrame], 
                         id_field: str = 'ID') -> pd.DataFrame:
        """Join multiple layer dataframes by the ID field"""
        logger.info(f"🔗 Joining {len(layer_dataframes)} layers by '{id_field}' field...")
        
        if not layer_dataframes:
            raise ValueError("No dataframes to join")
        
        # Start with the first dataframe
        layer_names = list(layer_dataframes.keys())
        result_df = layer_dataframes[layer_names[0]].copy()
        
        if id_field not in result_df.columns:
            raise ValueError(f"ID field '{id_field}' not found in first layer")
        
        logger.info(f"📋 Starting with layer '{layer_names[0]}': {len(result_df)} records")
        
        # Join each subsequent layer
        for i, layer_name in enumerate(layer_names[1:], 1):
            df = layer_dataframes[layer_name]
            
            if id_field not in df.columns:
                logger.warning(f"ID field '{id_field}' not found in layer '{layer_name}', skipping...")
                continue
            
            # Add suffix to avoid column name conflicts
            suffix = f"_{layer_name}" if layer_name.isdigit() else f"_{layer_name}"
            
            # Perform left join to preserve all records from the base layer
            before_count = len(result_df)
            result_df = result_df.merge(
                df, 
                on=id_field, 
                how='left', 
                suffixes=('', suffix)
            )
            
            logger.info(f"🔗 Joined layer '{layer_name}': {before_count} → {len(result_df)} records")
        
        logger.info(f"✅ Final joined dataset: {len(result_df)} records, {len(result_df.columns)} columns")
        return result_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the data for SHAP analysis"""
        logger.info("🧹 Cleaning and preparing data...")
        
        # Remove completely empty columns
        before_cols = len(df.columns)
        df = df.dropna(axis=1, how='all')
        after_cols = len(df.columns)
        
        if before_cols != after_cols:
            logger.info(f"📉 Removed {before_cols - after_cols} empty columns")
        
        # Convert object columns that are actually numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if not numeric_series.isna().all():
                    # If more than 50% of values are numeric, convert the column
                    non_null_count = df[col].notna().sum()
                    numeric_count = numeric_series.notna().sum()
                    
                    if numeric_count / non_null_count > 0.5:
                        df[col] = numeric_series
                        logger.info(f"🔢 Converted column '{col}' to numeric")
        
        # Fill NaN values with appropriate defaults
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna('')
        
        logger.info(f"✅ Data cleaning complete: {len(df)} records, {len(df.columns)} columns")
        return df

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def save_to_csv(df: pd.DataFrame, output_path: str):
    """Save dataframe to CSV with proper formatting"""
    logger.info(f"💾 Saving data to {output_path}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    # Log summary
    logger.info(f"✅ Saved {len(df)} records with {len(df.columns)} columns to {output_path}")
    logger.info(f"📊 File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "base_url": "https://your-arcgis-server.com/arcgis/rest/services/YourService/MapServer",
        "layers": [
            {
                "id": "0",
                "name": "demographics",
                "description": "Demographic data by zip code"
            },
            {
                "id": "1", 
                "name": "economic",
                "description": "Economic indicators by zip code"
            },
            {
                "id": "2",
                "name": "housing",
                "description": "Housing statistics by zip code"
            }
        ],
        "id_field": "ID",
        "where_clause": "1=1",
        "output_file": "data/nesto_merge_0.csv",
        "max_records_per_request": 2000
    }
    
    config_path = "config/feature_services.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"📝 Created sample config at {config_path}")
    return config_path

def main():
    parser = argparse.ArgumentParser(description="Extract and join ArcGIS feature service data")
    parser.add_argument('--config', help='Path to JSON configuration file')
    parser.add_argument('--url', help='Base URL of the ArcGIS feature service')
    parser.add_argument('--layers', help='Comma-separated list of layer IDs')
    parser.add_argument('--output', help='Output CSV file path', default='data/nesto_merge_0.csv')
    parser.add_argument('--id-field', help='Field to join on', default='ID')
    parser.add_argument('--create-config', action='store_true', help='Create a sample configuration file')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
        return
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
            base_url = config['base_url']
            layer_ids = [layer['id'] for layer in config['layers']]
            id_field = config.get('id_field', 'ID')
            output_file = config.get('output_file', 'data/nesto_merge_0.csv')
        elif args.url and args.layers:
            base_url = args.url
            layer_ids = args.layers.split(',')
            id_field = args.id_field
            output_file = args.output
        else:
            logger.error("Either --config or both --url and --layers must be provided")
            parser.print_help()
            return
        
        # Initialize extractor
        extractor = ArcGISDataExtractor(base_url)
        
        # Extract data from each layer
        layer_dataframes = {}
        
        for layer_id in layer_ids:
            try:
                df = extractor.extract_layer_data(layer_id.strip())
                if not df.empty:
                    layer_dataframes[layer_id.strip()] = df
                else:
                    logger.warning(f"Layer {layer_id} returned no data")
            except Exception as e:
                logger.error(f"Failed to extract layer {layer_id}: {e}")
                continue
        
        if not layer_dataframes:
            logger.error("No data extracted from any layers")
            return
        
        # Join all layers
        joined_df = extractor.join_layers_by_id(layer_dataframes, id_field)
        
        # Clean the data
        clean_df = extractor.clean_data(joined_df)
        
        # Save to CSV
        save_to_csv(clean_df, output_file)
        
        # Print summary
        print(f"\n🎉 SUCCESS!")
        print(f"📁 Output file: {output_file}")
        print(f"📊 Records: {len(clean_df)}")
        print(f"📋 Columns: {len(clean_df.columns)}")
        print(f"🔑 ID field: {id_field}")
        print(f"🗂️ Layers joined: {len(layer_dataframes)}")
        
        # Show first few column names
        print(f"\n📝 Sample columns:")
        for i, col in enumerate(clean_df.columns[:10]):
            print(f"  {i+1}. {col}")
        if len(clean_df.columns) > 10:
            print(f"  ... and {len(clean_df.columns) - 10} more")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 