#!/usr/bin/env python3
"""
SHAP Values File Verification Script
Analyzes the structure, content, and quality of the SHAP values file
"""

import pandas as pd
import pickle
from datetime import datetime
import os
import numpy as np

def verify_shap_data():
    print('=== SHAP VALUES FILE VERIFICATION ===')
    
    # Check file info
    shap_file = 'precalculated/shap_values.pkl.gz'
    if not os.path.exists(shap_file):
        print(f'❌ SHAP file not found: {shap_file}')
        return False
    
    file_size = os.path.getsize(shap_file) / (1024*1024)  # MB
    file_time = datetime.fromtimestamp(os.path.getmtime(shap_file))
    print(f'File: {shap_file}')
    print(f'Size: {file_size:.2f} MB')
    print(f'Last modified: {file_time}')
    
    # Load and analyze SHAP data
    print('\n=== SHAP DATA STRUCTURE ===')
    try:
        shap_df = pd.read_pickle(shap_file, compression='gzip')
        print(f'✅ Successfully loaded SHAP data')
        print(f'Total rows: {len(shap_df):,}')
        print(f'Total columns: {len(shap_df.columns)}')
        
        # Separate data columns from SHAP columns
        data_cols = [col for col in shap_df.columns if not col.startswith('shap_')]
        shap_cols = [col for col in shap_df.columns if col.startswith('shap_')]
        feature_names = [col.replace('shap_', '') for col in shap_cols]
        
        print(f'Data columns: {len(data_cols)}')
        print(f'SHAP columns: {len(shap_cols)}')
        print(f'Features analyzed: {len(feature_names)}')
        
        print(f'\nData columns: {data_cols}')
        print(f'\nFirst 10 features: {feature_names[:10]}')
        print(f'Last 10 features: {feature_names[-10:]}')
        
        # Check for key fields we need
        key_fields = ['ID', 'CONVERSION_RATE']
        missing_fields = [field for field in key_fields if field not in data_cols]
        if missing_fields:
            print(f'\n⚠️  Missing key fields: {missing_fields}')
        else:
            print(f'\n✅ All key fields present: {key_fields}')
        
        # Check data quality
        print(f'\n=== DATA QUALITY ===')
        if 'ID' in shap_df.columns:
            print(f'ID range: {shap_df["ID"].min()} to {shap_df["ID"].max()}')
            print(f'Unique IDs: {shap_df["ID"].nunique():,}')
            print(f'Null values in ID: {shap_df["ID"].isnull().sum()}')
        
        if 'CONVERSION_RATE' in shap_df.columns:
            print(f'CONVERSION_RATE range: {shap_df["CONVERSION_RATE"].min():.6f} to {shap_df["CONVERSION_RATE"].max():.6f}')
            print(f'CONVERSION_RATE mean: {shap_df["CONVERSION_RATE"].mean():.6f}')
            print(f'Null values in CONVERSION_RATE: {shap_df["CONVERSION_RATE"].isnull().sum()}')
        
        # Check SHAP values quality
        print(f'\n=== SHAP VALUES QUALITY ===')
        if shap_cols:
            # Sample a few SHAP columns for analysis
            sample_shap_cols = shap_cols[:5]
            for col in sample_shap_cols:
                values = shap_df[col]
                print(f'{col}: range [{values.min():.6f}, {values.max():.6f}], mean {values.mean():.6f}, nulls {values.isnull().sum()}')
        
        # Check if features match between SHAP and metadata
        print(f'\n=== METADATA COMPARISON ===')
        metadata_file = 'precalculated/models/metadata.json'
        if os.path.exists(metadata_file):
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if 'conversion_rate_model' in metadata:
                model_features = metadata['conversion_rate_model'].get('features', [])
                print(f'Features in metadata: {len(model_features)}')
                print(f'Features in SHAP data: {len(feature_names)}')
                
                # Check overlap
                metadata_set = set(model_features)
                shap_set = set(feature_names)
                
                missing_in_shap = metadata_set - shap_set
                missing_in_metadata = shap_set - metadata_set
                
                if missing_in_shap:
                    print(f'⚠️  Features in metadata but not in SHAP: {list(missing_in_shap)[:10]}')
                if missing_in_metadata:
                    print(f'⚠️  Features in SHAP but not in metadata: {list(missing_in_metadata)[:10]}')
                
                if not missing_in_shap and not missing_in_metadata:
                    print('✅ Perfect match between SHAP features and metadata')
                else:
                    print(f'Overlap: {len(metadata_set & shap_set)} / {len(metadata_set | shap_set)} features')
        else:
            print(f'⚠️  Metadata file not found: {metadata_file}')
        
        # Check data freshness vs cleaned data
        print(f'\n=== DATA FRESHNESS ===')
        cleaned_data_file = 'cleaned_data.csv'
        if os.path.exists(cleaned_data_file):
            cleaned_time = datetime.fromtimestamp(os.path.getmtime(cleaned_data_file))
            print(f'Cleaned data modified: {cleaned_time}')
            print(f'SHAP data modified: {file_time}')
            
            if file_time > cleaned_time:
                print('✅ SHAP data is newer than cleaned data')
            else:
                print('⚠️  SHAP data is older than cleaned data - may need regeneration')
        
        return True
        
    except Exception as e:
        print(f'❌ Error loading SHAP data: {e}')
        return False

if __name__ == '__main__':
    verify_shap_data() 