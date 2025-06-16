#!/usr/bin/env python3
"""
SHAP Data Status Verification Script
Use this script to regularly verify that your SHAP values file is up-to-date and properly structured.

Usage: python3 verify_shap_status.py
"""

import pandas as pd
import json
import os
from datetime import datetime
import sys

def check_file_status():
    """Check basic file information"""
    print('=== FILE STATUS ===')
    
    shap_file = 'precalculated/shap_values.pkl.gz'
    metadata_file = 'precalculated/models/metadata.json'
    
    files_status = {}
    
    for file_path, name in [(shap_file, 'SHAP data'), (metadata_file, 'Metadata')]:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024*1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            files_status[name] = {
                'exists': True,
                'size_mb': size_mb,
                'modified': mod_time
            }
            print(f'✅ {name}: {size_mb:.2f} MB, modified {mod_time}')
        else:
            files_status[name] = {'exists': False}
            print(f'❌ {name}: File not found at {file_path}')
    
    return files_status

def verify_data_structure():
    """Verify SHAP data structure and quality"""
    print('\n=== DATA STRUCTURE ===')
    
    try:
        shap_df = pd.read_pickle('precalculated/shap_values.pkl.gz', compression='gzip')
        
        # Basic structure
        total_rows = len(shap_df)
        total_cols = len(shap_df.columns)
        data_cols = [col for col in shap_df.columns if not col.startswith('shap_')]
        shap_cols = [col for col in shap_df.columns if col.startswith('shap_')]
        
        print(f'✅ Loaded successfully: {total_rows:,} rows, {total_cols} columns')
        print(f'   Data columns: {len(data_cols)}, SHAP columns: {len(shap_cols)}')
        
        # Check required fields
        required_fields = ['ID', 'CONVERSION_RATE']
        missing_fields = [field for field in required_fields if field not in data_cols]
        
        if missing_fields:
            print(f'❌ Missing required fields: {missing_fields}')
            return False
        else:
            print(f'✅ All required fields present: {required_fields}')
        
        return shap_df, shap_cols
        
    except Exception as e:
        print(f'❌ Error loading SHAP data: {e}')
        return False

def verify_metadata_alignment(shap_df, shap_cols):
    """Verify SHAP data aligns with metadata"""
    print('\n=== METADATA ALIGNMENT ===')
    
    try:
        with open('precalculated/models/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        if 'conversion' not in metadata:
            print('❌ Missing conversion model in metadata')
            return False
        
        metadata_features = set(metadata['conversion']['features'])
        shap_features = set(col.replace('shap_', '') for col in shap_cols)
        
        if metadata_features == shap_features:
            print(f'✅ Perfect alignment: {len(metadata_features)} features match exactly')
            return True
        else:
            missing_in_shap = metadata_features - shap_features
            missing_in_metadata = shap_features - metadata_features
            
            print(f'⚠️  Feature mismatch:')
            print(f'   Metadata: {len(metadata_features)} features')
            print(f'   SHAP: {len(shap_features)} features')
            print(f'   Overlap: {len(metadata_features & shap_features)} features')
            
            if missing_in_shap:
                print(f'   Missing in SHAP: {len(missing_in_shap)} features')
            if missing_in_metadata:
                print(f'   Missing in metadata: {len(missing_in_metadata)} features')
            
            return False
            
    except Exception as e:
        print(f'❌ Error checking metadata: {e}')
        return False

def verify_data_quality(shap_df):
    """Verify data quality metrics"""
    print('\n=== DATA QUALITY ===')
    
    # ID column quality
    id_unique = shap_df['ID'].nunique()
    id_nulls = shap_df['ID'].isnull().sum()
    total_rows = len(shap_df)
    
    print(f'ID column: {id_unique:,} unique values, {id_nulls} nulls out of {total_rows:,} rows')
    
    if id_nulls > 0:
        print(f'⚠️  ID column has {id_nulls} null values')
    else:
        print('✅ ID column has no null values')
    
    # CONVERSION_RATE quality
    cr_nulls = shap_df['CONVERSION_RATE'].isnull().sum()
    cr_null_pct = (cr_nulls / total_rows) * 100
    
    if cr_nulls > 0:
        cr_min = shap_df['CONVERSION_RATE'].min()
        cr_max = shap_df['CONVERSION_RATE'].max()
        cr_mean = shap_df['CONVERSION_RATE'].mean()
        
        print(f'CONVERSION_RATE: {cr_nulls} nulls ({cr_null_pct:.1f}%)')
        print(f'   Range: {cr_min:.6f} to {cr_max:.6f}, mean: {cr_mean:.6f}')
        
        if cr_null_pct > 10:
            print(f'⚠️  High null rate in CONVERSION_RATE ({cr_null_pct:.1f}%)')
        else:
            print(f'✅ Acceptable null rate in CONVERSION_RATE ({cr_null_pct:.1f}%)')
    else:
        print('✅ CONVERSION_RATE has no null values')
    
    # SHAP values quality check
    shap_cols = [col for col in shap_df.columns if col.startswith('shap_')]
    if shap_cols:
        sample_col = shap_cols[0]
        shap_values = shap_df[sample_col]
        shap_nulls = shap_values.isnull().sum()
        
        if shap_nulls == 0:
            print(f'✅ SHAP values complete (checked {sample_col})')
        else:
            print(f'⚠️  SHAP values have nulls: {shap_nulls} in {sample_col}')
    
    return cr_null_pct <= 10 and id_nulls == 0

def main():
    """Main verification function"""
    print('SHAP DATA STATUS VERIFICATION')
    print('=' * 50)
    
    # Check file status
    file_status = check_file_status()
    
    if not file_status.get('SHAP data', {}).get('exists', False):
        print('\n❌ SHAP data file not found. Cannot proceed with verification.')
        sys.exit(1)
    
    if not file_status.get('Metadata', {}).get('exists', False):
        print('\n❌ Metadata file not found. Cannot proceed with verification.')
        sys.exit(1)
    
    # Verify data structure
    structure_result = verify_data_structure()
    if not structure_result:
        print('\n❌ Data structure verification failed.')
        sys.exit(1)
    
    shap_df, shap_cols = structure_result
    
    # Verify metadata alignment
    metadata_ok = verify_metadata_alignment(shap_df, shap_cols)
    
    # Verify data quality
    quality_ok = verify_data_quality(shap_df)
    
    # Final summary
    print('\n=== FINAL SUMMARY ===')
    
    if metadata_ok and quality_ok:
        print('✅ SHAP data is UP-TO-DATE and HIGH QUALITY!')
        print('✅ Ready for production use')
        print(f'✅ {len(shap_df):,} records with {len(shap_cols)} SHAP features')
    else:
        print('⚠️  SHAP data needs attention:')
        if not metadata_ok:
            print('   - Metadata alignment issues')
        if not quality_ok:
            print('   - Data quality issues')
        print('\nConsider regenerating SHAP data if issues persist.')
    
    # Show file ages
    shap_age = datetime.now() - file_status['SHAP data']['modified']
    print(f'\nSHAP data age: {shap_age.days} days, {shap_age.seconds // 3600} hours')
    
    if shap_age.days > 7:
        print('⚠️  SHAP data is over a week old - consider updating')
    elif shap_age.days > 1:
        print('ℹ️  SHAP data is a few days old')
    else:
        print('✅ SHAP data is recent')

if __name__ == '__main__':
    main() 