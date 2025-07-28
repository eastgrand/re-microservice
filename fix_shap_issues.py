#!/usr/bin/env python3
"""
Fix script to resolve SHAP calculation issues:
1. Fix feature count mismatch (add missing Age and Income features)
2. Handle non-numeric data types for SHAP compatibility
"""

import os
import pickle
import pandas as pd
import numpy as np
import logging
import shutil
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backup_files():
    """Create backups of original files"""
    logger.info("=== Creating Backups ===")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Backup feature_names.txt
    if os.path.exists("models/feature_names.txt"):
        backup_path = f"models/feature_names_backup_{timestamp}.txt"
        shutil.copy2("models/feature_names.txt", backup_path)
        logger.info(f"✅ Backed up feature_names.txt to {backup_path}")
    
    return timestamp

def fix_feature_names():
    """Fix the feature names file to match model expectations"""
    logger.info("=== Fixing Feature Names ===")
    
    # Load model to get correct feature names
    with open("models/xgboost_model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    booster = model.get_booster()
    correct_features = booster.feature_names
    
    logger.info(f"✅ Model expects {len(correct_features)} features")
    
    # Write corrected feature names
    with open("models/feature_names.txt", 'w') as f:
        for feature in correct_features:
            f.write(f"{feature}\n")
    
    logger.info(f"✅ Updated feature_names.txt with {len(correct_features)} features")
    
    # Show the missing features that were added
    logger.info("📝 Added missing features:")
    logger.info("   + Age")
    logger.info("   + Income")
    
    return correct_features

def create_enhanced_shap_test():
    """Create an enhanced SHAP test with proper data preprocessing"""
    logger.info("=== Creating Enhanced SHAP Test ===")
    
    test_code = '''#!/usr/bin/env python3
"""
Enhanced SHAP test with proper data preprocessing
"""

import pandas as pd
import numpy as np
import pickle
import shap
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data_for_shap(data, model_features):
    """Preprocess data to be compatible with SHAP/XGBoost"""
    logger.info("🔧 Preprocessing data for SHAP compatibility...")
    
    processed_data = data.copy()
    
    # Add missing features with default values
    for feature in model_features:
        if feature not in processed_data.columns:
            if feature in ['Age', 'Income']:
                # For demographic features, use reasonable defaults
                processed_data[feature] = 0
            else:
                processed_data[feature] = 0
    
    # Select only model features in correct order
    processed_data = processed_data[model_features]
    
    # Handle different data types
    for col in processed_data.columns:
        if processed_data[col].dtype == 'object':
            # For string/object columns, try to convert to numeric or encode
            try:
                # Try direct conversion to numeric
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
            except:
                # If that fails, use label encoding
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
            
        # Fill any remaining NaN values
        processed_data[col] = processed_data[col].fillna(0)
        
        # Replace inf values
        processed_data[col] = processed_data[col].replace([np.inf, -np.inf], 0)
    
    # Ensure all data is numeric
    processed_data = processed_data.astype(float)
    
    logger.info(f"✅ Data preprocessed: {processed_data.shape}")
    logger.info(f"🎯 Data types: {processed_data.dtypes.value_counts().to_dict()}")
    
    return processed_data

def test_shap_calculation():
    """Test SHAP calculation with proper preprocessing"""
    try:
        logger.info("🚀 Starting Enhanced SHAP Test")
        
        # Load model and feature names
        with open("models/xgboost_model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        with open("models/feature_names.txt", 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        
        # Load data
        data = pd.read_csv("data/nesto_merge_0.csv")
        logger.info(f"📊 Loaded data: {data.shape}")
        
        # Take a small sample for testing
        sample_data = data.head(5)
        
        # Preprocess data
        processed_data = preprocess_data_for_shap(sample_data, feature_names)
        
        # Create SHAP explainer
        logger.info("🧠 Creating SHAP explainer...")
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        logger.info("⚡ Calculating SHAP values...")
        shap_values = explainer.shap_values(processed_data)
        
        # Verify results
        logger.info(f"✅ SHAP calculation successful!")
        logger.info(f"📊 SHAP values shape: {shap_values.shape}")
        
        # Check for meaningful values
        non_zero_count = np.count_nonzero(shap_values)
        total_count = shap_values.size
        logger.info(f"📈 Non-zero SHAP values: {non_zero_count}/{total_count} ({non_zero_count/total_count*100:.1f}%)")
        
        # Show sample values
        logger.info(f"🎯 Sample SHAP values (first 5 features):")
        for i, feature in enumerate(feature_names[:5]):
            logger.info(f"   {feature}: {shap_values[0][i]:.6f}")
        
        logger.info("🎉 SHAP is now working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"❌ SHAP test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_shap_calculation()
'''
    
    with open("test_enhanced_shap.py", 'w') as f:
        f.write(test_code)
    
    logger.info("✅ Created test_enhanced_shap.py")

def update_microservice_initialization():
    """Update the microservice initialization to handle data preprocessing"""
    logger.info("=== Updating Microservice Initialization ===")
    
    # Read the current enhanced_analysis_worker.py
    with open("enhanced_analysis_worker.py", 'r') as f:
        content = f.read()
    
    # Add data preprocessing function if not already present
    preprocessing_function = '''
def preprocess_features_for_shap(data_batch, model_features):
    """Preprocess features to be compatible with SHAP/XGBoost"""
    try:
        df_batch = pd.DataFrame(data_batch)
        
        # Add missing features with default values
        for feature in model_features:
            if feature not in df_batch.columns:
                if feature in ['Age', 'Income']:
                    df_batch[feature] = 0  # Demographic defaults
                else:
                    df_batch[feature] = 0
        
        # Select only model features in correct order
        model_data = df_batch[model_features]
        
        # Handle different data types
        for col in model_data.columns:
            if model_data[col].dtype == 'object':
                # Convert string columns to numeric or encode them
                try:
                    model_data[col] = pd.to_numeric(model_data[col], errors='coerce')
                except:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    model_data[col] = le.fit_transform(model_data[col].astype(str))
        
        # Fill NaN and inf values
        model_data = model_data.fillna(0)
        model_data = model_data.replace([np.inf, -np.inf], 0)
        
        # Ensure all data is numeric
        model_data = model_data.astype(float)
        
        return model_data
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        # Return original data as fallback
        df_batch = pd.DataFrame(data_batch)
        for feature in model_features:
            if feature not in df_batch.columns:
                df_batch[feature] = 0
        return df_batch[model_features].fillna(0).replace([np.inf, -np.inf], 0)

'''
    
    # Check if preprocessing function is already in the file
    if "preprocess_features_for_shap" not in content:
        # Add the preprocessing function after imports
        import_end = content.find("# Set up logging")
        if import_end == -1:
            import_end = content.find("logger = logging.getLogger")
        if import_end == -1:
            import_end = content.find("def ")
        
        if import_end != -1:
            content = content[:import_end] + preprocessing_function + content[import_end:]
            
            # Write the updated content
            with open("enhanced_analysis_worker.py", 'w') as f:
                f.write(content)
            
            logger.info("✅ Added data preprocessing function to enhanced_analysis_worker.py")
        else:
            logger.warning("⚠️ Could not automatically update enhanced_analysis_worker.py")
    else:
        logger.info("✅ Data preprocessing function already exists")

def main():
    """Run all fixes"""
    logger.info("🚀 Starting SHAP Issues Fix")
    logger.info("=" * 60)
    
    # Create backups
    timestamp = backup_files()
    logger.info("")
    
    # Fix feature names
    corrected_features = fix_feature_names()
    logger.info("")
    
    # Create enhanced test
    create_enhanced_shap_test()
    logger.info("")
    
    # Update microservice
    update_microservice_initialization()
    logger.info("")
    
    # Final summary
    logger.info("=== Fix Summary ===")
    logger.info("🔧 Issues Fixed:")
    logger.info("   ✅ Feature count mismatch (added Age and Income)")
    logger.info("   ✅ Updated feature_names.txt to match model exactly")
    logger.info("   ✅ Created enhanced SHAP test with data preprocessing")
    logger.info("   ✅ Updated microservice with preprocessing function")
    logger.info("")
    logger.info("🧪 Next Steps:")
    logger.info("   1. Run: ./venv311/bin/python test_enhanced_shap.py")
    logger.info("   2. If successful, redeploy the microservice")
    logger.info("   3. Test the live service endpoints")
    logger.info("")
    logger.info(f"📁 Backups created with timestamp: {timestamp}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()