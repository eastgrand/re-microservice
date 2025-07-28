#!/usr/bin/env python3
"""
Train a new XGBoost model using only the 191 filtered features
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_filtered_features():
    """Load the list of 191 features to keep"""
    try:
        with open('../mpiq-ai-chat/complete_field_list_keep.txt', 'r') as f:
            features = [line.strip() for line in f if line.strip()]
        logger.info(f"✅ Loaded {len(features)} features to keep")
        
        # Remove SHAP fields and non-feature fields, and map value_ prefixed names
        feature_fields = []
        exclude_prefixes = ['shap_', 'DESCRIPTION', 'Creator', 'Editor', 'CreationDate', 'EditDate']
        exclude_exact = ['ID', 'OBJECTID']
        
        for field in features:
            if not any(field.startswith(prefix) for prefix in exclude_prefixes) and field not in exclude_exact:
                # Remove 'value_' prefix if present
                clean_field = field.replace('value_', '') if field.startswith('value_') else field
                feature_fields.append(clean_field)
        
        logger.info(f"✅ Filtered to {len(feature_fields)} model features")
        return feature_fields
    except FileNotFoundError:
        logger.error("❌ complete_field_list_keep.txt not found!")
        return []

def train_filtered_model():
    """Train XGBoost model with filtered features"""
    
    # Load training data
    logger.info("Loading training data...")
    training_data_path = "data/nesto_merge_0.csv"
    
    if not os.path.exists(training_data_path):
        logger.error(f"Training data not found at {training_data_path}")
        return
    
    df = pd.read_csv(training_data_path)
    logger.info(f"✅ Loaded training data: {df.shape}")
    
    # Load filtered features
    feature_fields = load_filtered_features()
    if not feature_fields:
        return
    
    # Define target variable
    target_variable = 'MP30034A_B_P'  # Nike
    
    # Check which features exist in the dataset and are numeric
    available_features = []
    missing_features = []
    
    for f in feature_fields:
        if f in df.columns:
            # Check if numeric
            if df[f].dtype in ['int64', 'float64', 'int32', 'float32']:
                available_features.append(f)
            else:
                logger.info(f"   Skipping non-numeric field: {f} (dtype: {df[f].dtype})")
        else:
            missing_features.append(f)
    
    logger.info(f"✅ Available features: {len(available_features)}")
    logger.info(f"⚠️  Missing features: {len(missing_features)}")
    
    if missing_features[:5]:
        logger.info(f"   Sample missing: {missing_features[:5]}")
    
    # Ensure target variable is in the dataset
    if target_variable not in df.columns:
        logger.error(f"❌ Target variable {target_variable} not found in dataset")
        return
    
    # Remove target from features if present
    if target_variable in available_features:
        available_features.remove(target_variable)
    
    logger.info(f"🎯 Training with {len(available_features)} features for target: {target_variable}")
    
    # Prepare data
    X = df[available_features].fillna(0)
    y = df[target_variable].fillna(0)
    
    # Replace inf values
    X = X.replace([np.inf, -np.inf], 0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"📊 Training set: {X_train.shape}")
    logger.info(f"📊 Test set: {X_test.shape}")
    
    # Train XGBoost model
    logger.info("🚀 Training XGBoost model...")
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    logger.info(f"✅ Model performance:")
    logger.info(f"   R² Score: {r2:.4f}")
    logger.info(f"   RMSE: {rmse:.4f}")
    
    # Save model
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, "xgboost_model_filtered.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"💾 Model saved to {model_path}")
    
    # Save feature names
    features_path = os.path.join(output_dir, "feature_names_filtered.txt")
    with open(features_path, 'w') as f:
        for feature in available_features:
            f.write(f"{feature}\n")
    logger.info(f"💾 Feature names saved to {features_path}")
    
    # Create a backup of the original model
    original_model = os.path.join(output_dir, "xgboost_model.pkl")
    if os.path.exists(original_model):
        backup_path = os.path.join(output_dir, "xgboost_model_original_543_features.pkl")
        os.rename(original_model, backup_path)
        logger.info(f"📦 Original model backed up to {backup_path}")
    
    original_features = os.path.join(output_dir, "feature_names.txt")
    if os.path.exists(original_features):
        backup_features = os.path.join(output_dir, "feature_names_original_543.txt")
        os.rename(original_features, backup_features)
        logger.info(f"📦 Original features backed up to {backup_features}")
    
    # Rename filtered model to be the main model
    os.rename(model_path, original_model)
    os.rename(features_path, original_features)
    logger.info(f"✅ Filtered model is now the main model")
    
    # Show feature importance
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': available_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    logger.info("\n🏆 Top 10 Important Features:")
    for idx, row in importance_df.head(10).iterrows():
        logger.info(f"   {row['feature']}: {row['importance']:.4f}")
    
    logger.info("\n🎉 Model training complete!")
    logger.info(f"📊 Model trained on {len(available_features)} features")
    logger.info(f"🎯 Target variable: {target_variable}")
    logger.info(f"✅ Ready for SHAP analysis with filtered features")

if __name__ == "__main__":
    train_filtered_model()