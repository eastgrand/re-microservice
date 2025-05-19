import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path):
    """Load and preprocess the dataset."""
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_features(df, target_col=None):
    """Prepare features for training."""
    # Remove any non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].copy()
    
    # Handle missing values in features
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if target_col and target_col in df.columns:
        y = df[target_col].copy()
        # Handle missing values in target
        y = y.fillna(y.mean())
        # Validate target values
        if y.isna().any() or np.isinf(y).any():
            raise ValueError("Target variable contains NaN or infinite values after preprocessing")
        if (y < 0).any() or (y > 100).any():
            raise ValueError("Target variable contains values outside valid range [0, 100]")
        return X_scaled, y, numeric_cols
    return X_scaled, numeric_cols

def train_prediction_model(X, y):
    """Train a model for general predictions."""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
    )
    model.fit(X, y)
    return model

def train_hotspot_model(X, y):
    """Train a model for hotspot detection."""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        min_child_weight=3  # More conservative predictions for hotspots
    )
    model.fit(X, y)
    return model

def train_anomaly_model(X, y):
    """Train a model for anomaly detection."""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8  # More sensitive to anomalies
    )
    model.fit(X, y)
    return model

def train_correlation_model(X, y):
    """Train a model for correlation analysis."""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        colsample_bytree=0.8  # Focus on feature relationships
    )
    model.fit(X, y)
    return model

def train_multivariate_model(X, y):
    """Train a model for multivariate analysis."""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        colsample_bytree=0.9  # Consider more features
    )
    model.fit(X, y)
    return model

def train_network_model(X, y):
    """Train a model for network analysis."""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.9  # Consider more samples for network patterns
    )
    model.fit(X, y)
    return model

def save_model(model, model_type, feature_names, output_dir):
    """Save model and feature names."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f'{model_type}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature names
    feature_names_path = os.path.join(output_dir, 'feature_names.txt')
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(feature_names))
    
    logger.info(f"Saved {model_type} model and features to {output_dir}")

def main():
    # Configuration
    data_path = 'data/nesto_training_data_0.csv'  # Update path to your dataset
    output_dir = 'models'
    target_column = 'CONVERSION_RATE'  # Using conversion rate as target
    
    try:
        # Load and prepare data
        df = load_data(data_path)
        
        # Validate required features
        required_features = [
            '2024 Total Population',  # Population
            '2024 Household Average Income (Current Year $)',  # Income
            '2024 Household Median Income (Current Year $)',  # Median_Income
            '2024 Maintainers - Median Age',  # Age
            '2024 Tenure: Owned (%)',  # Homeownership_Pct
            '2024 Labour Force - Labour Employment Rate',  # Employment_Rate
            '2024 Labour Force - Labour Unemployment Rate',  # Unemployment_Rate
            '2024 Property Taxes (Shelter)',  # Property_Tax_Total
            '2024 Property Taxes (Shelter) (Avg)',  # Avg_Property_Tax
            '2024 Regular Mortgage Payments (Shelter)',  # Mortgage_Payments_Total
            '2024 Regular Mortgage Payments (Shelter) (Avg)',  # Avg_Mortgage_Payment
            '2024 Household Aggregate Income',  # Total_Income
            '2024 Household Discretionary Aggregate Income',  # Discretionary_Income
            '2024 Household Disposable Aggregate Income',  # Disposable_Income
            '2024 Financial Services'  # Financial_Services_Total
        ]
        
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Prepare features
        X, y, feature_names = prepare_features(df, target_column)
        
        # Train models for each type
        model_types = {
            'prediction': train_prediction_model,
            'hotspot': train_hotspot_model,
            'anomaly': train_anomaly_model,
            'correlation': train_correlation_model,
            'multivariate': train_multivariate_model,
            'network': train_network_model
        }
        
        for model_type, train_func in model_types.items():
            logger.info(f"Training {model_type} model...")
            model = train_func(X, y)
            save_model(model, model_type, feature_names, output_dir)
        
        logger.info("All models trained and saved successfully!")
        
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}")
        raise

if __name__ == '__main__':
    main() 