import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
from data_versioning import DataVersionTracker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train-model")

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

logger.info("Starting model training process...")

# Initialize data version tracker
version_tracker = DataVersionTracker()

# Load real data from CSV
data_path = os.path.join('data', 'sales_data.csv')
sample_path = os.path.join('data', 'sample_data.csv')

if os.path.exists(data_path):
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded dataset with {df.shape[0]} records and {df.shape[1]} columns")
    data_source = "Production sales data"
    data_description = "Real sales data from production database"
    dataset_path = data_path
elif os.path.exists(sample_path):
    logger.info(f"Loading sample data from {sample_path}...")
    df = pd.read_csv(sample_path)
    logger.info(f"Loaded sample dataset with {df.shape[0]} records and {df.shape[1]} columns")
    data_source = "Sample data generator"
    data_description = "Generated sample data for development and testing"
    dataset_path = sample_path
else:
    # If no data file exists, generate sample data
    logger.info("No data file found. Generating sample data...")
    try:
        # Try to import generate_sample_data
        import generate_sample_data
        generate_sample_data.generate_sample_data()
        logger.info("Sample data generated successfully.")
        # Load the generated sample data
        df = pd.read_csv(sample_path)
        logger.info(f"Loaded generated sample dataset with {df.shape[0]} records and {df.shape[1]} columns")
        data_source = "Sample data generator"
        data_description = "Freshly generated sample data"
        dataset_path = sample_path
    except (ImportError, FileNotFoundError) as e:
        raise FileNotFoundError(f"No data file found and couldn't generate sample data. Error: {str(e)}")

# Basic data cleaning
print("Cleaning data...")
df = df.dropna(subset=['zip_code', 'Nike_Sales'])  # Adjust column names to match your dataset

# Data validation - ensure all required columns exist
required_columns = ['zip_code', 'Nike_Sales', 'Income', 'Age', 'Hispanic_Population']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Required columns missing from dataset: {missing_columns}")

# Make sure numeric columns are properly typed
numeric_columns = ['Income', 'Age', 'Hispanic_Population', 'Nike_Sales']
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle any additional NAs created by type conversion
df = df.dropna(subset=numeric_columns)

logger.info(f"After cleaning: {df.shape[0]} records and {df.shape[1]} columns")

# Save cleaned data for reference
cleaned_data_path = 'data/cleaned_data.csv'
df.to_csv(cleaned_data_path, index=False)
logger.info("Cleaned data saved")

# Register cleaned dataset in version tracker
dataset_version_id = version_tracker.register_dataset(
    cleaned_data_path,
    description=f"Cleaned {data_description}",
    source=data_source
)
logger.info(f"Registered cleaned dataset with version ID: {dataset_version_id}")

# Prepare features and target
print("Preparing data for training...")
# Exclude non-feature columns - adjust based on your dataset structure
exclude_cols = ['zip_code', 'Nike_Sales']
if 'latitude' in df.columns:
    exclude_cols.append('latitude')
if 'longitude' in df.columns:
    exclude_cols.append('longitude')

X = df.drop(exclude_cols, axis=1)
y = df['Nike_Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training XGBoost model...")
# Define model parameters
model_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Use cross-validation to evaluate model robustness
if len(X) >= 50:  # Only do cross-validation if we have enough data
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold

    print("Performing 5-fold cross-validation...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model_cv = xgb.XGBRegressor(**model_params)
    cv_scores = cross_val_score(model_cv, X, y, cv=cv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    print(f"Cross-validation RMSE: {cv_rmse.mean():.4f} (±{cv_rmse.std():.4f})")

# Train the final model on all training data
model = xgb.XGBRegressor(**model_params)
model.fit(X_train, y_train)

# Evaluate model on test set
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
logger.info(f"Test set evaluation - RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Save model
logger.info("Saving model...")
model_path = 'models/xgboost_model.pkl'
pickle.dump(model, open(model_path, 'wb'))

# Save feature names
feature_names_path = 'models/feature_names.txt'
with open(feature_names_path, 'w') as f:
    for feature in X.columns:
        f.write(f"{feature}\n")

# Register model in version tracker with metrics
metrics = {
    "rmse": float(rmse),
    "r2": float(r2),
    "test_size": len(y_test),
}

if locals().get('cv_rmse') is not None:
    metrics["cv_rmse_mean"] = float(cv_rmse.mean())
    metrics["cv_rmse_std"] = float(cv_rmse.std())

model_version_id = version_tracker.register_model(
    model_path, 
    dataset_version_id, 
    feature_names_path=feature_names_path,
    metrics=metrics
)

logger.info(f"Registered model with version ID: {model_version_id}")
logger.info("Done! Model and feature names saved.")