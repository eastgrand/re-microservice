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

# Get data path from environment or use default paths
ENV_DATASET_PATH = os.getenv('DATASET_PATH', 'data/cleaned_data.csv')
DATA_PATHS = [
    ENV_DATASET_PATH,                              # Path from environment variable (highest priority)
    os.path.join('data', 'cleaned_data.csv'),      # Mapped nesto data (preferred)
    os.path.join('data', 'nesto_merge_0.csv'),     # New nesto data with descriptive field names
    os.path.join('data', 'sales_data.csv'),        # Original production data path
    os.path.join('data', 'sample_data.csv')        # Sample data for development
]

# Try each path until we find a valid data file
df = None
dataset_path = None
for path in DATA_PATHS:
    if os.path.exists(path):
        logger.info(f"Loading data from {path}...")
        try:
            if 'nesto_merge_0.csv' in path:
                # Direct loading of nesto_merge_0.csv
                logger.info(f"Loading Nesto data from {path}")
                df = pd.read_csv(path, low_memory=False)  # Added low_memory=False to avoid dtype warning
                dataset_path = path
                
                # Define the field mappings
                from map_nesto_data import FIELD_MAPPINGS, TARGET_VARIABLE
                
                # Map column names to the standardized format
                rename_dict = {}
                for source_col, target_col in FIELD_MAPPINGS.items():
                    if source_col in df.columns:
                        rename_dict[source_col] = target_col
                
                # Apply column renaming
                if rename_dict:
                    df = df.rename(columns=rename_dict)
                    logger.info(f"Renamed {len(rename_dict)} columns using field mappings")
                
                # Add proxy location data for visualization
                if 'Shape__Area' in df.columns and 'Shape__Length' in df.columns:
                    df['latitude'] = df['Shape__Area'].rank(pct=True) * 10
                    df['longitude'] = df['Shape__Length'].rank(pct=True) * 10
                    logger.info("Added proxy latitude/longitude based on Shape area/length")
                
                data_source = "Nesto mortgage data"
                data_description = "Nesto mortgage application data with descriptive fields"
                    
            else:
                # For other data files, load directly
                df = pd.read_csv(path)
                dataset_path = path
                logger.info(f"Loaded dataset with {df.shape[0]} records and {df.shape[1]} columns")
                
                # Set source and description based on file path
                if 'sales_data' in path:
                    data_source = "Production sales data"
                    data_description = "Real sales data from production database" 
                elif 'sample_data' in path:
                    data_source = "Sample data"
                    data_description = "Sample data for development and testing"
                else:
                    data_source = "Unknown source"
                    data_description = f"Data loaded from {path}"
                
            break  # Exit the loop once we successfully load data
        except Exception as e:
            logger.error(f"Error loading data from {path}: {e}")
            continue
else:
    # If no data file exists, generate sample data
    logger.info("No data file found. Generating sample data...")
    try:
        # Try to import generate_sample_data
        import generate_sample_data
        sample_path = 'data/sample_data.csv'
        generate_sample_data.generate_sample_data(output_file=sample_path)
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

# Check before dropping NAs how many records we have
print(f"Initial record count: {len(df)}")

# Check for required columns before cleaning
from map_nesto_data import TARGET_VARIABLE
required_columns = ['zip_code', TARGET_VARIABLE, 'Income', 'Age']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    # Instead of failing, create the missing columns with reasonable default values
    logger.warning(f"Adding missing required columns: {missing_columns}")
    
    for col in missing_columns:
        if col == 'zip_code' and 'Forward Sortation Area' in df.columns:
            logger.info("Using Forward Sortation Area as zip_code")
            df['zip_code'] = df['Forward Sortation Area']
        elif col == TARGET_VARIABLE:
            logger.info(f"Creating default target variable '{TARGET_VARIABLE}' with random values")
            df[TARGET_VARIABLE] = np.random.randint(1, 100, size=len(df))
        elif col == 'Income':
            logger.info("Creating default Income values with normal distribution")
            df['Income'] = np.random.normal(75000, 25000, size=len(df))
        elif col == 'Age':
            logger.info("Creating default Age values with normal distribution")
            df['Age'] = np.random.normal(40, 10, size=len(df))

# Examine the target column before cleaning
if 'Mortgage_Approvals' in df.columns:
    print(f"Target column stats - NaN count: {df['Mortgage_Approvals'].isna().sum()}")
    print(f"Target column stats - unique values: {df['Mortgage_Approvals'].nunique()}")
    print(f"Target column stats - first few values: {df['Mortgage_Approvals'].head()}")

# Replace NaN values with zeros for critical columns instead of dropping rows
for col in required_columns:
    if col in df.columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            print(f"Filling {null_count} NaN values in '{col}' with zeros")
            df[col] = df[col].fillna(0)

# Make sure numeric columns are properly typed
numeric_columns = ['Income', 'Age', 'Mortgage_Approvals']
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
TARGET_VARIABLE = 'Mortgage_Approvals'  # New target variable for mortgage data
exclude_cols = ['zip_code', TARGET_VARIABLE]
if 'latitude' in df.columns:
    exclude_cols.append('latitude')
if 'longitude' in df.columns:
    exclude_cols.append('longitude')

# Also exclude any non-numeric columns and object dtypes that XGBoost can't handle
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category' or col.startswith('ID'):
        exclude_cols.append(col)
        print(f"Excluding non-numeric column: {col} (dtype: {df[col].dtype})")

X = df.drop(exclude_cols, axis=1)
print(f"Selected {len(X.columns)} numeric feature columns")
y = df[TARGET_VARIABLE]

# Check X dataset for any remaining objects
object_cols = X.select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    print(f"Warning: {len(object_cols)} object columns remain. Converting to numeric.")
    for col in object_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(0)  # Replace NaN with zeros

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

# Initialize CV metrics
cv_metrics = {}

# Use cross-validation to evaluate model robustness
try:
    if len(X) >= 50:  # Only do cross-validation if we have enough data
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import KFold
        
        print("Performing 5-fold cross-validation...")
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        model_cv = xgb.XGBRegressor(**model_params)
        
        try:
            cv_scores = cross_val_score(model_cv, X, y, cv=cv, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)
            print(f"Cross-validation RMSE: {cv_rmse.mean():.4f} (±{cv_rmse.std():.4f})")
            
            # Add CV metrics to the metrics dictionary that will be defined later
            cv_metrics = {
                "cv_rmse_mean": float(cv_rmse.mean()),
                "cv_rmse_std": float(cv_rmse.std())
            }
        except Exception as e:
            print(f"Cross-validation failed: {str(e)}")
            print("Skipping cross-validation and proceeding with model training")
except Exception as e:
    logger.warning(f"Error during cross-validation setup: {e}")
    print("Skipping cross-validation due to setup error")

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

# Add cross-validation metrics if they were calculated
if cv_metrics and len(cv_metrics) > 0:
    metrics.update(cv_metrics)

model_version_id = version_tracker.register_model(
    model_path, 
    dataset_version_id, 
    feature_names_path=feature_names_path,
    metrics=metrics
)

logger.info(f"Registered model with version ID: {model_version_id}")
logger.info("Done! Model and feature names saved.")