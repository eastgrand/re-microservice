import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import gc
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
from data_versioning import DataVersionTracker
from optimize_memory import (
    log_memory_usage, 
    optimize_dtypes, 
    load_and_optimize_data,
    is_memory_critical,
    sample_data_if_needed,
    reduce_model_complexity
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train-model")

# Check for memory optimization environment variable
MEMORY_OPTIMIZATION = os.environ.get('MEMORY_OPTIMIZATION', 'false').lower() == 'true'
if MEMORY_OPTIMIZATION:
    logger.info("Memory optimization mode is ENABLED")
    
    # Set smaller memory thresholds for Render deployment
    MEMORY_CRITICAL = 400  # MB
    MEMORY_WARNING = 350   # MB
    MEMORY_CAUTION = 300   # MB
    
    # Log system information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Running on: {os.uname().sysname} {os.uname().release}")
    
    # Print memory info at startup
    log_memory_usage("Script startup")
else:
    logger.info("Running in standard mode (memory optimization disabled)")
    
    # Higher thresholds for local development
    MEMORY_CRITICAL = 1800  # MB
    MEMORY_WARNING = 1500   # MB
    MEMORY_CAUTION = 1200   # MB

# Define data paths globally for import from app.py
DATA_PATHS = [
    os.getenv('DATASET_PATH', 'data/cleaned_data.csv'),  # Path from environment variable
    os.path.join('data', 'cleaned_data.csv'),            # Mapped nesto data 
    os.path.join('data', 'nesto_merge_0.csv'),           # Nesto data with descriptive fields
    os.path.join('data', 'sales_data.csv'),              # Original production data
    os.path.join('data', 'sample_data.csv')              # Sample data for development
]

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

# Log initial memory usage
log_memory_usage("Before data loading")

# Try each path until we find a valid data file
df = None
dataset_path = None

# For Render deployment, use more aggressive memory settings
is_render = 'RENDER' in os.environ
initial_sample_size = 20000 if is_render else 50000

for path in DATA_PATHS:
    if os.path.exists(path):
        logger.info(f"Loading data from {path}...")
        try:
            if 'nesto_merge_0.csv' in path:
                # Use memory-optimized loading for Nesto data
                logger.info(f"Loading Nesto data from {path} with memory optimization")
                # On Render with limited memory, we may need to use a sample of the data
                try:
                    # Start with a smaller sample on Render to avoid immediate OOM
                    if is_render:
                        logger.info(f"Running on Render, using initial sample of {initial_sample_size} rows")
                        df = load_and_optimize_data(path, low_memory=False, nrows=initial_sample_size)
                    else:
                        df = load_and_optimize_data(path, low_memory=False)
                except Exception as mem_error:
                    logger.warning(f"Full dataset loading failed: {mem_error}. Falling back to smaller sample.")
                    # Try with a smaller sample if full load fails
                    sample_size = 10000 if is_render else 25000
                    df = load_and_optimize_data(path, low_memory=False, nrows=sample_size)
                    
                # Even after loading, if memory is still high, prune columns
                from optimize_memory import prune_dataframe_columns
                
                if is_render or is_memory_critical(threshold_mb=400):
                    logger.warning("Memory still high after loading, removing legacy fields")
                    target_col = "Mortgage_Approvals"
                    df = prune_dataframe_columns(df, target_column=target_col)
                
                dataset_path = path
                
                # Force garbage collection before mapping
                gc.collect()
                log_memory_usage("After loading Nesto data")
                
                # Define the field mappings
                from map_nesto_data import FIELD_MAPPINGS, TARGET_VARIABLE
                
                # Map column names to the standardized format in a memory-efficient way
                rename_dict = {}
                for source_col, target_col in FIELD_MAPPINGS.items():
                    if source_col in df.columns:
                        rename_dict[source_col] = target_col
                
                # Apply column renaming
                if rename_dict:
                    df = df.rename(columns=rename_dict)
                    logger.info(f"Renamed {len(rename_dict)} columns using field mappings")
                
                # Only add proxy location if memory isn't critical
                if 'Shape__Area' in df.columns and 'Shape__Length' in df.columns and not is_memory_critical(400):
                    df['latitude'] = df['Shape__Area'].rank(pct=True) * 10
                    df['longitude'] = df['Shape__Length'].rank(pct=True) * 10
                    logger.info("Added proxy latitude/longitude based on Shape area/length")
                
                data_source = "Nesto mortgage data"
                data_description = "Nesto mortgage application data with descriptive fields"
                    
            else:
                # For other data files, use memory-optimized loading
                logger.info(f"Loading data from {path} with memory optimization")
                try:
                    df = load_and_optimize_data(path)
                except Exception as mem_error:
                    logger.warning(f"Optimized loading failed: {mem_error}. Trying basic load.")
                    df = pd.read_csv(path)
                
                dataset_path = path
                log_memory_usage("After loading data")
                
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
                
            # Check memory usage and sample data if needed
            df = sample_data_if_needed(df, len(df))
            
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
log_memory_usage("Before data cleaning")

# For extremely memory constrained environments, only focus on essential cleaning
is_render = 'RENDER' in os.environ
memory_critical = is_render or is_memory_critical(threshold_mb=400)

# Delete unnecessary objects before proceeding
if 'rename_dict' in locals():
    del rename_dict
gc.collect()

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

# Make sure numeric columns are properly typed - use columns that actually exist
numeric_columns = []
for col in ['Income', 'Age', 'TOTPOP_CY', 'MEDDI_CY']:
    if col in df.columns:
        numeric_columns.append(col)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Only drop NAs if we have numeric columns to check
if numeric_columns:
    df = df.dropna(subset=numeric_columns)

logger.info(f"After cleaning: {df.shape[0]} records and {df.shape[1]} columns")

# Save cleaned data for reference
cleaned_data_path = 'data/cleaned_data.csv'
df.to_csv(cleaned_data_path, index=False)
logger.info("Cleaned data saved")

# Register cleaned dataset in version tracker
dataset_version_id = version_tracker.track_dataset({
    "path": cleaned_data_path,
    "description": f"Cleaned {data_description}",
    "source": data_source
})
logger.info(f"Registered cleaned dataset with version ID: {dataset_version_id}")

# Prepare features and target
print("Preparing data for training...")


# Exclude non-feature columns. 'ID' is the FSA (postal code) and should only be used as a descriptor, never as a model feature.

# Use total population as target variable since it's a meaningful demographic metric
TARGET_VARIABLE = 'TOTPOP_CY'  # Total population as target variable
exclude_cols = ['ID', TARGET_VARIABLE]
if 'latitude' in df.columns:
    exclude_cols.append('latitude')
if 'longitude' in df.columns:
    exclude_cols.append('longitude')

# Explicitly exclude legacy/geographic identifier fields and their mapped equivalents from model features
# These must never be used in analysis or inference
exclude_cols += [
    'OBJECTID', 'Shape__Area', 'Shape__Length',
    'Geographic_Area', 'Geographic_Length'
]

# Also exclude any non-numeric columns and object dtypes that XGBoost can't handle
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        exclude_cols.append(col)
        print(f"Excluding non-numeric column: {col} (dtype: {df[col].dtype})")

# Remove duplicates in exclude_cols and only drop columns that exist
exclude_cols = list(set([col for col in exclude_cols if col in df.columns]))

# Drop excluded columns for model training
X = df.drop(exclude_cols, axis=1)
print(f"Selected {len(X.columns)} numeric feature columns (after explicit exclusion of ID (FSA), OBJECTID, Shape__Area, Shape__Length, Geographic_Area, Geographic_Length)")
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
# Define model parameters (will be adjusted based on memory)
model_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'tree_method': 'hist',  # Use histogram-based algorithm (faster and uses less memory)
    'enable_categorical': True  # Enable categorical feature support
}

# Check memory and adjust model parameters if needed
model_params = reduce_model_complexity(model_params)

log_memory_usage("Before model training")

# Initialize CV metrics
cv_metrics = {}

# Use cross-validation to evaluate model robustness only if memory allows
if not is_memory_critical(MEMORY_CRITICAL):
    try:
        if len(X) >= 50:  # Only do cross-validation if we have enough data
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import KFold
            
            print("Performing cross-validation (reduced folds if memory limited)...")
            # Use fewer folds if memory is tight
            n_splits = 3 if is_memory_critical(MEMORY_WARNING) else 5
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # Create a reduced version of X for cross-validation if needed
            X_cv = X
            y_cv = y
            if is_memory_critical(MEMORY_WARNING) and len(X) > 10000:
                sample_size = min(10000, int(len(X) * 0.5))
                indices = np.random.choice(len(X), sample_size, replace=False)
                X_cv = X.iloc[indices]
                y_cv = y.iloc[indices]
                print(f"Using {len(X_cv)} samples for cross-validation to save memory")
            
            # Use model with reduced complexity for CV
            cv_model_params = reduce_model_complexity(model_params, max_mb=MEMORY_WARNING)
            model_cv = xgb.XGBRegressor(**cv_model_params)
            
            try:
                cv_scores = cross_val_score(model_cv, X_cv, y_cv, cv=cv, scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores)
                print(f"Cross-validation RMSE: {cv_rmse.mean():.4f} (±{cv_rmse.std():.4f})")
                
                # Add CV metrics to the metrics dictionary that will be defined later
                cv_metrics = {
                    "cv_rmse_mean": float(cv_rmse.mean()),
                    "cv_rmse_std": float(cv_rmse.std())
                }
                
                # Free memory used by CV
                del X_cv, y_cv, model_cv
                gc.collect()
                
            except Exception as e:
                print(f"Cross-validation failed: {str(e)}")
                print("Skipping cross-validation and proceeding with model training")
    except Exception as e:
        logger.warning(f"Error during cross-validation setup: {e}")
        print("Skipping cross-validation due to setup error")
else:
    logger.warning("Memory usage too high, skipping cross-validation")

# Train the final model on all training data
log_memory_usage("Before final model training")
model = xgb.XGBRegressor(**model_params)

# For very large datasets, use early stopping with a separate validation set
if len(X_train) > 20000 and not is_memory_critical(MEMORY_CRITICAL):
    # Create a validation set
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    model.fit(
        X_t, y_t,
        eval_set=[(X_v, y_v)],
        early_stopping_rounds=10,
        verbose=True
    )
    # Free memory used by separate validation set
    del X_t, X_v, y_t, y_v
    gc.collect()
else:
    # Regular training without validation set
    if MEMORY_OPTIMIZATION and is_memory_critical(MEMORY_CAUTION):
        logger.warning("Memory very low before training, reducing training size")
        # Use a smaller subset for training in extremely low memory conditions
        sample_size = min(3000 if 'RENDER' in os.environ else 5000, int(len(X_train) * 0.3))
        X_train_sample = X_train.sample(n=sample_size, random_state=42)
        y_train_sample = y_train.loc[X_train_sample.index]
        
        # Release the original dataframes to save memory
        del X_train, y_train
        gc.collect()
        model.fit(X_train_sample, y_train_sample)
        logger.info(f"Model trained on reduced dataset of {sample_size} samples")
        del X_train_sample, y_train_sample
        gc.collect()
    else:
        # Regular training on full training set
        model.fit(X_train, y_train)

log_memory_usage("After model training")

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

model_version_id = version_tracker.track_model({
    "model_path": model_path,
    "dataset_version_id": dataset_version_id,
    "feature_names_path": feature_names_path,
    "metrics": metrics
})

logger.info(f"Registered model with version ID: {model_version_id}")
logger.info("Done! Model and feature names saved.")

def train_and_save_model():
    """Function to train and save the model, to be called from other modules."""
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    logger.info("Starting model training process from train_and_save_model function...")
    log_memory_usage("Starting train_and_save_model function")
    
    # Initialize data version tracker
    version_tracker = DataVersionTracker()
    
    # Check if we're in Render's environment for stricter memory management
    is_render = 'RENDER' in os.environ
    
    # Try each path until we find a valid data file
    df = None
    dataset_path = None
    for path in DATA_PATHS:
        if os.path.exists(path):
            logger.info(f"Loading data from {path}...")
            try:
                if 'nesto_merge_0.csv' in path:
                    # Use memory-optimized loading for Nesto data
                    logger.info(f"Loading Nesto data from {path} with memory optimization")
                    try:
                        # Start with a smaller sample if in Render environment
                        sample_size = 15000 if is_render else None
                        df = load_and_optimize_data(path, low_memory=False, nrows=sample_size)
                    except Exception as mem_error:
                        logger.warning(f"Optimized loading failed: {mem_error}. Trying with minimal sample.")
                        sample_size = 10000 if is_render else 25000
                        df = load_and_optimize_data(path, low_memory=False, nrows=sample_size)
                    
                    dataset_path = path
                    log_memory_usage("After loading Nesto data")
                    
                    # Define the field mappings
                    from map_nesto_data import FIELD_MAPPINGS, TARGET_VARIABLE
                    
                    # Process in a memory-efficient way - map columns one at a time
                    # instead of creating a duplicate dataframe
                    rename_dict = {}
                    for source_col, target_col in FIELD_MAPPINGS.items():
                        if source_col in df.columns:
                            rename_dict[source_col] = target_col
                    
                    # Apply rename at once instead of creating a new dataframe
                    if rename_dict:
                        df = df.rename(columns=rename_dict)
                        logger.info(f"Renamed {len(rename_dict)} columns using field mappings")
                    
                    # Clean up to save memory
                    del rename_dict
                    gc.collect()
                    
                else:
                    # Direct loading of already processed data
                    logger.info(f"Loading processed data from {path}")
                    df = pd.read_csv(path)
                    dataset_path = path
                    
                # Basic data checks
                if df is not None and not df.empty:
                    logger.info(f"Successfully loaded data from {path} with shape {df.shape}")
                    break
            except Exception as e:
                logger.warning(f"Error loading data from {path}: {str(e)}")
                continue
    
    # If we couldn't find any valid data, exit
    if df is None or df.empty:
        logger.error("Could not load data from any of the specified paths")
        # Create minimal dataset for testing
        logger.warning("Creating minimal test dataset")
        df = pd.DataFrame({
            "Mortgage_Approvals": [10, 20, 30, 40, 50],
            "Income": [50000, 60000, 70000, 80000, 90000],
            "Age": [30, 40, 50, 60, 70],
            "Homeownership_Pct": [70, 80, 60, 50, 40]
        })
        dataset_path = 'data/minimal_test_data.csv'
        df.to_csv(dataset_path, index=False)
    
    # Register the dataset
    dataset_version_id = version_tracker.track_dataset({
        "path": dataset_path,
        "description": "Dataset for model training",
        "source": "Train_and_save_model function"
    })
    
    # Identify target variable
    target_variable = os.getenv('DEFAULT_TARGET', 'Mortgage_Approvals')
    
    # Check if target exists in dataframe
    if target_variable not in df.columns:
        logger.warning(f"Target variable {target_variable} not found in DataFrame columns")
        possible_targets = [col for col in df.columns if 'approv' in col.lower() or 'fund' in col.lower()]
        if possible_targets:
            target_variable = possible_targets[0]
            logger.info(f"Using {target_variable} as target instead")
        else:
            # Just use the last column as target if we can't find an approval-related column
            target_variable = df.columns[-1]
            logger.warning(f"Using {target_variable} as target (fallback)")
    
    # Basic preprocessing
    logger.info(f"Preprocessing data with target variable: {target_variable}")
    y = df[target_variable].copy()
    X = df.drop(columns=[target_variable], errors='ignore')
    
    # Remove identifier/non-useful columns
    id_columns = [col for col in X.columns if 'id' in col.lower() or 'code' in col.lower()]
    X = X.drop(columns=id_columns, errors='ignore')
    
    # Remove columns with too many missing values
    missing_threshold = 0.5
    missing_cols = [col for col in X.columns if X[col].isna().mean() > missing_threshold]
    if missing_cols:
        logger.info(f"Removing {len(missing_cols)} columns with >{missing_threshold*100}% missing values")
        X = X.drop(columns=missing_cols)
    
    # Fill remaining missing values
    logger.info("Filling missing values")
    X = X.fillna(X.mean())
    
    # Handle any columns that are still objects
    object_cols = X.select_dtypes(include=['object']).columns
    for col in object_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define model parameters
    model_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # Train the model
    logger.info("Training XGBoost model...")
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
    
    model_version_id = version_tracker.track_model({
        "model_path": model_path,
        "dataset_version_id": dataset_version_id,
        "feature_names_path": feature_names_path,
        "metrics": metrics
    })
    
    logger.info(f"Registered model with version ID: {model_version_id}")
    logger.info("Done! Model and feature names saved.")
    
    return model, X.columns.tolist()

# Execute main code only if script is run directly
if __name__ == "__main__":
    # Original script execution continues here
    pass