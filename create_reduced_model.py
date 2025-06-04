import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os

# Load the dataset
print("Loading dataset...")
dataset_path = 'data/nesto_merge_0.csv'
if not os.path.exists(dataset_path):
    dataset_path = 'data/cleaned_data.csv'

df = pd.read_csv(dataset_path)
print(f"Dataset loaded with shape: {df.shape}")
print(f"Total columns: {len(df.columns)}")

# Get all columns except excluded ones
exclude_cols = ['OBJECTID', 'ID', 'Shape__Area', 'Shape__Length']
analysis_cols = [col for col in df.columns if col not in exclude_cols]

print(f"Columns available for analysis: {len(analysis_cols)}")

# Smart feature selection: prefer percentage columns over raw values
selected_features = []
used_base_names = set()

# First pass: identify and select percentage columns
for col in analysis_cols:
    if ' (%)' in col:
        # This is a percentage column
        base_name = col.replace(' (%)', '')
        selected_features.append(col)
        used_base_names.add(base_name)
        print(f"Selected percentage: {col}")

print(f"Found {len(selected_features)} percentage columns")

# Second pass: for non-percentage columns, only include if no percentage version exists
for col in analysis_cols:
    if ' (%)' not in col and col not in selected_features:
        # Check if this column has a percentage counterpart
        potential_pct_col = col + ' (%)'
        if potential_pct_col not in df.columns:
            # No percentage version exists, so include the raw column
            selected_features.append(col)
            print(f"Selected raw value (no % version): {col}")

print(f"Final feature count: {len(selected_features)}")
print(f"Reduced from {len(analysis_cols)} to {len(selected_features)} features")

# Remove target column if it exists in features
target_candidates = ['CONVERSION_RATE', 'SUM_FUNDED', 'FREQUENCY']
target_col = None

for candidate in target_candidates:
    if candidate in df.columns:
        target_col = candidate
        if candidate in selected_features:
            selected_features.remove(candidate)
        break

if target_col is None:
    print("ERROR: No target column found")
    exit(1)

print(f"Using target column: {target_col}")
print(f"Final feature count after removing target: {len(selected_features)}")

# Prepare data - only use numeric columns
numeric_features = []
for col in selected_features:
    if pd.api.types.is_numeric_dtype(df[col]):
        numeric_features.append(col)
    else:
        print(f"Skipping non-numeric column: {col}")

print(f"Numeric features selected: {len(numeric_features)}")

if len(numeric_features) < 5:
    print("ERROR: Too few numeric features available")
    exit(1)

# Prepare training data
X = df[numeric_features].copy()
y = df[target_col].copy()

# Handle missing values
print("Handling missing values...")
X = X.fillna(X.median())
y = y.fillna(y.median())

print(f"Training data shape: {X.shape}")
print(f"Target data shape: {y.shape}")

# Train reduced model with actual dataset column names
print("Training XGBoost model with actual dataset column names...")
model = xgb.XGBRegressor(
    n_estimators=100,  # Reasonable size
    max_depth=4,       # Not too deep
    learning_rate=0.1,
    random_state=42,
    tree_method='hist'  # Memory efficient
)

model.fit(X, y)

# Evaluate model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_eval = xgb.XGBRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, tree_method='hist'
)
model_eval.fit(X_train, y_train)
y_pred = model_eval.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model performance - RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Save the model with actual dataset column names
print("Saving model with actual dataset column names...")
with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save feature names (actual dataset column names)
with open('models/feature_names.txt', 'w') as f:
    for feature in numeric_features:
        f.write(f"{feature}\n")

print(f"✅ Model saved with {len(numeric_features)} actual dataset features")
print("Files updated:")
print("- models/xgboost_model.pkl (overwritten with reduced model)")
print("- models/feature_names.txt (updated with actual column names)")
print(f"Selected features include:")
for i, feat in enumerate(numeric_features[:10]):
    print(f"  {i+1}. {feat}")
if len(numeric_features) > 10:
    print(f"  ... and {len(numeric_features) - 10} more features") 