import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
import os
import gc
import time
from tqdm import tqdm

print("🚀 Starting SHAP pre-calculation for entire dataset...")

# Load the 83-feature model (go back to the smart feature selection)
print("📊 Loading dataset and model...")
df = pd.read_csv('data/nesto_merge_0.csv')  # Use the original dataset with brand fields
print(f"Dataset shape: {df.shape}")
print(f"Available brand fields: {[col for col in df.columns if 'MP30034A_B' in col or 'MP30029A_B' in col]}")

# Load the 83-feature model we created earlier
try:
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    print(f"Model loaded with {len(feature_names)} features")
except:
    print("❌ Model not found. Running create_reduced_model.py first...")
    # Run the 83-feature creation script
    exec(open('create_reduced_model.py').read())
    
    # Load the created model
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]

print(f"Using {len(feature_names)} features for SHAP calculation")

# Prepare the data exactly as the model expects
print("🔧 Preparing data...")
# Make sure we have all the features the model expects
available_features = [f for f in feature_names if f in df.columns]
missing_features = [f for f in feature_names if f not in df.columns]

if missing_features:
    print(f"⚠️ Missing features: {missing_features}")
    print("Using only available features...")
    feature_names = available_features

# Prepare X data - only use numeric features
X = df[feature_names].copy()

# Convert to numeric and handle non-numeric columns
for col in X.columns:
    if X[col].dtype == 'object':
        # Skip non-numeric columns or convert if possible
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            print(f"Dropping non-numeric column: {col}")
            X = X.drop(col, axis=1)
            feature_names = [f for f in feature_names if f != col]

# Fill NaN values with column medians for numeric columns only
numeric_cols = X.select_dtypes(include=[np.number]).columns
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

print(f"Data prepared: {X.shape}")
print(f"Features: {feature_names[:5]}... (showing first 5)")

# Create SHAP explainer
print("🧠 Creating SHAP explainer...")
explainer = shap.TreeExplainer(model)

# Pre-calculate SHAP values in batches to manage memory
print("⚡ Computing SHAP values in batches...")
batch_size = 100  # Process 100 rows at a time
total_rows = len(X)
num_batches = (total_rows + batch_size - 1) // batch_size

all_shap_values = []
batch_times = []

for i in tqdm(range(0, total_rows, batch_size), desc="Processing batches"):
    batch_start_time = time.time()
    
    # Get batch
    end_idx = min(i + batch_size, total_rows)
    batch_X = X.iloc[i:end_idx].copy()
    
    # Compute SHAP values for batch
    try:
        batch_shap = explainer.shap_values(batch_X, check_additivity=False)
        all_shap_values.append(batch_shap)
        
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        
        if (i // batch_size + 1) % 5 == 0:  # Every 5 batches
            avg_time = np.mean(batch_times[-5:])
            print(f"  Batch {i//batch_size + 1}/{num_batches} - Avg time: {avg_time:.2f}s")
        
        # Force garbage collection
        gc.collect()
        
    except Exception as e:
        print(f"❌ Error in batch {i//batch_size + 1}: {str(e)}")
        # Create zero array as fallback
        fallback_shap = np.zeros((len(batch_X), len(feature_names)))
        all_shap_values.append(fallback_shap)

# Combine all SHAP values
print("🔗 Combining SHAP values...")
if len(all_shap_values) > 1:
    shap_values = np.concatenate(all_shap_values, axis=0)
else:
    shap_values = all_shap_values[0]

print(f"Final SHAP values shape: {shap_values.shape}")

# Create comprehensive results dataframe
print("💾 Creating results dataframe...")
results_data = {
    'ID': df['ID'].values,
    'TOTPOP_CY': df['TOTPOP_CY'].values,  # Use total population instead of conversion rate
}

# Add SHAP values for each feature
for i, feature in enumerate(feature_names):
    results_data[f'shap_{feature}'] = shap_values[:, i]

# Add original feature values for context  
for feature in feature_names:
    if feature in df.columns:
        results_data[f'value_{feature}'] = df[feature].fillna(df[feature].median() if df[feature].dtype in ['int64', 'float64'] else 0).values

# Create results DataFrame
results_df = pd.DataFrame(results_data)

# Save the pre-calculated SHAP values
print("💾 Saving pre-calculated SHAP values...")
os.makedirs('precalculated', exist_ok=True)

# Save as compressed pickle for fastest loading
results_df.to_pickle('precalculated/shap_values.pkl.gz', compression='gzip')

# Also save as CSV for inspection
sample_df = results_df.head(100)  # Just first 100 rows for CSV
sample_df.to_csv('precalculated/shap_values_sample.csv', index=False)

# Save metadata
metadata = {
    'total_rows': len(results_df),
    'features': feature_names,
    'model_features': len(feature_names),
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'batch_size': batch_size,
    'total_batches': num_batches,
    'avg_batch_time': np.mean(batch_times) if batch_times else 0
}

with open('precalculated/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print(f"✅ Pre-calculation complete!")
print(f"📈 Processed {len(results_df):,} rows")
print(f"🎯 Features: {len(feature_names)}")
print(f"⏱️ Avg batch time: {np.mean(batch_times):.2f}s")
print(f"💾 Files saved:")
print(f"   - precalculated/shap_values.pkl.gz ({os.path.getsize('precalculated/shap_values.pkl.gz') / (1024*1024):.1f} MB)")
print(f"   - precalculated/shap_values_sample.csv (for inspection)")
print(f"   - precalculated/metadata.pkl")

print("\n🎉 SHAP values are now pre-calculated and ready for instant querying!") 