import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
import os
import gc
import time
from tqdm import tqdm
import json

# Define different model configurations
MODEL_CONFIGS = {
    'conversion': {
        'target': 'CONVERSION_RATE',
        'features': 'all',  # Use all 83 features
        'description': 'Predicts mortgage conversion rates'
    },
    'volume': {
        'target': 'SUM_FUNDED', 
        'features': 'financial',  # Focus on financial features
        'description': 'Predicts total loan volume'
    },
    'frequency': {
        'target': 'FREQUENCY',
        'features': 'demographic',  # Focus on demographic features  
        'description': 'Predicts application frequency'
    },
    'demographic_analysis': {
        'target': 'CONVERSION_RATE',
        'features': 'demographic_only',
        'description': 'Conversion analysis focused on demographics'
    },
    'geographic_analysis': {
        'target': 'CONVERSION_RATE', 
        'features': 'geographic_only',
        'description': 'Conversion analysis focused on geography'
    }
}

# Feature selection strategies
def get_features_by_type(df, feature_type):
    """Select features based on analysis type"""
    all_features = [col for col in df.columns if col not in ['ID', 'CONVERSION_RATE', 'SUM_FUNDED', 'FREQUENCY']]
    
    if feature_type == 'all':
        return all_features
    elif feature_type == 'demographic':
        # Prioritize demographic features but include others
        demo_features = [col for col in all_features if any(x in col.lower() for x in 
                        ['visible minority', 'population', 'age', 'married', 'divorced', 'single'])]
        other_features = [col for col in all_features if col not in demo_features]
        return demo_features + other_features[:20]  # Top demographic + some others
    elif feature_type == 'demographic_only':
        # Only demographic features
        return [col for col in all_features if any(x in col.lower() for x in 
                ['visible minority', 'population', 'age', 'married', 'divorced', 'single', 'maintainer'])]
    elif feature_type == 'financial':
        # Financial features priority
        fin_features = [col for col in all_features if any(x in col.lower() for x in 
                       ['income', 'mortgage', 'property', 'employment', 'financial', 'shelter', 'tax'])]
        other_features = [col for col in all_features if col not in fin_features]
        return fin_features + other_features[:15]  # Top financial + some others
    elif feature_type == 'geographic_only':
        # Only geographic/housing features
        return [col for col in all_features if any(x in col.lower() for x in 
                ['housing', 'construction', 'tenure', 'structure', 'condominium'])]
    else:
        return all_features

def train_model(df, target_col, features, model_name):
    """Train a model with specified target and features"""
    print(f"\n🔧 Training model: {model_name}")
    print(f"Target: {target_col}")
    print(f"Features: {len(features)}")
    
    # Prepare data
    X = df[features].fillna(df[features].median())
    y = df[target_col].fillna(df[target_col].median())
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    # Calculate performance
    predictions = model.predict(X)
    r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - y.mean()) ** 2))
    
    print(f"Model performance - R²: {r2:.4f}")
    
    return model, features, r2

def precalculate_shap_for_model(model, df, features, model_name, batch_size=100):
    """Pre-calculate SHAP values for a specific model"""
    print(f"\n⚡ Computing SHAP values for {model_name}...")
    
    # Prepare data
    X = df[features].fillna(df[features].median())
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Process in batches
    total_rows = len(X)
    num_batches = (total_rows + batch_size - 1) // batch_size
    all_shap_values = []
    
    for i in tqdm(range(0, total_rows, batch_size), desc=f"SHAP for {model_name}"):
        end_idx = min(i + batch_size, total_rows)
        batch_X = X.iloc[i:end_idx].copy()
        
        try:
            batch_shap = explainer.shap_values(batch_X, check_additivity=False)
            all_shap_values.append(batch_shap)
            gc.collect()
        except Exception as e:
            print(f"❌ Error in batch {i//batch_size + 1}: {str(e)}")
            fallback_shap = np.zeros((len(batch_X), len(features)))
            all_shap_values.append(fallback_shap)
    
    # Combine SHAP values
    if len(all_shap_values) > 1:
        shap_values = np.concatenate(all_shap_values, axis=0)
    else:
        shap_values = all_shap_values[0]
    
    return shap_values

def main():
    print("🚀 Starting multi-model SHAP pre-calculation...")
    
    # Load dataset
    df = pd.read_csv('data/nesto_merge_0.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Create output directories
    os.makedirs('precalculated/models', exist_ok=True)
    os.makedirs('models/multi', exist_ok=True)
    
    # Store all results
    all_results = {}
    
    for model_name, config in MODEL_CONFIGS.items():
        try:
            print(f"\n{'='*50}")
            print(f"Processing model: {model_name.upper()}")
            print(f"Description: {config['description']}")
            
            # Get features for this model type
            features = get_features_by_type(df, config['features'])
            target = config['target']
            
            # Skip if target doesn't exist
            if target not in df.columns:
                print(f"⚠️ Target {target} not found, skipping {model_name}")
                continue
            
            print(f"Selected {len(features)} features for {model_name}")
            
            # Train model
            model, final_features, r2_score = train_model(df, target, features, model_name)
            
            # Save model
            model_path = f'models/multi/{model_name}_model.pkl'
            features_path = f'models/multi/{model_name}_features.txt'
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            with open(features_path, 'w') as f:
                for feature in final_features:
                    f.write(f"{feature}\n")
            
            # Pre-calculate SHAP values
            shap_values = precalculate_shap_for_model(model, df, final_features, model_name)
            
            # Create results dataframe
            results_data = {
                'ID': df['ID'].values,
                target: df[target].fillna(df[target].median()).values,
            }
            
            # Add SHAP values
            for i, feature in enumerate(final_features):
                results_data[f'shap_{feature}'] = shap_values[:, i]
            
            # Add original feature values
            for feature in final_features:
                results_data[f'value_{feature}'] = df[feature].fillna(df[feature].median()).values
            
            results_df = pd.DataFrame(results_data)
            
            # Save pre-calculated SHAP data
            shap_output_path = f'precalculated/models/{model_name}_shap.pkl.gz'
            results_df.to_pickle(shap_output_path, compression='gzip')
            
            # Store metadata
            all_results[model_name] = {
                'target': target,
                'features': final_features,
                'feature_count': len(final_features),
                'r2_score': r2_score,
                'description': config['description'],
                'shap_file': shap_output_path,
                'model_file': model_path,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_rows': len(results_df)
            }
            
            print(f"✅ {model_name} completed successfully!")
            print(f"   - R² score: {r2_score:.4f}")
            print(f"   - Features: {len(final_features)}")
            print(f"   - SHAP file: {shap_output_path}")
            
        except Exception as e:
            print(f"❌ Error processing {model_name}: {str(e)}")
            continue
    
    # Save overall metadata
    metadata_path = 'precalculated/models/metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n🎉 Multi-model pre-calculation complete!")
    print(f"📊 Successfully processed {len(all_results)} models:")
    for name, info in all_results.items():
        print(f"   - {name}: {info['feature_count']} features, R² = {info['r2_score']:.4f}")
    print(f"📁 Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main() 