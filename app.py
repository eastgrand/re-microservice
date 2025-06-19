from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import pandas as pd
import numpy as np
import logging
import traceback
import os
import pickle
import shap
import xgboost as xgb
import re
import json

# Import the master schema and data processing function
from map_nesto_data import MASTER_SCHEMA, TARGET_VARIABLE, load_and_preprocess_data, initialize_schema
# Import the real analysis function
from enhanced_analysis_worker import enhanced_analysis_worker

# --- Redis/RQ Imports for Async Jobs ---
import redis

# --- Custom NaN-Safe JSON Handler ---
def safe_jsonify(data, status_code=200):
    """Safe jsonify that handles NaN values by converting them to None"""
    def convert_nan(obj):
        if isinstance(obj, dict):
            return {key: convert_nan(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_nan(item) for item in obj]
        elif isinstance(obj, float):
            if pd.isna(obj) or np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        elif pd.isna(obj):  # For pandas NA values
            return None
        return obj
    
    # Convert NaN values to None throughout the data structure
    safe_data = convert_nan(data)
    response = jsonify(safe_data)
    response.status_code = status_code
    return response

# --- Flask App Initialization ---
app = Flask(__name__)
# Enable CORS for all routes, allowing the frontend to fetch the schema
CORS(app)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Paths and Globals for Model ---
MODEL_PATH = "models/xgboost_model.pkl"
FEATURE_NAMES_PATH = "models/feature_names.txt"
model = None
feature_names = None

# --- Data and Model Loading ---
df = None
AVAILABLE_COLUMNS = set()

def load_model_and_features():
    """Loads the XGBoost model and feature names from disk."""
    global model, feature_names
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Successfully loaded XGBoost model from {MODEL_PATH}")
        else:
            logger.error(f"Model file not found at {MODEL_PATH}. The worker will not be able to perform analysis.")

        if os.path.exists(FEATURE_NAMES_PATH):
            with open(FEATURE_NAMES_PATH, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            logger.info(f"Successfully loaded {len(feature_names)} feature names from {FEATURE_NAMES_PATH}")
        else:
            logger.error(f"Feature names file not found at {FEATURE_NAMES_PATH}.")

    except Exception as e:
        logger.error(f"An error occurred during model loading: {e}")
        logger.error(traceback.format_exc())

try:
    # Force regeneration of cleaned_data.csv on every startup to ensure consistency.
    logger.info("--- Forcing regeneration of cleaned_data.csv from master schema ---")
    load_and_preprocess_data()
    logger.info("--- Data regeneration complete. ---")

    # Load the newly generated dataset
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, 'data', 'cleaned_data.csv')
    df = pd.read_csv(data_path)
    AVAILABLE_COLUMNS = set(df.columns)
    logger.info("Successfully loaded fresh cleaned_data.csv.")
    logger.info(f"Available columns for analysis: {', '.join(sorted(list(AVAILABLE_COLUMNS)))}")
    
    # Initialize the dynamic schema
    initialize_schema(df)
    logger.info("Dynamic schema initialized with all available fields.")
except Exception as e:
    logger.error(f"FATAL: Could not load or process data on startup: {e}")
    logger.error(traceback.format_exc())
    df = None
    AVAILABLE_COLUMNS = set()

# Load the model on startup
load_model_and_features()

# === Analysis Worker Function ===
def analysis_worker(analysis_request):
    """
    This is the original, full-featured analysis worker. It performs real SHAP
    analysis using the loaded model.
    """
    try:
        logger.info(f"Worker starting real analysis for request: {analysis_request}")

        if df is None or model is None or feature_names is None:
            raise ValueError("Worker is missing essential resources (data, model, or features). Cannot proceed.")

        # Use the enhanced analysis worker function
        result = enhanced_analysis_worker(analysis_request)
        
        logger.info("Worker completed real analysis successfully.")
        return result

    except Exception as e:
        logger.error(f"An error occurred in the real analysis_worker: {e}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'message': 'An error occurred during SHAP analysis.',
            'error_details': str(e)
        }

# === API Endpoints ===

@app.route('/api/v1/schema', methods=['GET'])
def get_schema():
    """
    Exposes the complete data schema to the frontend.
    This includes all available fields from the dataset, not just the predefined ones.
    """
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded. Schema unavailable."}, 500)

    # Generate dynamic schema from actual data columns
    dynamic_schema = {}
    known_fields = []
    
    for col in df.columns:
        # Check if this field has a predefined mapping in MASTER_SCHEMA
        predefined = None
        for key, details in MASTER_SCHEMA.items():
            if details['canonical_name'] == col:
                predefined = details
                break
        
        if predefined:
            # Use predefined schema info
            dynamic_schema[col] = predefined
        else:
            # Generate schema info for unmapped fields
            dynamic_schema[col] = {
                'canonical_name': col,
                'raw_mapping': col,  # No mapping needed
                'aliases': [col.lower(), col.replace(' ', '_').lower()],
                'description': f'Data field: {col}',
                'type': 'numeric' if df[col].dtype in ['int64', 'float64'] else 'string'
            }
        
        known_fields.append(col)
    
    logger.info(f"Generated schema for {len(known_fields)} fields")
    
    return safe_jsonify({
        "fields": dynamic_schema,
        "known_fields": sorted(known_fields)
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint. It validates the request against available data columns
    and performs real SHAP analysis using the analysis worker.
    """
    if df is None:
        abort(500, description="Dataset not loaded. Cannot perform analysis.")

    # --- Request Validation ---
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")

    data = request.json
    target_variable = data.get('target_variable')
    matched_fields = data.get('matched_fields', [])
    
    # Resolve field names from aliases
    def resolve_field_name(field_name):
        logger.info(f"Attempting to resolve field: '{field_name}'")
        
        if field_name in AVAILABLE_COLUMNS:
            logger.info(f"Field '{field_name}' found directly in available columns")
            return field_name
        
        # Handle common field aliases that the frontend might use
        common_aliases = {
            'household_average_income': 'median_income',
            'household_income': 'median_income',
            'average_income': 'median_income',
            'income': 'median_income',
            'household_median_income': 'median_income',
            'disposable_household_income': 'disposable_income',
            'mortgage_approval': 'mortgage_approvals',
            'mortgage_approval_rate': 'mortgage_approvals',
            'approval_rate': 'mortgage_approvals',
        }
        
        if field_name.lower() in common_aliases:
            resolved_field = common_aliases[field_name.lower()]
            logger.info(f"Resolved common alias: '{field_name}' -> '{resolved_field}'")
            return resolved_field
        
        # Dynamic field resolution for all fields
        field_lower = field_name.lower()
        logger.info(f"Searching for field with lowercase: '{field_lower}'")
        
        # First, try exact match in MASTER_SCHEMA aliases
        for canonical_name, details in MASTER_SCHEMA.items():
            if field_lower in [alias.lower() for alias in details.get('aliases', [])]:
                logger.info(f"Found field '{field_name}' in MASTER_SCHEMA aliases, mapping to '{details['raw_mapping']}'")
                return details['raw_mapping']
        
        # If not found in aliases, try pattern matching for all fields
        for col in AVAILABLE_COLUMNS:
            col_lower = col.lower()
            
            # Direct match
            if field_lower == col_lower:
                logger.info(f"Direct match found: '{field_name}' -> '{col}'")
                return col
            
            # Try snake_case conversion of the column name (preserve hyphens as underscores)
            col_snake = re.sub(r'[^\w\s-]', '', col_lower)  # Keep hyphens
            col_snake = re.sub(r'[-\s]+', '_', col_snake.strip())  # Convert hyphens and spaces to underscores
            if field_lower == col_snake:
                logger.info(f"Snake case match found: '{field_name}' -> '{col}' (snake: '{col_snake}')")
                return col
            
            # Try without year prefixes
            col_clean = re.sub(r'\b(2024|2023|2022|2021|census|current|year|\$|%)\b', '', col_lower)
            col_clean = re.sub(r'[^\w\s-]', '', col_clean)  # Keep hyphens
            col_clean = re.sub(r'[-\s]+', '_', col_clean.strip())  # Convert hyphens and spaces to underscores
            if field_lower == col_clean:
                logger.info(f"Clean match found: '{field_name}' -> '{col}' (clean: '{col_clean}')")
                return col
            
            # For percentage fields, try with _pct suffix
            if field_name.endswith('_pct') and '(%)' in col:
                base_field = field_name[:-4]  # Remove _pct
                col_base = col.replace('(%)', '').strip()
                col_base_snake = re.sub(r'[^\w\s-]', '', col_base.lower())  # Keep hyphens
                col_base_snake = re.sub(r'[-\s]+', '_', col_base_snake.strip())  # Convert hyphens and spaces to underscores
                
                if base_field == col_base_snake:
                    logger.info(f"Percentage match found: '{field_name}' -> '{col}' (base: '{col_base_snake}')")
                    return col
                
                # Try without year prefixes
                col_base_clean = re.sub(r'\b(2024|2023|2022|2021)\b', '', col_base).strip()
                if col_base_clean:
                    col_base_clean_snake = re.sub(r'[^\w\s-]', '', col_base_clean.lower())  # Keep hyphens
                    col_base_clean_snake = re.sub(r'[-\s]+', '_', col_base_clean_snake.strip())  # Convert hyphens and spaces to underscores
                    if base_field == col_base_clean_snake:
                        logger.info(f"Clean percentage match found: '{field_name}' -> '{col}' (clean base: '{col_base_clean_snake}')")
                        return col
                    
                    # Try partial matching - check if the base field is contained in the clean snake case
                    if base_field in col_base_clean_snake:
                        # Additional validation: check if key terms match
                        base_terms = set(base_field.split('_'))
                        col_terms = set(col_base_clean_snake.split('_'))
                        if base_terms.issubset(col_terms):
                            logger.info(f"Partial percentage match found: '{field_name}' -> '{col}' (partial: '{base_field}' in '{col_base_clean_snake}')")
                            return col
            
            # Try partial matching for key terms
            field_terms = set(re.findall(r'\w+', field_lower))
            col_terms = set(re.findall(r'\w+', col_lower))
            
            # If field has specific housing/demographic terms, check for matches
            key_terms = {'single', 'detached', 'house', 'apartment', 'condominium', 'condo', 
                        'visible', 'minority', 'income', 'population', 'structure', 'type'}
            
            if field_terms & key_terms:  # If field contains key terms
                # Check if most important terms match
                important_field_terms = field_terms & key_terms
                important_col_terms = col_terms & key_terms
                
                if important_field_terms and important_field_terms.issubset(important_col_terms):
                    # Additional check for percentage fields
                    if (field_name.endswith('_pct') and '(%)' in col) or (not field_name.endswith('_pct') and '(%)' not in col):
                        logger.info(f"Key terms match found: '{field_name}' -> '{col}' (terms: {important_field_terms})")
                        return col
        
        logger.warning(f"No match found for field: '{field_name}'")
        return field_name
    
    # Resolve target variable and matched fields
    if target_variable:
        resolved_target = resolve_field_name(target_variable)
        if resolved_target != target_variable:
            logger.info(f"Resolved target variable '{target_variable}' to '{resolved_target}'")
            data['target_variable'] = resolved_target
            target_variable = resolved_target
    
    resolved_matched_fields = []
    for field in matched_fields:
        resolved_field = resolve_field_name(field)
        if resolved_field != field:
            logger.info(f"Resolved field '{field}' to '{resolved_field}'")
        resolved_matched_fields.append(resolved_field)
    
    if resolved_matched_fields != matched_fields:
        data['matched_fields'] = resolved_matched_fields
        matched_fields = resolved_matched_fields
    
    # Validate that the requested fields are present in our dataset
    all_requested_fields = [target_variable] + matched_fields
    unknown_fields = [field for field in all_requested_fields if field and field not in AVAILABLE_COLUMNS]

    if unknown_fields:
        available_sample = sorted(list(AVAILABLE_COLUMNS))[:20]  # Show first 20 available fields
        error_msg = f"Unknown metric fields requested: {', '.join(unknown_fields)}. Available fields include: {', '.join(available_sample)}... (total: {len(AVAILABLE_COLUMNS)} fields)"
        abort(400, description=error_msg)

    logger.info(f"Received analysis request with target '{target_variable}' and fields {matched_fields}")

    # --- Real Analysis ---
    # Call the real analysis worker function
    try:
        analysis_results = analysis_worker(data)
        
        # Ensure results have proper geographic identifiers for data joining
        if analysis_results.get('success') and 'results' in analysis_results:
            # Add FSA_ID field to each result for proper geographic joining
            for result in analysis_results['results']:
                if 'geo_id' in result and 'FSA_ID' not in result:
                    result['FSA_ID'] = result['geo_id']
                if 'ID' not in result and 'geo_id' in result:
                    result['ID'] = result['geo_id']
        
        logger.info(f"Analysis completed successfully. Returning {len(analysis_results.get('results', []))} results.")
        return safe_jsonify(analysis_results)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error(traceback.format_exc())
        abort(500, description=f"Analysis failed: {str(e)}")

@app.route('/factor-importance', methods=['POST'])
def calculate_factor_importance():
    """
    Calculate feature importance for predictive analysis using SHAP values.
    This endpoint is specifically designed for predictive queries like 
    "what factors predict high conversion rates?"
    """
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded. Cannot perform factor importance analysis."}, 500)
    
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")
    
    data = request.json
    target_field = data.get('target_field')
    features = data.get('features', [])  # If empty, use all available features
    method = data.get('method', 'shap')
    max_factors = data.get('max_factors', 10)
    
    logger.info(f"Factor importance request - target: {target_field}, method: {method}, max_factors: {max_factors}")
    
    try:
        # Validate target field
        if not target_field:
            abort(400, description="Missing required field: target_field")
        
        if target_field not in df.columns:
            abort(400, description=f"Target field '{target_field}' not found in dataset. Available columns: {', '.join(sorted(df.columns[:10]))}...")
        
        # Identify numeric features for analysis
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target field from features if present
        if target_field in numeric_features:
            numeric_features.remove(target_field)
        
        # Filter by requested features if provided
        if features:
            requested_features = [f for f in features if f in numeric_features]
            if requested_features:
                numeric_features = requested_features
            else:
                logger.warning(f"None of the requested features {features} are available numeric features")
        
        # Prepare X (features) and y (target)
        X = df[numeric_features].fillna(0)
        y = df[target_field].fillna(0)
        
        logger.info(f"Using {len(numeric_features)} features for factor importance analysis")
        
        # Train a simple model for SHAP analysis if model not available or doesn't match features
        if model is None or set(feature_names or []) != set(numeric_features):
            logger.info("Training new model for factor importance analysis")
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
            temp_model.fit(X_train, y_train)
            model_accuracy = temp_model.score(X_test, y_test)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(temp_model)
            shap_values = explainer.shap_values(X_test)
        else:
            logger.info("Using existing model for factor importance analysis")
            # Use existing model
            temp_model = model
            # Calculate model accuracy on current data
            model_accuracy = temp_model.score(X, y) if hasattr(temp_model, 'score') else 0.85
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(temp_model)
            # Use a sample for SHAP calculation to avoid memory issues
            sample_size = min(100, len(X))
            X_sample = X.sample(n=sample_size, random_state=42)
            shap_values = explainer.shap_values(X_sample)
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(0)
        
        # Calculate correlations
        correlations = X.corrwith(y).abs()
        
        # Combine and rank factors
        factors = []
        for i, feature in enumerate(numeric_features):
            # Find example areas with high values for this feature
            top_areas = df.nlargest(3, feature)
            example_areas = []
            
            # Try different ID columns for area names
            for id_col in ['CSDNAME', 'FSA_ID', 'DESCRIPTION', 'NAME', 'ID']:
                if id_col in top_areas.columns:
                    example_areas = top_areas[id_col].fillna('Unknown').tolist()
                    break
            
            if not example_areas:
                example_areas = [f'Area_{j+1}' for j in range(3)]
            
            factors.append({
                'name': feature,
                'importance': float(feature_importance[i]),
                'correlation': float(correlations.get(feature, 0)),
                'description': f"Factor analyzing {feature} impact on {target_field}",
                'shap_values': shap_values[:, i].tolist()[:10] if len(shap_values.shape) > 1 else [],  # First 10 SHAP values
                'example_areas': example_areas[:3]
            })
        
        # Sort by importance and limit
        factors.sort(key=lambda x: x['importance'], reverse=True)
        factors = factors[:max_factors]
        
        logger.info(f"Factor importance analysis completed. Found {len(factors)} factors.")
        
        result = {
            'factors': factors,
            'target_variable': target_field,
            'model_accuracy': float(model_accuracy),
            'method_used': method,
            'features_analyzed': len(numeric_features)
        }
        
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in factor importance calculation: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "error": f"Factor importance calculation failed: {str(e)}"
        }, 500)

@app.route('/feature-interactions', methods=['POST'])
def calculate_feature_interactions():
    """
    Calculate feature interactions using SHAP interaction values.
    This endpoint identifies how pairs of features work together to influence the target.
    """
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded. Cannot perform interaction analysis."}, 500)
    
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")
    
    data = request.json
    target_field = data.get('target_field')
    max_interactions = data.get('max_interactions', 10)
    interaction_threshold = data.get('interaction_threshold', 0.1)
    
    logger.info(f"Feature interaction request - target: {target_field}, max_interactions: {max_interactions}")
    
    try:
        # Validate target field
        if not target_field:
            abort(400, description="Missing required field: target_field")
        
        if target_field not in df.columns:
            abort(400, description=f"Target field '{target_field}' not found in dataset. Available columns: {', '.join(sorted(df.columns[:10]))}...")
        
        # Identify numeric features for analysis
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target field from features if present
        if target_field in numeric_features:
            numeric_features.remove(target_field)
        
        # Limit to top features to avoid combinatorial explosion
        # Use correlation as a filter to select most relevant features
        X_full = df[numeric_features].fillna(0)
        y = df[target_field].fillna(0)
        
        correlations = X_full.corrwith(y).abs().sort_values(ascending=False)
        top_features = correlations.head(8).index.tolist()  # Limit to top 8 most correlated features
        
        X = X_full[top_features]
        
        logger.info(f"Using {len(top_features)} top features for interaction analysis: {top_features}")
        
        # Train a model for SHAP interaction analysis
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        temp_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
        temp_model.fit(X_train, y_train)
        model_performance = temp_model.score(X_test, y_test)
        
        # Calculate SHAP interaction values
        explainer = shap.TreeExplainer(temp_model)
        
        # Use a smaller sample for interaction calculation (computationally expensive)
        sample_size = min(50, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)
        
        # Get SHAP interaction values (3D array: [samples, features, features])
        shap_interaction_values = explainer.shap_interaction_values(X_sample)
        
        # Calculate interaction strengths
        interactions = []
        n_features = len(top_features)
        
        for i in range(n_features):
            for j in range(i + 1, n_features):  # Only upper triangle to avoid duplicates
                feature_1 = top_features[i]
                feature_2 = top_features[j]
                
                # Extract interaction values for this feature pair
                interaction_values = shap_interaction_values[:, i, j]
                
                # Calculate interaction strength (mean absolute interaction)
                interaction_strength = np.abs(interaction_values).mean()
                
                # Determine interaction type based on the pattern
                correlation_12 = X[[feature_1, feature_2]].corr().iloc[0, 1]
                
                # Simple heuristic for interaction type
                if interaction_strength > interaction_threshold:
                    if abs(correlation_12) > 0.3:
                        if correlation_12 > 0:
                            interaction_type = "synergistic"  # Both features tend to amplify each other
                        else:
                            interaction_type = "antagonistic"  # Features partially cancel each other
                    else:
                        interaction_type = "conditional"  # Effect depends on context
                    
                    # Find example areas where this interaction is strongest
                    interaction_abs = np.abs(interaction_values)
                    strongest_indices = np.argsort(interaction_abs)[-3:]  # Top 3 strongest interactions
                    
                    example_areas = []
                    for idx in strongest_indices:
                        sample_idx = X_sample.iloc[idx].name
                        
                        # Try different ID columns for area names
                        area_name = "Unknown"
                        for id_col in ['CSDNAME', 'FSA_ID', 'DESCRIPTION', 'NAME', 'ID']:
                            if id_col in df.columns and sample_idx in df.index:
                                area_name = str(df.loc[sample_idx, id_col])
                                if area_name and area_name != 'nan':
                                    break
                        
                        example_areas.append({
                            'area': area_name,
                            'feature_1_value': float(X_sample.iloc[idx][feature_1]),
                            'feature_2_value': float(X_sample.iloc[idx][feature_2]),
                            'interaction_strength': float(interaction_values[idx])
                        })
                    
                    interactions.append({
                        'feature_1': feature_1,
                        'feature_2': feature_2,
                        'interaction_strength': float(interaction_strength),
                        'interaction_type': interaction_type,
                        'correlation_between_features': float(correlation_12),
                        'individual_importance_1': float(correlations[feature_1]),
                        'individual_importance_2': float(correlations[feature_2]),
                        'example_areas': example_areas[:3],
                        'sample_interaction_values': interaction_values.tolist()[:10]  # First 10 for debugging
                    })
        
        # Sort by interaction strength and limit
        interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
        interactions = interactions[:max_interactions]
        
        # Calculate summary statistics
        strong_interactions = [i for i in interactions if i['interaction_strength'] > 0.2]
        total_interactions = len([i for i in interactions if i['interaction_strength'] > interaction_threshold])
        
        logger.info(f"Feature interaction analysis completed. Found {len(interactions)} interactions, {len(strong_interactions)} strong interactions.")
        
        result = {
            'interactions': interactions,
            'target_variable': target_field,
            'model_performance': float(model_performance),
            'features_analyzed': top_features,
            'total_interactions': total_interactions,
            'strong_interactions_count': len(strong_interactions),
            'interaction_threshold': interaction_threshold
        }
        
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in feature interaction calculation: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "error": f"Feature interaction calculation failed: {str(e)}"
        }, 500)

@app.route('/outlier-detection', methods=['POST'])
def detect_outliers():
    """
    Detect outliers using SHAP values to explain what makes areas statistically unusual.
    This endpoint identifies areas that deviate significantly from the norm and explains
    which features contribute most to their outlier behavior.
    """
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded. Cannot perform outlier detection."}, 500)
    
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")
    
    data = request.json
    target_field = data.get('target_field')
    outlier_method = data.get('method', 'isolation_forest')  # 'isolation_forest', 'iqr', 'zscore'
    outlier_threshold = data.get('outlier_threshold', 0.1)  # Fraction of data to consider outliers
    max_outliers = data.get('max_outliers', 20)
    explain_outliers = data.get('explain_outliers', True)
    
    logger.info(f"Outlier detection request - target: {target_field}, method: {outlier_method}, threshold: {outlier_threshold}")
    
    try:
        # Resolve target field name
        actual_target_field = resolve_field_name(target_field)
        if actual_target_field not in df.columns:
            return safe_jsonify({
                "error": f"Target field '{target_field}' not found in dataset. Available fields: {list(df.columns)}"
            }, 400)
        
        # Prepare data for outlier detection
        X_full = df.copy()
        
        # Remove non-numeric columns for outlier detection
        exclude_columns = ['ID', 'CSDNAME', 'FSA_ID', 'DESCRIPTION', 'NAME']
        for col in exclude_columns:
            if col in X_full.columns:
                X_full = X_full.drop(col, axis=1)
        
        # Handle missing values
        X_full = X_full.fillna(X_full.median())
        
        # Get target variable
        if actual_target_field not in X_full.columns:
            return safe_jsonify({
                "error": f"Target field '{actual_target_field}' not found after preprocessing"
            }, 400)
        
        y = X_full[actual_target_field]
        X = X_full.drop(actual_target_field, axis=1)
        
        # Select relevant features for analysis
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_columns]
        
        logger.info(f"Using {len(numeric_columns)} features for outlier detection")
        
        # Detect outliers using specified method
        outlier_indices = []
        outlier_scores = []
        
        if outlier_method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            detector = IsolationForest(contamination=outlier_threshold, random_state=42)
            outlier_predictions = detector.fit_predict(X)
            outlier_scores_raw = detector.score_samples(X)
            
            # Get outlier indices (where prediction is -1)
            outlier_indices = np.where(outlier_predictions == -1)[0]
            # Convert scores to positive anomaly scores (lower values = more anomalous)
            outlier_scores = -outlier_scores_raw[outlier_indices]
            
        elif outlier_method == 'iqr':
            # Use IQR method on target variable
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (y < lower_bound) | (y > upper_bound)
            outlier_indices = np.where(outlier_mask)[0]
            # Calculate outlier scores based on distance from bounds
            outlier_scores = np.where(y[outlier_indices] < lower_bound, 
                                    lower_bound - y[outlier_indices], 
                                    y[outlier_indices] - upper_bound)
            
        elif outlier_method == 'zscore':
            # Use Z-score method on target variable
            z_threshold = 2.5
            z_scores = np.abs((y - y.mean()) / y.std())
            outlier_indices = np.where(z_scores > z_threshold)[0]
            outlier_scores = z_scores[outlier_indices]
        
        else:
            return safe_jsonify({
                "error": f"Unknown outlier detection method: {outlier_method}. Use 'isolation_forest', 'iqr', or 'zscore'"
            }, 400)
        
        # Limit number of outliers returned
        if len(outlier_indices) > max_outliers:
            # Sort by outlier score and take top outliers
            sorted_indices = np.argsort(outlier_scores)[::-1][:max_outliers]
            outlier_indices = outlier_indices[sorted_indices]
            outlier_scores = outlier_scores[sorted_indices]
        
        logger.info(f"Detected {len(outlier_indices)} outliers using {outlier_method} method")
        
        # Prepare outlier results
        outliers = []
        shap_explanations = {}
        
        if explain_outliers and len(outlier_indices) > 0:
            # Calculate SHAP values for outliers to explain their anomalous behavior
            try:
                # Train a model for SHAP explanation
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import train_test_split
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                temp_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
                temp_model.fit(X_train, y_train)
                model_performance = temp_model.score(X_test, y_test)
                
                # Calculate SHAP values for outliers
                explainer = shap.TreeExplainer(temp_model)
                outlier_data = X.iloc[outlier_indices]
                shap_values = explainer.shap_values(outlier_data)
                
                # Calculate feature importance for outliers
                shap_explanations = {}
                for i, idx in enumerate(outlier_indices):
                    feature_contributions = {}
                    for j, feature in enumerate(X.columns):
                        feature_contributions[feature] = float(shap_values[i, j])
                    
                    # Sort by absolute contribution
                    sorted_contributions = sorted(feature_contributions.items(), 
                                                key=lambda x: abs(x[1]), reverse=True)
                    
                    shap_explanations[int(idx)] = {
                        'top_contributing_features': sorted_contributions[:5],
                        'all_contributions': feature_contributions
                    }
                
                logger.info(f"SHAP explanations calculated for {len(outlier_indices)} outliers")
                
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {str(e)}. Continuing without explanations.")
                model_performance = None
        
        # Build outlier results
        for i, idx in enumerate(outlier_indices):
            # Get area information
            area_info = {}
            area_name = "Unknown"
            
            # Try different ID columns for area identification
            for id_col in ['CSDNAME', 'FSA_ID', 'DESCRIPTION', 'NAME', 'ID']:
                if id_col in df.columns and idx in df.index:
                    area_value = df.iloc[idx][id_col]
                    if area_value and str(area_value) != 'nan':
                        area_info[id_col.lower()] = str(area_value)
                        if area_name == "Unknown":
                            area_name = str(area_value)
            
            outlier_data = {
                'rank': i + 1,
                'area_name': area_name,
                'area_info': area_info,
                'target_value': float(y.iloc[idx]),
                'outlier_score': float(outlier_scores[i]),
                'outlier_type': 'high' if y.iloc[idx] > y.median() else 'low',
                'deviation_from_mean': float(y.iloc[idx] - y.mean()),
                'percentile_rank': float((y <= y.iloc[idx]).mean() * 100),
                'data_index': int(idx)
            }
            
            # Add SHAP explanation if available
            if int(idx) in shap_explanations:
                outlier_data['shap_explanation'] = shap_explanations[int(idx)]
                
                # Create human-readable explanation
                top_features = shap_explanations[int(idx)]['top_contributing_features'][:3]
                explanation_text = f"This area is anomalous primarily due to: "
                explanations = []
                for feature, contribution in top_features:
                    direction = "unusually high" if contribution > 0 else "unusually low"
                    explanations.append(f"{direction} {feature.lower().replace('_', ' ')}")
                outlier_data['explanation'] = explanation_text + ", ".join(explanations) + "."
            
            outliers.append(outlier_data)
        
        # Calculate dataset statistics for context
        dataset_stats = {
            'total_areas': len(df),
            'outliers_detected': len(outliers),
            'outlier_percentage': (len(outliers) / len(df)) * 100,
            'target_mean': float(y.mean()),
            'target_median': float(y.median()),
            'target_std': float(y.std()),
            'target_min': float(y.min()),
            'target_max': float(y.max())
        }
        
        result = {
            'outliers': outliers,
            'target_variable': actual_target_field,
            'detection_method': outlier_method,
            'outlier_threshold': outlier_threshold,
            'dataset_statistics': dataset_stats,
            'model_performance': model_performance if explain_outliers else None,
            'has_shap_explanations': explain_outliers and len(shap_explanations) > 0
        }
        
        logger.info(f"Outlier detection completed successfully. Found {len(outliers)} outliers.")
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in outlier detection: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "error": f"Outlier detection failed: {str(e)}"
        }, 500)

@app.route('/scenario-analysis', methods=['POST'])
def analyze_scenario():
    """
    Perform scenario analysis ("what-if" analysis) using SHAP values to predict
    how changes in key features would affect the target variable.
    """
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded. Cannot perform scenario analysis."}, 500)
    
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")
    
    data = request.json
    target_field = data.get('target_field')
    scenarios = data.get('scenarios', [])  # List of scenario definitions
    base_area = data.get('base_area')  # Optional: specific area to analyze
    top_features = data.get('top_features', 10)  # Number of features to consider
    
    logger.info(f"Scenario analysis request - target: {target_field}, scenarios: {len(scenarios)}")
    
    try:
        # Resolve target field name
        actual_target_field = resolve_field_name(target_field)
        if actual_target_field not in df.columns:
            return safe_jsonify({
                "error": f"Target field '{target_field}' not found in dataset. Available fields: {list(df.columns)}"
            }, 400)
        
        # Prepare data for scenario analysis
        X_full = df.copy()
        
        # Remove non-numeric columns
        exclude_columns = ['ID', 'CSDNAME', 'FSA_ID', 'DESCRIPTION', 'NAME']
        for col in exclude_columns:
            if col in X_full.columns:
                X_full = X_full.drop(col, axis=1)
        
        # Handle missing values
        X_full = X_full.fillna(X_full.median())
        
        # Get target variable and features
        if actual_target_field not in X_full.columns:
            return safe_jsonify({
                "error": f"Target field '{actual_target_field}' not found after preprocessing"
            }, 400)
        
        y = X_full[actual_target_field]
        X = X_full.drop(actual_target_field, axis=1)
        
        # Select numeric features only
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_columns]
        
        logger.info(f"Using {len(numeric_columns)} features for scenario analysis")
        
        # Train a model for scenario prediction
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
        model.fit(X_train, y_train)
        model_performance = model.score(X_test, y_test)
        
        # Calculate SHAP values for baseline understanding
        explainer = shap.TreeExplainer(model)
        sample_size = min(100, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        # Calculate feature importance for scenario guidance
        feature_importance = {}
        for i, feature in enumerate(X.columns):
            importance = np.abs(shap_values[:, i]).mean()
            feature_importance[feature] = importance
        
        # Sort features by importance and get top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_feature_names = [f[0] for f in sorted_features[:top_features]]
        
        # Determine base scenario
        if base_area:
            # Find specific area if provided
            base_data = None
            for id_col in ['CSDNAME', 'FSA_ID', 'DESCRIPTION', 'NAME', 'ID']:
                if id_col in df.columns:
                    matching_rows = df[df[id_col].astype(str).str.contains(str(base_area), case=False, na=False)]
                    if len(matching_rows) > 0:
                        base_idx = matching_rows.index[0]
                        base_data = X.loc[base_idx].copy()
                        break
            
            if base_data is None:
                return safe_jsonify({
                    "error": f"Base area '{base_area}' not found in dataset"
                }, 400)
        else:
            # Use median values as baseline
            base_data = X.median()
        
        baseline_prediction = model.predict([base_data])[0]
        
        # Analyze scenarios
        scenario_results = []
        
        # If no scenarios provided, create default percentage change scenarios
        if not scenarios:
            scenarios = [
                {"name": "10% Income Increase", "changes": {"Median_Income": 1.1}},
                {"name": "20% Income Increase", "changes": {"Median_Income": 1.2}},
                {"name": "10% Education Improvement", "changes": {"Education_Score": 1.1}},
                {"name": "Combined Income+Education Boost", "changes": {"Median_Income": 1.15, "Education_Score": 1.1}}
            ]
        
        for scenario in scenarios:
            scenario_name = scenario.get('name', 'Unnamed Scenario')
            changes = scenario.get('changes', {})
            
            # Apply changes to base data
            modified_data = base_data.copy()
            applied_changes = []
            
            for feature, change_value in changes.items():
                # Try to find the feature in the dataset
                matching_feature = None
                for col in X.columns:
                    if feature.lower() in col.lower() or col.lower() in feature.lower():
                        matching_feature = col
                        break
                
                if matching_feature:
                    original_value = modified_data[matching_feature]
                    
                    # Apply change (can be multiplier or absolute change)
                    if isinstance(change_value, dict):
                        # Complex change specification
                        change_type = change_value.get('type', 'multiply')
                        change_amount = change_value.get('value', 1.0)
                        
                        if change_type == 'multiply':
                            new_value = original_value * change_amount
                        elif change_type == 'add':
                            new_value = original_value + change_amount
                        elif change_type == 'set':
                            new_value = change_amount
                        else:
                            new_value = original_value * change_amount
                    else:
                        # Simple multiplier
                        if change_value > 0 and change_value < 10:  # Assume it's a multiplier
                            new_value = original_value * change_value
                        else:  # Assume it's an absolute value
                            new_value = change_value
                    
                    modified_data[matching_feature] = new_value
                    applied_changes.append({
                        'feature': matching_feature,
                        'original_value': float(original_value),
                        'new_value': float(new_value),
                        'change_percent': float(((new_value - original_value) / original_value) * 100) if original_value != 0 else 0
                    })
                else:
                    logger.warning(f"Feature '{feature}' not found in dataset for scenario '{scenario_name}'")
            
            # Predict new outcome
            scenario_prediction = model.predict([modified_data])[0]
            impact = scenario_prediction - baseline_prediction
            impact_percent = (impact / baseline_prediction * 100) if baseline_prediction != 0 else 0
            
            # Calculate SHAP values for this scenario to understand the impact
            scenario_shap = explainer.shap_values([modified_data])
            baseline_shap = explainer.shap_values([base_data])
            
            # Calculate SHAP difference to understand what drove the change
            shap_differences = {}
            for i, feature in enumerate(X.columns):
                shap_diff = scenario_shap[0, i] - baseline_shap[0, i]
                shap_differences[feature] = float(shap_diff)
            
            # Sort by absolute contribution to change
            top_contributors = sorted(shap_differences.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            
            scenario_result = {
                'scenario_name': scenario_name,
                'baseline_prediction': float(baseline_prediction),
                'scenario_prediction': float(scenario_prediction),
                'predicted_impact': float(impact),
                'impact_percentage': float(impact_percent),
                'applied_changes': applied_changes,
                'top_contributing_factors': [
                    {
                        'feature': feature,
                        'shap_contribution': contribution,
                        'contribution_percentage': (abs(contribution) / sum(abs(v) for v in shap_differences.values())) * 100 if sum(abs(v) for v in shap_differences.values()) > 0 else 0
                    }
                    for feature, contribution in top_contributors
                ],
                'feasibility_assessment': assess_scenario_feasibility(applied_changes, X)
            }
            
            scenario_results.append(scenario_result)
        
        # Calculate overall insights
        best_scenario = max(scenario_results, key=lambda x: x['predicted_impact']) if scenario_results else None
        worst_scenario = min(scenario_results, key=lambda x: x['predicted_impact']) if scenario_results else None
        
        # Suggest additional scenarios based on top features
        suggested_scenarios = []
        for feature, importance in sorted_features[:3]:
            feature_median = X[feature].median()
            feature_std = X[feature].std()
            
            # Suggest modest improvement scenarios
            suggested_scenarios.append({
                'name': f"Improve {feature.replace('_', ' ')} by 1 Standard Deviation",
                'rationale': f"This feature has high predictive power (importance: {importance:.3f})",
                'changes': {feature: feature_median + feature_std},
                'expected_feasibility': 'medium'
            })
        
        result = {
            'scenarios': scenario_results,
            'baseline_info': {
                'target_field': actual_target_field,
                'baseline_prediction': float(baseline_prediction),
                'base_area': base_area if base_area else 'Dataset median',
                'model_performance': float(model_performance)
            },
            'insights': {
                'best_scenario': best_scenario['scenario_name'] if best_scenario else None,
                'best_impact': float(best_scenario['predicted_impact']) if best_scenario else 0,
                'worst_scenario': worst_scenario['scenario_name'] if worst_scenario else None,
                'worst_impact': float(worst_scenario['predicted_impact']) if worst_scenario else 0,
                'most_impactful_features': top_feature_names[:5]
            },
            'suggested_scenarios': suggested_scenarios[:3],
            'feature_importance': {feature: float(importance) for feature, importance in sorted_features[:10]}
        }
        
        logger.info(f"Scenario analysis completed successfully. Analyzed {len(scenario_results)} scenarios.")
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in scenario analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "error": f"Scenario analysis failed: {str(e)}"
        }, 500)

def assess_scenario_feasibility(applied_changes, X):
    """
    Assess how feasible a scenario is based on the changes required
    """
    if not applied_changes:
        return 'unknown'
    
    total_change_percent = sum(abs(change['change_percent']) for change in applied_changes)
    avg_change_percent = total_change_percent / len(applied_changes)
    
    if avg_change_percent < 10:
        return 'high'
    elif avg_change_percent < 25:
        return 'medium'
    else:
        return 'low'

@app.route('/threshold-analysis', methods=['POST'])
def analyze_thresholds():
    """
    Perform threshold analysis to identify critical inflection points where behavior changes.
    Uses SHAP values to determine at what feature levels the target variable significantly changes.
    """
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded. Cannot perform threshold analysis."}, 500)
    
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")
    
    data = request.json
    target_field = data.get('target_field')
    threshold_features = data.get('features', [])  # Features to analyze thresholds for
    num_bins = data.get('num_bins', 10)  # Number of bins to analyze
    min_samples_per_bin = data.get('min_samples_per_bin', 50)
    
    try:
        logger.info(f"Starting threshold analysis for target: {target_field}")
        
        # Validate target field
        if target_field not in df.columns:
            return safe_jsonify({"error": f"Target field '{target_field}' not found in dataset"}, 400)
        
        # If no features specified, use all numeric features except target
        if not threshold_features:
            threshold_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                                if col != target_field and not col.startswith('geometry')]
        
        # Prepare features and target
        feature_cols = [col for col in threshold_features if col in df.columns]
        if not feature_cols:
            return safe_jsonify({"error": "No valid features found for threshold analysis"}, 400)
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_field].fillna(df[target_field].median())
        
        # Train model for SHAP analysis
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Analyze thresholds for each feature
        threshold_results = []
        
        for i, feature in enumerate(feature_cols):
            feature_values = X[feature].values
            feature_shap = shap_values[:, i]
            
            # Create bins for the feature
            min_val, max_val = feature_values.min(), feature_values.max()
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Analyze SHAP values in each bin
            bin_stats = []
            for j in range(len(bin_centers)):
                mask = (feature_values >= bin_edges[j]) & (feature_values < bin_edges[j + 1])
                if j == len(bin_centers) - 1:  # Include max value in last bin
                    mask = (feature_values >= bin_edges[j]) & (feature_values <= bin_edges[j + 1])
                
                if np.sum(mask) >= min_samples_per_bin:
                    bin_shap_mean = np.mean(feature_shap[mask])
                    bin_target_mean = np.mean(y.values[mask])
                    bin_sample_count = np.sum(mask)
                    
                    bin_stats.append({
                        'bin_center': float(bin_centers[j]),
                        'bin_range': [float(bin_edges[j]), float(bin_edges[j + 1])],
                        'shap_impact': float(bin_shap_mean),
                        'target_mean': float(bin_target_mean),
                        'sample_count': int(bin_sample_count)
                    })
            
            # Find inflection points (where SHAP impact changes significantly)
            inflection_points = []
            if len(bin_stats) >= 3:
                shap_impacts = [stat['shap_impact'] for stat in bin_stats]
                
                # Calculate rate of change
                for k in range(1, len(shap_impacts) - 1):
                    rate_change = abs(shap_impacts[k + 1] - shap_impacts[k - 1]) / 2
                    if rate_change > np.std(shap_impacts) * 0.5:  # Significant change threshold
                        inflection_points.append({
                            'threshold_value': bin_stats[k]['bin_center'],
                            'shap_change_rate': float(rate_change),
                            'target_impact': bin_stats[k]['target_mean'],
                            'confidence': min(1.0, rate_change / np.std(shap_impacts))
                        })
            
            # Find optimal threshold (point of maximum positive impact)
            optimal_threshold = None
            if bin_stats:
                max_impact_idx = np.argmax([stat['shap_impact'] for stat in bin_stats])
                optimal_threshold = {
                    'value': bin_stats[max_impact_idx]['bin_center'],
                    'impact': bin_stats[max_impact_idx]['shap_impact'],
                    'target_value': bin_stats[max_impact_idx]['target_mean']
                }
            
            threshold_results.append({
                'feature': feature,
                'bin_analysis': bin_stats,
                'inflection_points': sorted(inflection_points, key=lambda x: x['confidence'], reverse=True)[:3],
                'optimal_threshold': optimal_threshold,
                'feature_importance': float(np.mean(np.abs(feature_shap)))
            })
        
        # Sort by feature importance
        threshold_results.sort(key=lambda x: x['feature_importance'], reverse=True)
        
        # Generate insights
        insights = {
            'most_critical_feature': threshold_results[0]['feature'] if threshold_results else None,
            'total_inflection_points': sum(len(result['inflection_points']) for result in threshold_results),
            'features_with_clear_thresholds': len([r for r in threshold_results if r['inflection_points']]),
            'recommended_actions': []
        }
        
        # Generate recommendations
        for result in threshold_results[:3]:  # Top 3 features
            if result['optimal_threshold']:
                insights['recommended_actions'].append(
                    f"For {result['feature']}: target values above {result['optimal_threshold']['value']:.2f} "
                    f"for maximum impact on {target_field}"
                )
        
        result = {
            "threshold_analysis": threshold_results,
            "insights": insights,
            "model_performance": {
                "r2_score": float(model.score(X, y)),
                "feature_count": len(feature_cols),
                "sample_count": len(X)
            },
            "target_variable": target_field,
            "analysis_parameters": {
                "num_bins": num_bins,
                "min_samples_per_bin": min_samples_per_bin,
                "features_analyzed": feature_cols
            }
        }
        
        logger.info(f"Threshold analysis completed successfully. Analyzed {len(feature_cols)} features.")
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in threshold analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "error": f"Threshold analysis failed: {str(e)}"
        }, 500)

@app.route('/segment-profiling', methods=['POST'])
def profile_segments():
    """
    Perform segment profiling to characterize different groups (e.g., high vs low performers).
    Uses SHAP values to explain what makes each segment unique.
    """
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded. Cannot perform segment profiling."}, 500)
    
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")
    
    data = request.json
    target_field = data.get('target_field')
    segment_method = data.get('method', 'percentile')  # 'percentile', 'kmeans', 'custom'
    num_segments = data.get('num_segments', 3)
    percentile_thresholds = data.get('percentile_thresholds', [33, 67])  # For percentile method
    
    try:
        logger.info(f"Starting segment profiling for target: {target_field}")
        
        # Validate target field
        if target_field not in df.columns:
            return safe_jsonify({"error": f"Target field '{target_field}' not found in dataset"}, 400)
        
        # Prepare features
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col != target_field and not col.startswith('geometry')]
        
        if not feature_cols:
            return safe_jsonify({"error": "No numeric features found for profiling"}, 400)
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_field].fillna(df[target_field].median())
        
        # Create segments based on method
        if segment_method == 'percentile':
            thresholds = np.percentile(y, percentile_thresholds)
            segments = np.digitize(y, thresholds)
            segment_labels = [f"Low (≤{thresholds[0]:.1f})", f"Medium ({thresholds[0]:.1f}-{thresholds[1]:.1f})", f"High (>{thresholds[1]:.1f})"]
        elif segment_method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_segments, random_state=42)
            segments = kmeans.fit_predict(X)
            segment_labels = [f"Cluster {i+1}" for i in range(num_segments)]
        else:
            return safe_jsonify({"error": "Invalid segment method. Use 'percentile' or 'kmeans'"}, 400)
        
        # Train model for SHAP analysis
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Analyze each segment
        segment_profiles = []
        
        for segment_id in range(len(segment_labels)):
            mask = segments == segment_id
            if np.sum(mask) < 10:  # Skip segments with too few samples
                continue
            
            segment_X = X[mask]
            segment_y = y[mask]
            segment_shap = shap_values[mask]
            
            # Calculate segment characteristics
            feature_profiles = []
            for i, feature in enumerate(feature_cols):
                feature_mean = float(np.mean(segment_X[feature]))
                feature_std = float(np.std(segment_X[feature]))
                shap_mean = float(np.mean(segment_shap[:, i]))
                shap_std = float(np.std(segment_shap[:, i]))
                
                # Compare to overall population
                overall_mean = float(np.mean(X[feature]))
                difference_from_mean = feature_mean - overall_mean
                relative_difference = (difference_from_mean / overall_mean * 100) if overall_mean != 0 else 0
                
                feature_profiles.append({
                    'feature': feature,
                    'segment_mean': feature_mean,
                    'segment_std': feature_std,
                    'overall_mean': overall_mean,
                    'difference_from_overall': float(difference_from_mean),
                    'relative_difference_percent': float(relative_difference),
                    'shap_impact': shap_mean,
                    'shap_variability': shap_std,
                    'importance_rank': 0  # Will be set later
                })
            
            # Rank features by SHAP impact magnitude
            feature_profiles.sort(key=lambda x: abs(x['shap_impact']), reverse=True)
            for i, profile in enumerate(feature_profiles):
                profile['importance_rank'] = i + 1
            
            # Identify distinguishing characteristics
            distinguishing_features = [
                profile for profile in feature_profiles[:5]  # Top 5
                if abs(profile['relative_difference_percent']) > 10  # >10% difference from overall
            ]
            
            segment_profiles.append({
                'segment_id': segment_id,
                'segment_label': segment_labels[segment_id],
                'sample_count': int(np.sum(mask)),
                'target_statistics': {
                    'mean': float(np.mean(segment_y)),
                    'median': float(np.median(segment_y)),
                    'std': float(np.std(segment_y)),
                    'min': float(np.min(segment_y)),
                    'max': float(np.max(segment_y))
                },
                'feature_profiles': feature_profiles,
                'distinguishing_features': distinguishing_features,
                'segment_description': f"Segment with {len(distinguishing_features)} key distinguishing features"
            })
        
        # Generate comparative insights
        insights = {
            'total_segments': len(segment_profiles),
            'segment_method': segment_method,
            'key_differentiators': [],
            'segment_rankings': []
        }
        
        # Find features that differentiate segments most
        if len(segment_profiles) >= 2:
            for feature in feature_cols:
                feature_means = [seg['target_statistics']['mean'] for seg in segment_profiles]
                if len(set(feature_means)) > 1:  # Has variation
                    max_diff = max(feature_means) - min(feature_means)
                    insights['key_differentiators'].append({
                        'feature': feature,
                        'max_difference': float(max_diff),
                        'segments_affected': len(segment_profiles)
                    })
        
        # Rank segments by target performance
        insights['segment_rankings'] = sorted(
            [{'segment': seg['segment_label'], 'performance': seg['target_statistics']['mean']} 
             for seg in segment_profiles],
            key=lambda x: x['performance'], reverse=True
        )
        
        result = {
            "segment_profiles": segment_profiles,
            "insights": insights,
            "model_performance": {
                "r2_score": float(model.score(X, y)),
                "feature_count": len(feature_cols),
                "total_samples": len(X)
            },
            "target_variable": target_field,
            "analysis_parameters": {
                "segment_method": segment_method,
                "num_segments": num_segments,
                "features_analyzed": feature_cols
            }
        }
        
        logger.info(f"Segment profiling completed successfully. Created {len(segment_profiles)} segments.")
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in segment profiling: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "error": f"Segment profiling failed: {str(e)}"
        }, 500)

@app.route('/comparative-analysis', methods=['POST'])
def compare_groups():
    """
    Perform comparative analysis between different groups (e.g., urban vs rural).
    Uses SHAP values to explain differences between groups.
    """
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded. Cannot perform comparative analysis."}, 500)
    
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")
    
    data = request.json
    target_field = data.get('target_field')
    grouping_field = data.get('grouping_field')  # Field to group by (e.g., 'urban_rural')
    comparison_groups = data.get('groups', [])  # Specific groups to compare
    
    try:
        logger.info(f"Starting comparative analysis for target: {target_field}, grouping: {grouping_field}")
        
        # Validate fields
        if target_field not in df.columns:
            return safe_jsonify({"error": f"Target field '{target_field}' not found in dataset"}, 400)
        
        if grouping_field not in df.columns:
            return safe_jsonify({"error": f"Grouping field '{grouping_field}' not found in dataset"}, 400)
        
        # Get available groups
        available_groups = df[grouping_field].dropna().unique()
        if not comparison_groups:
            comparison_groups = list(available_groups)[:4]  # Limit to 4 groups for clarity
        
        # Validate comparison groups
        valid_groups = [group for group in comparison_groups if group in available_groups]
        if len(valid_groups) < 2:
            return safe_jsonify({"error": "Need at least 2 valid groups for comparison"}, 400)
        
        # Prepare features
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col != target_field and col != grouping_field and not col.startswith('geometry')]
        
        if not feature_cols:
            return safe_jsonify({"error": "No numeric features found for comparison"}, 400)
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[target_field].fillna(df[target_field].median())
        
        # Train model for SHAP analysis
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Analyze each group
        group_analyses = []
        
        for group in valid_groups:
            mask = df[grouping_field] == group
            if np.sum(mask) < 10:  # Skip groups with too few samples
                continue
            
            group_X = X[mask]
            group_y = y[mask]
            group_shap = shap_values[mask]
            
            # Calculate group statistics
            feature_stats = []
            for i, feature in enumerate(feature_cols):
                feature_mean = float(np.mean(group_X[feature]))
                feature_median = float(np.median(group_X[feature]))
                feature_std = float(np.std(group_X[feature]))
                shap_mean = float(np.mean(group_shap[:, i]))
                
                feature_stats.append({
                    'feature': feature,
                    'mean': feature_mean,
                    'median': feature_median,
                    'std': feature_std,
                    'shap_impact': shap_mean
                })
            
            group_analyses.append({
                'group': str(group),
                'sample_count': int(np.sum(mask)),
                'target_statistics': {
                    'mean': float(np.mean(group_y)),
                    'median': float(np.median(group_y)),
                    'std': float(np.std(group_y)),
                    'min': float(np.min(group_y)),
                    'max': float(np.max(group_y))
                },
                'feature_statistics': feature_stats
            })
        
        # Perform pairwise comparisons
        comparisons = []
        for i in range(len(group_analyses)):
            for j in range(i + 1, len(group_analyses)):
                group_a = group_analyses[i]
                group_b = group_analyses[j]
                
                # Compare target variable
                target_diff = group_a['target_statistics']['mean'] - group_b['target_statistics']['mean']
                target_percent_diff = (target_diff / group_b['target_statistics']['mean'] * 100) if group_b['target_statistics']['mean'] != 0 else 0
                
                # Compare features
                feature_differences = []
                for k, feature in enumerate(feature_cols):
                    a_stat = group_a['feature_statistics'][k]
                    b_stat = group_b['feature_statistics'][k]
                    
                    mean_diff = a_stat['mean'] - b_stat['mean']
                    percent_diff = (mean_diff / b_stat['mean'] * 100) if b_stat['mean'] != 0 else 0
                    shap_diff = a_stat['shap_impact'] - b_stat['shap_impact']
                    
                    feature_differences.append({
                        'feature': feature,
                        'mean_difference': float(mean_diff),
                        'percent_difference': float(percent_diff),
                        'shap_difference': float(shap_diff),
                        'significance': 'high' if abs(percent_diff) > 20 else 'medium' if abs(percent_diff) > 10 else 'low'
                    })
                
                # Sort by significance
                feature_differences.sort(key=lambda x: abs(x['percent_difference']), reverse=True)
                
                comparisons.append({
                    'group_a': group_a['group'],
                    'group_b': group_b['group'],
                    'target_difference': {
                        'absolute': float(target_diff),
                        'percent': float(target_percent_diff),
                        'direction': 'higher' if target_diff > 0 else 'lower'
                    },
                    'feature_differences': feature_differences[:10],  # Top 10 differences
                    'key_differentiators': [fd for fd in feature_differences[:5] if fd['significance'] in ['high', 'medium']]
                })
        
        # Generate insights
        insights = {
            'groups_compared': [analysis['group'] for analysis in group_analyses],
            'total_comparisons': len(comparisons),
            'most_different_groups': None,
            'least_different_groups': None,
            'common_differentiators': []
        }
        
        if comparisons:
            # Find most and least different groups
            comparison_scores = [(comp['group_a'], comp['group_b'], abs(comp['target_difference']['percent'])) 
                               for comp in comparisons]
            comparison_scores.sort(key=lambda x: x[2], reverse=True)
            
            insights['most_different_groups'] = {
                'groups': [comparison_scores[0][0], comparison_scores[0][1]],
                'difference_percent': comparison_scores[0][2]
            }
            insights['least_different_groups'] = {
                'groups': [comparison_scores[-1][0], comparison_scores[-1][1]],
                'difference_percent': comparison_scores[-1][2]
            }
            
            # Find common differentiating features
            feature_mentions = {}
            for comp in comparisons:
                for diff in comp['key_differentiators']:
                    feature = diff['feature']
                    if feature not in feature_mentions:
                        feature_mentions[feature] = 0
                    feature_mentions[feature] += 1
            
            insights['common_differentiators'] = [
                {'feature': feature, 'frequency': count}
                for feature, count in sorted(feature_mentions.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
        
        result = {
            "group_analyses": group_analyses,
            "pairwise_comparisons": comparisons,
            "insights": insights,
            "model_performance": {
                "r2_score": float(model.score(X, y)),
                "feature_count": len(feature_cols),
                "total_samples": len(X)
            },
            "target_variable": target_field,
            "grouping_variable": grouping_field,
            "analysis_parameters": {
                "groups_analyzed": valid_groups,
                "features_analyzed": feature_cols
            }
        }
        
        logger.info(f"Comparative analysis completed successfully. Compared {len(valid_groups)} groups.")
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in comparative analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "error": f"Comparative analysis failed: {str(e)}"
        }, 500)

# === Error Handlers ===

@app.errorhandler(400)
def bad_request(error):
    logger.error(f"Bad Request: {error.description}")
    response = safe_jsonify({'error': 'Bad Request', 'message': error.description}, 400)
    return response

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal Server Error: {traceback.format_exc()}")
    response = safe_jsonify({'error': 'Internal Server Error', 'message': 'An unexpected error occurred.'}, 500)
    return response

if __name__ == '__main__':
    # Note: Running in debug mode is not recommended for production
    app.run(debug=True, port=5001)

""" 
# ---------------------------------------------------------------------------
# (Optional) ASYNC ANALYSIS ENDPOINTS – Disabled by default
# ---------------------------------------------------------------------------
# The following Flask routes implement the original asynchronous pattern using
# Redis + RQ.  They are **commented out** to keep the current deployment
# simple and synchronous, but can be re-enabled easily if long-running jobs or
# high request concurrency make it necessary again.
#
# Requirements:
#   pip install rq redis
#   export REDIS_URL=redis://:password@hostname:port/0
#
# How to enable:
#   1. Uncomment the code below.
#   2. Start an RQ worker:  rq worker shap-queue --url $REDIS_URL
#   3. Point the frontend to POST /submit_analysis instead of /analyze.
# ---------------------------------------------------------------------------

"""

# from rq import Queue
# from uuid import uuid4
#
# # --- Redis connection & queue ---
# redis_url = os.getenv('REDIS_URL')
# if redis_url:
#     redis_conn = redis.from_url(redis_url)
#     job_queue = Queue('shap-queue', connection=redis_conn)
#     logger.info('Redis queue initialised for async analysis.')
# else:
#     redis_conn = None
#     job_queue = None
#     logger.warning('REDIS_URL not set – async analysis endpoints will refuse requests.')
#
# @app.route('/submit_analysis', methods=['POST'])
# def submit_analysis():
#     """Enqueue a long-running SHAP analysis job and return a job_id."""
#     if job_queue is None:
#         abort(503, description='Async analysis service not configured (REDIS_URL missing).')
#
#     data = request.json or {}
#     job_id = str(uuid4())
#     job = job_queue.enqueue(analysis_worker, data, job_id=job_id)
#     logger.info(f'Enqueued analysis job {job_id}')
#     return jsonify({'job_id': job_id, 'status': job.get_status()}), 202
#
# @app.route('/job_status/<job_id>', methods=['GET'])
# def job_status(job_id):
#     """Check the status or retrieve the result of a queued analysis job."""
#     if job_queue is None:
#         abort(503, description='Async analysis service not configured.')
#
#     job = job_queue.fetch_job(job_id)
#     if job is None:
#         abort(404, description='Job ID not found')
#
#     if job.is_finished:
#         return jsonify({'status': 'completed', 'result': job.result})
#     if job.is_failed:
#         return jsonify({'status': 'failed', 'error': str(job.exc_info)}), 500
#
#     return jsonify({'status': job.get_status()}), 202

# ---------------------------------------------------------------------------
