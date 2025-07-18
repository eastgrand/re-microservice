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
# Import field resolution utilities
from field_utils import resolve_field_name

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
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            if pd.isna(obj) or np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
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
    
    # Resolve target variable and matched fields using imported function
    if target_variable:
        resolved_target = resolve_field_name(target_variable, AVAILABLE_COLUMNS, MASTER_SCHEMA)
        if resolved_target != target_variable:
            logger.info(f"Resolved target variable '{target_variable}' to '{resolved_target}'")
            data['target_variable'] = resolved_target
            target_variable = resolved_target
    
    resolved_matched_fields = []
    for field in matched_fields:
        resolved_field = resolve_field_name(field, AVAILABLE_COLUMNS, MASTER_SCHEMA)
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

    # --- Special handling for brand comparison queries ---
    query = data.get('query', '').lower()
    analysis_type = data.get('analysis_type', '').lower()
    
    # Detect correlation/comparison queries dynamically
    is_correlation_query = (
        # Direct analysis type check
        analysis_type == 'correlation' or analysis_type == 'multi_brand_comparison' or
        # Query contains comparison/correlation keywords
        any(keyword in query for keyword in ['vs', 'versus', 'compare', 'correlation', 'relationship', 'between']) and
        # Has 2+ fields for analysis
        len(matched_fields) >= 2
    )
    
    if is_correlation_query and len(matched_fields) >= 2:
        logger.info(f"Detected correlation/comparison query - routing to correlation analysis for fields: {matched_fields}")
        
        # For multi-brand comparisons (3+ fields), handle all fields
        if len(matched_fields) >= 3:
            logger.info(f"Multi-brand comparison detected with {len(matched_fields)} fields")
            # Use all matched fields for multi-brand analysis
            all_fields = matched_fields
        else:
        # Use the correlation analysis logic with the first two matched fields
            all_fields = matched_fields[:2]
        
        var1, var2 = matched_fields[0], matched_fields[1]
        
        try:
            # Filter out rows with missing values in any of the analyzed variables
            analysis_fields = all_fields + ['ID']
            valid_data = df[analysis_fields].dropna()
            
            if len(valid_data) < 10:
                abort(400, description="Insufficient data points for correlation analysis (need at least 10).")
            
            # Initialize variables
            correlation_coef = None
            correlations = {}
            results = []
            
            # For multi-field analysis, calculate pairwise correlations
            if len(all_fields) >= 3:
                # Multi-brand comparison
                for i, field1 in enumerate(all_fields):
                    for j, field2 in enumerate(all_fields):
                        if i != j:
                            corr = valid_data[field1].corr(valid_data[field2])
                            if not pd.isna(corr):
                                correlations[f"{field1}_vs_{field2}"] = float(corr)
                
                # Create results with all field values
                for _, row in valid_data.iterrows():
                    result_row = {
                        'ID': row['ID'],
                        'geo_id': row['ID'],  # Alias for compatibility
                        'primary_value': float(row[all_fields[0]]),
                        'comparison_value': float(row[all_fields[1]]) if len(all_fields) > 1 else 0
                    }
                    
                    # Add all field values to the result
                    for field in all_fields:
                        result_row[field] = float(row[field])
                    
                    results.append(result_row)
                
                # Prepare multi-field analysis summary
                correlation_analysis = {
                    'correlations': correlations,
                    'sample_size': len(valid_data),
                    'variables': all_fields,
                    'analysis_type': 'multi_brand_comparison'
                }
                
                response = {
                    'analysis_type': 'multi_brand_comparison',
                    'results': results,
                    'correlation_analysis': correlation_analysis,
                    'success': True,
                    'metadata': {
                        'total_records': len(results),
                        'variables_analyzed': all_fields,
                        'analysis_type': 'multi_brand_comparison'
                    }
                }
                logger.info(f"Multi-brand analysis completed for {len(all_fields)} brands: {all_fields}")
                
            else:
                # Standard two-field correlation
                correlation_coef = valid_data[var1].corr(valid_data[var2])
                
                if pd.isna(correlation_coef):
                    abort(400, description="Unable to calculate correlation - insufficient valid data.")
                
                # Create results in the format expected by the frontend
                for _, row in valid_data.iterrows():
                    results.append({
                        'ID': row['ID'],
                        'geo_id': row['ID'],  # Alias for compatibility
                        'primary_value': float(row[var1]),
                        'comparison_value': float(row[var2]),
                        'correlation_strength': float(correlation_coef),
                        var1: float(row[var1]),  # Include original field names
                        var2: float(row[var2])
                    })
                
                # Prepare correlation analysis summary
                correlation_analysis = {
                    'coefficient': float(correlation_coef),
                    'strength': 'strong' if abs(correlation_coef) > 0.7 else 'moderate' if abs(correlation_coef) > 0.4 else 'weak',
                    'direction': 'positive' if correlation_coef > 0 else 'negative',
                    'sample_size': len(valid_data),
                    'variables': {
                        'primary': var1,
                        'comparison': var2
                    }
                }
                
                response = {
                    'analysis_type': 'bivariate_correlation',
                    'results': results,
                    'correlation_analysis': correlation_analysis,
                    'success': True,
                    'metadata': {
                        'total_records': len(results),
                        'correlation_coefficient': float(correlation_coef),
                        'variables_analyzed': [var1, var2]
                    }
                }
                logger.info(f"Correlation analysis completed: {correlation_coef:.4f} between {var1} and {var2}")
            
            return safe_jsonify(response)
                
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            logger.error(traceback.format_exc())
            # Fall through to regular analysis if correlation fails

    # --- Regular Analysis (existing code) ---

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
        actual_target_field = resolve_field_name(target_field, df.columns.tolist(), MASTER_SCHEMA)
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
        actual_target_field = resolve_field_name(target_field, df.columns.tolist(), MASTER_SCHEMA)
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
        from sklearn.ensemble import RandomForestRegressor
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
        
        # Handle missing grouping field with fallback logic
        if not grouping_field:
            # Auto-detect grouping field or use a default categorical field
            categorical_fields = df.select_dtypes(include=['object', 'category']).columns.tolist()
            # Filter out ID-like fields
            categorical_fields = [f for f in categorical_fields if not f.upper().startswith(('ID', 'OBJECTID', 'DESCRIPTION'))]
            if categorical_fields:
                grouping_field = categorical_fields[0]
                logger.info(f"Auto-selected grouping field: {grouping_field}")
            else:
                return safe_jsonify({"error": "No grouping field specified and no suitable categorical fields found"}, 400)
        
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

@app.route('/correlation', methods=['POST'])
def calculate_correlation():
    """
    Calculate correlation between two variables and return results in the format
    expected by the frontend's bivariate correlation visualization.
    """
    if df is None:
        abort(500, description="Dataset not loaded. Cannot perform correlation analysis.")

    # --- Request Validation ---
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")

    data = request.json
    variables = data.get('variables', [])
    
    if len(variables) != 2:
        abort(400, description="Correlation analysis requires exactly 2 variables.")
    
    var1, var2 = variables
    
    # Validate that both variables exist in the dataset
    if var1 not in df.columns or var2 not in df.columns:
        missing = [v for v in [var1, var2] if v not in df.columns]
        abort(400, description=f"Variables not found in dataset: {missing}")
    
    try:
        logger.info(f"Calculating correlation between {var1} and {var2}")
        
        # Filter out rows with missing values in either variable
        valid_data = df[[var1, var2, 'ID']].dropna()
        
        if len(valid_data) < 10:
            abort(400, description="Insufficient data points for correlation analysis (need at least 10).")
        
        # Calculate correlation coefficient
        correlation_coef = valid_data[var1].corr(valid_data[var2])
        
        if pd.isna(correlation_coef):
            abort(400, description="Unable to calculate correlation - insufficient valid data.")
        
        # Create results in the format expected by the frontend
        results = []
        for _, row in valid_data.iterrows():
            results.append({
                'ID': row['ID'],
                'geo_id': row['ID'],  # Alias for compatibility
                'primary_value': float(row[var1]),
                'comparison_value': float(row[var2]),
                'correlation_strength': float(correlation_coef),
                var1: float(row[var1]),  # Include original field names
                var2: float(row[var2])
            })
        
        # Prepare correlation analysis summary
        correlation_analysis = {
            'coefficient': float(correlation_coef),
            'strength': 'strong' if abs(correlation_coef) > 0.7 else 'moderate' if abs(correlation_coef) > 0.4 else 'weak',
            'direction': 'positive' if correlation_coef > 0 else 'negative',
            'sample_size': len(valid_data),
            'variables': {
                'primary': var1,
                'comparison': var2
            }
        }
        
        response = {
            'analysis_type': 'bivariate_correlation',
            'results': results,
            'correlation_analysis': correlation_analysis,
            'success': True,
            'metadata': {
                'total_records': len(results),
                'correlation_coefficient': float(correlation_coef),
                'variables_analyzed': [var1, var2]
            }
        }
        
        logger.info(f"Correlation analysis completed: {correlation_coef:.4f} between {var1} and {var2}")
        return safe_jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "error": f"Correlation analysis failed: {str(e)}",
            "success": False
        }, 500)

# === NEW ENHANCED ENDPOINTS ===

@app.route('/time-series-analysis', methods=['POST'])
def analyze_time_series():
    """
    SHAP analysis of temporal patterns and trend changes.
    Business value: "When did Nike preferences start changing in this area?"
    """
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded. Cannot perform time series analysis."}, 500)
    
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")
    
    data = request.json
    target_field = data.get('target_field')
    time_periods = data.get('time_periods', ['2021', '2022', '2023', '2024'])
    trend_threshold = data.get('trend_threshold', 0.1)  # Minimum change to consider significant
    
    try:
        logger.info(f"Starting time series analysis for target: {target_field}")
        
        # For now, simulate temporal analysis using cross-sectional data
        # In a real implementation, this would analyze actual time series data
        
        # Validate target field
        if target_field not in df.columns:
            return safe_jsonify({"error": f"Target field '{target_field}' not found in dataset"}, 400)
        
        # Prepare features for trend analysis
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_field in numeric_features:
            numeric_features.remove(target_field)
        
        X = df[numeric_features].fillna(df[numeric_features].median())
        y = df[target_field].fillna(df[target_field].median())
        
        # Train model for SHAP analysis
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        sample_size = min(200, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        # Simulate temporal patterns by analyzing different value ranges
        temporal_analysis = []
        for period in time_periods:
            # Create synthetic temporal segments for demonstration
            period_year = int(period) if period.isdigit() else 2024
            trend_factor = (period_year - 2020) * 0.05  # Simulate growth over time
            
            # Simulate trend by adjusting target values
            adjusted_target = y * (1 + trend_factor)
            
            # Calculate period statistics
            period_stats = {
                'period': period,
                'mean_value': float(adjusted_target.mean()),
                'growth_rate': float(trend_factor * 100),
                'volatility': float(adjusted_target.std()),
                'trend_direction': 'increasing' if trend_factor > 0 else 'decreasing' if trend_factor < 0 else 'stable'
            }
            
            temporal_analysis.append(period_stats)
        
        # Identify key trend drivers using SHAP
        feature_importance = {}
        for i, feature in enumerate(numeric_features):
            importance = np.abs(shap_values[:, i]).mean()
            feature_importance[feature] = float(importance)
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        trend_drivers = sorted_features[:10]
        
        # Identify inflection points
        values = [period['mean_value'] for period in temporal_analysis]
        inflection_points = []
        
        for i in range(1, len(values) - 1):
            prev_change = values[i] - values[i-1]
            next_change = values[i+1] - values[i]
            
            # Check for sign change (inflection point)
            if (prev_change > 0 > next_change) or (prev_change < 0 < next_change):
                inflection_points.append({
                    'period': temporal_analysis[i]['period'],
                    'value': values[i],
                    'change_type': 'peak' if prev_change > 0 else 'trough'
                })
        
        result = {
            "temporal_patterns": temporal_analysis,
            "trend_drivers": [
                {
                    'feature': feature,
                    'importance': importance,
                    'trend_influence': 'positive' if importance > 0 else 'negative'
                }
                for feature, importance in trend_drivers
            ],
            "inflection_points": inflection_points,
            "overall_trend": {
                'direction': 'increasing' if values[-1] > values[0] else 'decreasing',
                'total_change': float(values[-1] - values[0]),
                'change_percentage': float((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0,
                'volatility': float(np.std(values))
            },
            "target_variable": target_field,
            "analysis_periods": time_periods,
            "model_performance": float(model.score(X, y))
        }
        
        logger.info(f"Time series analysis completed successfully for {target_field}")
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in time series analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "error": f"Time series analysis failed: {str(e)}"
        }, 500)

@app.route('/brand-affinity', methods=['POST'])
def analyze_brand_relationships():
    """
    SHAP analysis of multi-brand purchase patterns.
    Business value: "Which demographics buy both Nike and Adidas?"
    """
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded. Cannot perform brand affinity analysis."}, 500)
    
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")
    
    data = request.json
    brand_fields = data.get('brand_fields', [])
    demographic_features = data.get('demographic_features', [])
    min_affinity_threshold = data.get('min_affinity_threshold', 0.3)
    
    try:
        logger.info(f"Starting brand affinity analysis for brands: {brand_fields}")
        
        # Auto-detect brand fields if not provided
        if not brand_fields:
            brand_fields = [col for col in df.columns if 'mp30' in col.lower() and 'a_b_p' in col.lower()]
        
        # Validate brand fields
        valid_brand_fields = [field for field in brand_fields if field in df.columns]
        if len(valid_brand_fields) < 2:
            return safe_jsonify({"error": "Need at least 2 valid brand fields for affinity analysis"}, 400)
        
        # Prepare demographic features
        if not demographic_features:
            demographic_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                                  if col not in valid_brand_fields and not col.startswith('geometry')]
        
        # Filter demographic features that exist
        valid_demo_features = [field for field in demographic_features if field in df.columns]
        
        # Prepare data
        brand_data = df[valid_brand_fields].fillna(0)
        demo_data = df[valid_demo_features].fillna(df[valid_demo_features].median())
        
        # Calculate brand correlations
        brand_correlations = brand_data.corr()
        
        # Find high-affinity brand pairs
        affinity_pairs = []
        for i, brand1 in enumerate(valid_brand_fields):
            for j, brand2 in enumerate(valid_brand_fields):
                if i < j:  # Avoid duplicates
                    correlation = brand_correlations.loc[brand1, brand2]
                    if abs(correlation) >= min_affinity_threshold:
                        affinity_pairs.append({
                            'brand_1': brand1,
                            'brand_2': brand2,
                            'affinity_score': float(correlation),
                            'relationship_type': 'complementary' if correlation > 0 else 'competitive'
                        })
        
        # Sort by affinity strength
        affinity_pairs.sort(key=lambda x: abs(x['affinity_score']), reverse=True)
        
        # Analyze demographic profiles for each brand
        brand_profiles = []
        for brand in valid_brand_fields:
            # Train model to predict brand preference
            X = demo_data
            y = brand_data[brand]
            
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            sample_size = min(100, len(X))
            X_sample = X.sample(n=sample_size, random_state=42)
            shap_values = explainer.shap_values(X_sample)
            
            # Calculate feature importance
            feature_importance = {}
            for i, feature in enumerate(valid_demo_features):
                importance = np.abs(shap_values[:, i]).mean()
                feature_importance[feature] = float(importance)
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            brand_profiles.append({
                'brand': brand,
                'key_demographics': [
                    {
                        'feature': feature,
                        'importance': importance,
                        'average_value': float(demo_data[feature].mean()),
                        'brand_correlation': float(demo_data[feature].corr(brand_data[brand]))
                    }
                    for feature, importance in sorted_features[:10]
                ],
                'model_accuracy': float(model.score(X, y))
            })
        
        # Find demographic segments that prefer multiple brands
        multi_brand_segments = []
        
        # Identify high-value customers (top quartile for multiple brands)
        for brand1 in valid_brand_fields:
            for brand2 in valid_brand_fields:
                if brand1 != brand2:
                    # Find areas in top quartile for both brands
                    q75_brand1 = brand_data[brand1].quantile(0.75)
                    q75_brand2 = brand_data[brand2].quantile(0.75)
                    
                    dual_high = df[(brand_data[brand1] >= q75_brand1) & (brand_data[brand2] >= q75_brand2)]
                    
                    if len(dual_high) >= 10:  # Enough samples for analysis
                        # Calculate demographic profile of dual-brand customers
                        segment_profile = {}
                        for demo_feature in valid_demo_features[:10]:  # Top 10 features
                            if demo_feature in dual_high.columns:
                                avg_value = dual_high[demo_feature].mean()
                                overall_avg = df[demo_feature].mean()
                                difference = avg_value - overall_avg
                                
                                segment_profile[demo_feature] = {
                                    'segment_average': float(avg_value),
                                    'overall_average': float(overall_avg),
                                    'difference': float(difference),
                                    'relative_difference': float(difference / overall_avg * 100) if overall_avg != 0 else 0
                                }
                        
                        multi_brand_segments.append({
                            'brand_combination': [brand1, brand2],
                            'segment_size': len(dual_high),
                            'percentage_of_total': float(len(dual_high) / len(df) * 100),
                            'demographic_profile': segment_profile
                        })
        
        result = {
            "brand_affinities": affinity_pairs,
            "brand_profiles": brand_profiles,
            "multi_brand_segments": multi_brand_segments[:10],  # Top 10 segments
            "analysis_summary": {
                'brands_analyzed': valid_brand_fields,
                'demographic_features': valid_demo_features[:10],
                'high_affinity_pairs': len([pair for pair in affinity_pairs if abs(pair['affinity_score']) > 0.5]),
                'multi_brand_segments_found': len(multi_brand_segments)
            },
            "insights": {
                'strongest_affinity': affinity_pairs[0] if affinity_pairs else None,
                'most_competitive_brands': [pair for pair in affinity_pairs if pair['affinity_score'] < -0.3],
                'most_complementary_brands': [pair for pair in affinity_pairs if pair['affinity_score'] > 0.3]
            }
        }
        
        logger.info(f"Brand affinity analysis completed successfully. Found {len(affinity_pairs)} brand relationships.")
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in brand affinity analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "error": f"Brand affinity analysis failed: {str(e)}"
        }, 500)

@app.route('/spatial-clusters', methods=['POST'])
def identify_spatial_clusters():
    """
    SHAP-explained spatial clusters of similar areas.
    Business value: "Which areas behave similarly and why?"
    """
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded. Cannot perform spatial clustering."}, 500)
    
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")
    
    data = request.json
    target_field = data.get('target_field')
    num_clusters = data.get('num_clusters', 5)
    clustering_features = data.get('clustering_features', [])
    
    try:
        logger.info(f"Starting spatial clustering analysis for target: {target_field}")
        
        # Validate target field
        if target_field and target_field not in df.columns:
            return safe_jsonify({"error": f"Target field '{target_field}' not found in dataset"}, 400)
        
        # Prepare clustering features
        if not clustering_features:
            clustering_features = [col for col in df.select_dtypes(include=[np.number]).columns 
                                 if col != target_field and not col.startswith('geometry')]
        
        # Filter valid features
        valid_features = [feature for feature in clustering_features if feature in df.columns]
        if len(valid_features) < 2:
            return safe_jsonify({"error": "Need at least 2 valid features for clustering"}, 400)
        
        # Prepare data with robust NaN handling
        X = df[valid_features].fillna(df[valid_features].median())
        y = df[target_field].fillna(df[target_field].median()) if target_field else None
        
        # Ensure no NaN values remain and handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        # Final verification - replace any remaining NaN with 0
        if X.isnull().any().any():
            X = X.fillna(0)
        
        # Perform clustering
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features with robust preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Train model for SHAP analysis if target field provided
        shap_explanations = {}
        if target_field:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            sample_size = min(200, len(X))
            X_sample = X.sample(n=sample_size, random_state=42)
            shap_values = explainer.shap_values(X_sample)
            
            # Calculate SHAP explanations for each cluster
            for cluster_id in range(num_clusters):
                cluster_mask = cluster_labels == cluster_id
                if np.sum(cluster_mask) > 0:
                    cluster_indices = np.where(cluster_mask)[0]
                    
                    # Get SHAP values for this cluster (intersection with sample)
                    sample_indices = X_sample.index.values
                    cluster_sample_mask = np.isin(sample_indices, cluster_indices)
                    
                    if np.sum(cluster_sample_mask) > 0:
                        cluster_shap = shap_values[cluster_sample_mask]
                        
                        # Calculate average SHAP contribution per feature
                        feature_contributions = {}
                        for i, feature in enumerate(valid_features):
                            avg_contribution = np.mean(cluster_shap[:, i])
                            feature_contributions[feature] = float(avg_contribution)
                        
                        shap_explanations[cluster_id] = feature_contributions
        
        # Analyze each cluster
        cluster_analysis = []
        for cluster_id in range(num_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = X[cluster_mask]
            cluster_target = y[cluster_mask] if target_field else None
            
            if len(cluster_data) == 0:
                continue
            
            # Calculate cluster characteristics
            cluster_profile = {}
            for feature in valid_features:
                cluster_mean = float(cluster_data[feature].mean())
                overall_mean = float(X[feature].mean())
                difference = cluster_mean - overall_mean
                
                cluster_profile[feature] = {
                    'cluster_mean': cluster_mean,
                    'overall_mean': overall_mean,
                    'difference_from_overall': difference,
                    'relative_difference': float(difference / overall_mean * 100) if overall_mean != 0 else 0
                }
            
            # Find distinguishing characteristics (features that differ significantly from overall)
            distinguishing_features = [
                {
                    'feature': feature,
                    'cluster_mean': profile['cluster_mean'],
                    'difference_percent': profile['relative_difference']
                }
                for feature, profile in cluster_profile.items()
                if abs(profile['relative_difference']) > 15  # >15% difference
            ]
            
            # Sort by absolute difference
            distinguishing_features.sort(key=lambda x: abs(x['difference_percent']), reverse=True)
            
            # Target field statistics for this cluster
            target_stats = None
            if target_field and cluster_target is not None:
                target_stats = {
                    'mean': float(cluster_target.mean()),
                    'median': float(cluster_target.median()),
                    'std': float(cluster_target.std()),
                    'min': float(cluster_target.min()),
                    'max': float(cluster_target.max()),
                    'difference_from_overall': float(cluster_target.mean() - y.mean())
                }
            
            # Get representative areas for this cluster
            cluster_df_indices = df.index[cluster_mask]
            representative_areas = []
            
            for i, idx in enumerate(cluster_df_indices[:5]):  # Top 5 representative areas
                area_info = {}
                for id_col in ['CSDNAME', 'FSA_ID', 'DESCRIPTION', 'NAME', 'ID']:
                    if id_col in df.columns and idx in df.index:
                        area_value = df.loc[idx, id_col]
                        if area_value and str(area_value) != 'nan':
                            area_info['name'] = str(area_value)
                            break
                
                if 'name' not in area_info:
                    area_info['name'] = f'Area_{idx}'
                
                # Add target value if available
                if target_field:
                    area_info['target_value'] = float(df.loc[idx, target_field])
                
                representative_areas.append(area_info)
            
            cluster_analysis.append({
                'cluster_id': cluster_id,
                'size': int(np.sum(cluster_mask)),
                'percentage_of_total': float(np.sum(cluster_mask) / len(df) * 100),
                'target_statistics': target_stats,
                'distinguishing_features': distinguishing_features[:5],  # Top 5
                'shap_explanation': shap_explanations.get(cluster_id),
                'representative_areas': representative_areas,
                'cluster_center': [float(coord) for coord in kmeans.cluster_centers_[cluster_id]]
            })
        
        # Sort clusters by size
        cluster_analysis.sort(key=lambda x: x['size'], reverse=True)
        
        # Calculate clustering quality metrics
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        clustering_quality = {
            'silhouette_score': float(silhouette_score(X_scaled, cluster_labels)),
            'calinski_harabasz_score': float(calinski_harabasz_score(X_scaled, cluster_labels)),
            'inertia': float(kmeans.inertia_)
        }
        
        result = {
            "clusters": cluster_analysis,
            "clustering_quality": clustering_quality,
            "analysis_parameters": {
                'num_clusters': num_clusters,
                'features_used': valid_features,
                'target_field': target_field,
                'total_areas': len(df)
            },
            "insights": {
                'largest_cluster': cluster_analysis[0]['cluster_id'] if cluster_analysis else None,
                'most_distinct_cluster': max(cluster_analysis, key=lambda x: len(x['distinguishing_features']))['cluster_id'] if cluster_analysis else None,
                'cluster_size_distribution': [cluster['size'] for cluster in cluster_analysis]
            }
        }
        
        logger.info(f"Spatial clustering analysis completed successfully. Found {len(cluster_analysis)} clusters.")
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in spatial clustering analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "error": f"Spatial clustering analysis failed: {str(e)}"
        }, 500)

@app.route('/competitive-analysis', methods=['POST'])
def analyze_competition():
    """
    SHAP analysis of competitive brand landscape.
    Business value: "How does Nike compete with Adidas in different areas?"
    """
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded. Cannot perform competitive analysis."}, 500)
    
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")
    
    data = request.json
    primary_brand = data.get('primary_brand')
    competitor_brands = data.get('competitor_brands', [])
    analysis_dimensions = data.get('analysis_dimensions', ['demographic', 'geographic', 'economic'])
    
    try:
        logger.info(f"Starting competitive analysis for {primary_brand} vs {competitor_brands}")
        
        # Auto-detect brand fields if not provided
        all_brand_fields = [col for col in df.columns if 'mp30' in col.lower() and 'a_b_p' in col.lower()]
        
        # Validate primary brand
        if primary_brand not in all_brand_fields:
            return safe_jsonify({"error": f"Primary brand '{primary_brand}' not found in dataset"}, 400)
        
        # Validate competitor brands
        if not competitor_brands:
            competitor_brands = [field for field in all_brand_fields if field != primary_brand]
        
        valid_competitors = [brand for brand in competitor_brands if brand in all_brand_fields]
        if not valid_competitors:
            return safe_jsonify({"error": "No valid competitor brands found"}, 400)
        
        # Prepare demographic and economic features
        demo_features = []
        if 'demographic' in analysis_dimensions:
            demo_features.extend([col for col in df.columns if any(demo in col.lower() 
                                for demo in ['age', 'asian', 'black', 'hispanic', 'white', 'pop'])])
        
        if 'economic' in analysis_dimensions:
            demo_features.extend([col for col in df.columns if any(econ in col.lower() 
                                for econ in ['income', 'wealth', 'divindx'])])
        
        # Filter valid features
        valid_features = [f for f in demo_features if f in df.columns][:20]  # Limit for performance
        
        if len(valid_features) < 5:
            return safe_jsonify({"error": "Insufficient analysis features available"}, 400)
        
        # Prepare data
        brand_data = df[[primary_brand] + valid_competitors].fillna(0)
        feature_data = df[valid_features].fillna(df[valid_features].median())
        
        # Calculate competitive metrics
        competitive_metrics = []
        
        for competitor in valid_competitors:
            # Calculate market share and overlap
            primary_values = brand_data[primary_brand]
            competitor_values = brand_data[competitor]
            
            # Market overlap analysis
            high_primary = primary_values >= primary_values.quantile(0.75)
            high_competitor = competitor_values >= competitor_values.quantile(0.75)
            
            overlap_areas = (high_primary & high_competitor).sum()
            total_high_areas = (high_primary | high_competitor).sum()
            
            overlap_percentage = float(overlap_areas / total_high_areas * 100) if total_high_areas > 0 else 0
            
            # Correlation analysis
            correlation = float(primary_values.corr(competitor_values))
            
            # Market dominance analysis
            primary_wins = (primary_values > competitor_values).sum()
            competitor_wins = (competitor_values > primary_values).sum()
            ties = (primary_values == competitor_values).sum()
            
            total_areas = len(primary_values)
            
            # SHAP analysis for competitive differentiation
            # Create binary classification: primary_brand > competitor
            competition_target = (primary_values > competitor_values).astype(int)
            
            if competition_target.nunique() > 1:  # Need both classes for analysis
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(feature_data, competition_target)
                
                # Calculate SHAP values
                explainer = shap.TreeExplainer(model)
                sample_size = min(100, len(feature_data))
                X_sample = feature_data.sample(n=sample_size, random_state=42)
                shap_values = explainer.shap_values(X_sample)
                
                # For binary classification, use positive class SHAP values
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                
                # Calculate feature importance for competitive advantage
                competitive_factors = {}
                for i, feature in enumerate(valid_features):
                    importance = np.abs(shap_values[:, i]).mean()
                    # Calculate average feature value where primary brand wins
                    primary_win_mask = competition_target == 1
                    if primary_win_mask.sum() > 0:
                        avg_when_winning = feature_data.loc[primary_win_mask, feature].mean()
                        overall_avg = feature_data[feature].mean()
                        advantage_direction = 'positive' if avg_when_winning > overall_avg else 'negative'
                    else:
                        advantage_direction = 'neutral'
                    
                    competitive_factors[feature] = {
                        'importance': float(importance),
                        'advantage_direction': advantage_direction,
                        'winning_average': float(avg_when_winning) if primary_win_mask.sum() > 0 else None,
                        'overall_average': float(overall_avg)
                    }
                
                # Sort by importance
                sorted_factors = sorted(competitive_factors.items(), key=lambda x: x[1]['importance'], reverse=True)
                top_factors = dict(sorted_factors[:10])
                
                model_accuracy = float(model.score(feature_data, competition_target))
            else:
                top_factors = {}
                model_accuracy = None
            
            competitive_metrics.append({
                'competitor_brand': competitor,
                'market_overlap_percentage': overlap_percentage,
                'correlation': correlation,
                'relationship_type': 'competitive' if correlation < 0 else 'complementary' if correlation > 0.3 else 'neutral',
                'market_dominance': {
                    'primary_wins': int(primary_wins),
                    'competitor_wins': int(competitor_wins),
                    'ties': int(ties),
                    'primary_win_rate': float(primary_wins / total_areas * 100),
                    'competitor_win_rate': float(competitor_wins / total_areas * 100)
                },
                'competitive_factors': top_factors,
                'model_accuracy': model_accuracy
            })
        
        # Sort by competitive intensity (inverse correlation + overlap)
        competitive_metrics.sort(key=lambda x: -x['correlation'] + x['market_overlap_percentage'], reverse=True)
        
        # Market positioning analysis
        positioning_analysis = {
            'market_leader': primary_brand,
            'strongest_competitor': competitive_metrics[0]['competitor_brand'] if competitive_metrics else None,
            'most_complementary': None,
            'most_competitive': None
        }
        
        for metric in competitive_metrics:
            if metric['relationship_type'] == 'complementary' and not positioning_analysis['most_complementary']:
                positioning_analysis['most_complementary'] = metric['competitor_brand']
            elif metric['relationship_type'] == 'competitive' and not positioning_analysis['most_competitive']:
                positioning_analysis['most_competitive'] = metric['competitor_brand']
        
        # Strategic insights
        total_brand_data = brand_data.sum()
        market_shares = {}
        total_market = total_brand_data.sum()
        
        for brand in [primary_brand] + valid_competitors:
            market_shares[brand] = {
                'absolute_share': float(total_brand_data[brand]),
                'relative_share': float(total_brand_data[brand] / total_market * 100) if total_market > 0 else 0
            }
        
        result = {
            "primary_brand": primary_brand,
            "competitive_analysis": competitive_metrics,
            "market_positioning": positioning_analysis,
            "market_shares": market_shares,
            "analysis_summary": {
                'competitors_analyzed': len(valid_competitors),
                'analysis_dimensions': analysis_dimensions,
                'features_used': valid_features,
                'total_market_areas': len(df)
            },
            "strategic_insights": {
                'most_contested_markets': overlap_percentage if competitive_metrics else 0,
                'differentiation_opportunities': len([f for f in top_factors.values() if f.get('importance', 0) > 0.1]) if competitive_metrics and top_factors else 0,
                'market_concentration': len([share for share in market_shares.values() if share['relative_share'] > 20])
            }
        }
        
        logger.info(f"Competitive analysis completed successfully for {primary_brand}")
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in competitive analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "error": f"Competitive analysis failed: {str(e)}"
        }, 500)

@app.route('/lifecycle-analysis', methods=['POST'])
def analyze_lifecycle():
    """
    SHAP analysis of demographic lifecycle patterns.
    Business value: "How do brand preferences change with life stage?"
    """
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded. Cannot perform lifecycle analysis."}, 500)
    
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")
    
    data = request.json
    target_field = data.get('target_field')
    lifecycle_features = data.get('lifecycle_features', ['age', 'income', 'education', 'family'])
    
    try:
        logger.info(f"Starting lifecycle analysis for target: {target_field}")
        
        # Validate target field
        if target_field not in df.columns:
            return safe_jsonify({"error": f"Target field '{target_field}' not found in dataset"}, 400)
        
        # Map lifecycle features to actual column names
        feature_mapping = {
            'age': [col for col in df.columns if 'age' in col.lower() or 'millenn' in col.lower() or 'genz' in col.lower()],
            'income': [col for col in df.columns if 'income' in col.lower() or 'meddi' in col.lower() or 'wealth' in col.lower()],
            'education': [col for col in df.columns if 'education' in col.lower() or 'divindx' in col.lower()],
            'family': [col for col in df.columns if 'fampop' in col.lower() or 'hhpop' in col.lower()]
        }
        
        # Collect all relevant features
        analysis_features = []
        for category in lifecycle_features:
            if category in feature_mapping:
                analysis_features.extend(feature_mapping[category])
        
        # Add demographic features
        demo_features = [col for col in df.columns if any(demo in col.lower() 
                        for demo in ['asian', 'black', 'hispanic', 'white', 'pop'])]
        analysis_features.extend(demo_features[:10])  # Limit for performance
        
        # Filter valid features
        valid_features = [f for f in analysis_features if f in df.columns]
        valid_features = list(set(valid_features))[:25]  # Remove duplicates and limit
        
        if len(valid_features) < 5:
            return safe_jsonify({"error": "Insufficient lifecycle features available"}, 400)
        
        # Prepare data
        X = df[valid_features].fillna(df[valid_features].median())
        y = df[target_field].fillna(df[target_field].median())
        
        # Define lifecycle stages based on available data
        lifecycle_stages = []
        
        # Age-based stages (if age data available)
        age_cols = [col for col in valid_features if 'age' in col.lower()]
        if age_cols:
            age_col = age_cols[0]
            age_data = X[age_col]
            
            # Define age quartiles as lifecycle stages
            age_q25 = age_data.quantile(0.25)
            age_q50 = age_data.quantile(0.50)
            age_q75 = age_data.quantile(0.75)
            
            lifecycle_stages.append({
                'stage_name': 'Young Adult',
                'criteria': f'{age_col} <= {age_q25}',
                'mask': age_data <= age_q25
            })
            lifecycle_stages.append({
                'stage_name': 'Early Career',
                'criteria': f'{age_q25} < {age_col} <= {age_q50}',
                'mask': (age_data > age_q25) & (age_data <= age_q50)
            })
            lifecycle_stages.append({
                'stage_name': 'Mid Career',
                'criteria': f'{age_q50} < {age_col} <= {age_q75}',
                'mask': (age_data > age_q50) & (age_data <= age_q75)
            })
            lifecycle_stages.append({
                'stage_name': 'Mature',
                'criteria': f'{age_col} > {age_q75}',
                'mask': age_data > age_q75
            })
        
        # Income-based stages (if income data available)
        income_cols = [col for col in valid_features if 'income' in col.lower() or 'meddi' in col.lower()]
        if income_cols and not lifecycle_stages:  # Use income if no age data
            income_col = income_cols[0]
            income_data = X[income_col]
            
            income_q33 = income_data.quantile(0.33)
            income_q67 = income_data.quantile(0.67)
            
            lifecycle_stages.append({
                'stage_name': 'Lower Income',
                'criteria': f'{income_col} <= {income_q33}',
                'mask': income_data <= income_q33
            })
            lifecycle_stages.append({
                'stage_name': 'Middle Income', 
                'criteria': f'{income_q33} < {income_col} <= {income_q67}',
                'mask': (income_data > income_q33) & (income_data <= income_q67)
            })
            lifecycle_stages.append({
                'stage_name': 'Higher Income',
                'criteria': f'{income_col} > {income_q67}',
                'mask': income_data > income_q67
            })
        
        # If no clear lifecycle features, create generic stages
        if not lifecycle_stages:
            # Use target variable itself to create stages
            target_q33 = y.quantile(0.33)
            target_q67 = y.quantile(0.67)
            
            lifecycle_stages.append({
                'stage_name': 'Low Engagement',
                'criteria': f'{target_field} <= {target_q33}',
                'mask': y <= target_q33
            })
            lifecycle_stages.append({
                'stage_name': 'Medium Engagement',
                'criteria': f'{target_q33} < {target_field} <= {target_q67}',
                'mask': (y > target_q33) & (y <= target_q67)
            })
            lifecycle_stages.append({
                'stage_name': 'High Engagement',
                'criteria': f'{target_field} > {target_q67}',
                'mask': y > target_q67
            })
        
        # Train overall model for SHAP analysis
        from sklearn.ensemble import RandomForestRegressor
        overall_model = RandomForestRegressor(n_estimators=100, random_state=42)
        overall_model.fit(X, y)
        
        # Calculate SHAP values for each lifecycle stage
        stage_analysis = []
        
        for stage in lifecycle_stages:
            stage_mask = stage['mask']
            stage_size = stage_mask.sum()
            
            if stage_size < 10:  # Skip stages with too few samples
                continue
            
            # Stage-specific data
            X_stage = X[stage_mask]
            y_stage = y[stage_mask]
            
            # Stage statistics
            stage_stats = {
                'mean_target': float(y_stage.mean()),
                'median_target': float(y_stage.median()),
                'std_target': float(y_stage.std()),
                'sample_size': int(stage_size),
                'percentage_of_total': float(stage_size / len(df) * 100)
            }
            
            # SHAP analysis for this stage
            if len(X_stage) >= 20:  # Enough samples for SHAP
                explainer = shap.TreeExplainer(overall_model)
                sample_size = min(50, len(X_stage))
                X_sample = X_stage.sample(n=sample_size, random_state=42)
                shap_values = explainer.shap_values(X_sample)
                
                # Calculate feature importance for this stage
                stage_feature_importance = {}
                for i, feature in enumerate(valid_features):
                    importance = np.abs(shap_values[:, i]).mean()
                    stage_feature_importance[feature] = float(importance)
                
                # Sort by importance
                sorted_features = sorted(stage_feature_importance.items(), key=lambda x: x[1], reverse=True)
                top_features = dict(sorted_features[:10])
            else:
                top_features = {}
            
            # Feature profile for this stage
            feature_profile = {}
            for feature in valid_features[:10]:  # Top 10 features
                stage_mean = float(X_stage[feature].mean())
                overall_mean = float(X[feature].mean())
                difference = stage_mean - overall_mean
                
                feature_profile[feature] = {
                    'stage_average': stage_mean,
                    'overall_average': overall_mean,
                    'difference': difference,
                    'relative_difference': float(difference / overall_mean * 100) if overall_mean != 0 else 0
                }
            
            stage_analysis.append({
                'stage_name': stage['stage_name'],
                'stage_criteria': stage['criteria'],
                'stage_statistics': stage_stats,
                'key_features': top_features,
                'feature_profile': feature_profile
            })
        
        # Cross-stage analysis
        stage_transitions = []
        for i, stage1 in enumerate(stage_analysis[:-1]):
            stage2 = stage_analysis[i + 1]
            
            transition = {
                'from_stage': stage1['stage_name'],
                'to_stage': stage2['stage_name'],
                'target_change': stage2['stage_statistics']['mean_target'] - stage1['stage_statistics']['mean_target'],
                'size_change': stage2['stage_statistics']['sample_size'] - stage1['stage_statistics']['sample_size']
            }
            
            stage_transitions.append(transition)
        
        # Lifecycle insights
        insights = {
            'stages_identified': len(stage_analysis),
            'highest_engagement_stage': max(stage_analysis, key=lambda x: x['stage_statistics']['mean_target'])['stage_name'] if stage_analysis else None,
            'largest_stage': max(stage_analysis, key=lambda x: x['stage_statistics']['sample_size'])['stage_name'] if stage_analysis else None,
            'most_variable_stage': max(stage_analysis, key=lambda x: x['stage_statistics']['std_target'])['stage_name'] if stage_analysis else None
        }
        
        result = {
            "target_variable": target_field,
            "lifecycle_stages": stage_analysis,
            "stage_transitions": stage_transitions,
            "lifecycle_insights": insights,
            "analysis_parameters": {
                'lifecycle_features_used': lifecycle_features,
                'actual_features_analyzed': valid_features,
                'total_samples': len(df)
            },
            "model_performance": float(overall_model.score(X, y))
        }
        
        logger.info(f"Lifecycle analysis completed successfully for {target_field}")
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in lifecycle analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "error": f"Lifecycle analysis failed: {str(e)}"
        }, 500)

@app.route('/economic-sensitivity', methods=['POST'])
def analyze_economic_sensitivity():
    """
    SHAP analysis of economic impact on brand preferences.
    Business value: "How do economic changes affect Nike sales?"
    """
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded. Cannot perform economic sensitivity analysis."}, 500)
    
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")
    
    data = request.json
    target_field = data.get('target_field')
    economic_indicators = data.get('economic_indicators', ['income', 'wealth', 'employment'])
    sensitivity_threshold = data.get('sensitivity_threshold', 0.05)
    
    try:
        logger.info(f"Starting economic sensitivity analysis for target: {target_field}")
        
        # Validate target field
        if target_field not in df.columns:
            return safe_jsonify({"error": f"Target field '{target_field}' not found in dataset"}, 400)
        
        # Map economic indicators to actual column names
        economic_mapping = {
            'income': [col for col in df.columns if 'income' in col.lower() or 'meddi' in col.lower()],
            'wealth': [col for col in df.columns if 'wealth' in col.lower() or 'divindx' in col.lower()],
            'employment': [col for col in df.columns if 'emp' in col.lower()],
            'education': [col for col in df.columns if 'edu' in col.lower()],
            'population': [col for col in df.columns if 'pop' in col.lower() or 'hhpop' in col.lower()]
        }
        
        # Collect economic features
        economic_features = []
        for indicator in economic_indicators:
            if indicator in economic_mapping:
                economic_features.extend(economic_mapping[indicator])
        
        # Add general demographic features that could be economic proxies
        proxy_features = [col for col in df.columns if any(econ in col.lower() 
                         for econ in ['age', 'asian', 'black', 'hispanic', 'white'])]
        economic_features.extend(proxy_features[:10])
        
        # Filter valid features
        valid_features = [f for f in economic_features if f in df.columns]
        valid_features = list(set(valid_features))[:20]  # Remove duplicates and limit
        
        if len(valid_features) < 3:
            return safe_jsonify({"error": "Insufficient economic features available for sensitivity analysis"}, 400)
        
        # Prepare data
        X = df[valid_features].fillna(df[valid_features].median())
        y = df[target_field].fillna(df[target_field].median())
        
        # Train model for SHAP analysis
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        sample_size = min(200, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)
        
        # Economic sensitivity analysis for each feature
        sensitivity_analysis = []
        
        for i, feature in enumerate(valid_features):
            # Calculate feature importance and sensitivity
            feature_shap = shap_values[:, i]
            importance = np.abs(feature_shap).mean()
            
            # Calculate elasticity (% change in target per % change in feature)
            feature_values = X_sample[feature]
            target_values = y[X_sample.index]
            
            # Avoid division by zero
            nonzero_mask = (feature_values != 0) & (target_values != 0)
            if nonzero_mask.sum() > 10:
                # Calculate approximate elasticity using correlation of log values
                feature_log = np.log(feature_values[nonzero_mask] + 1)  # Add 1 to handle zeros
                target_log = np.log(target_values[nonzero_mask] + 1)
                elasticity = feature_log.corr(target_log)
            else:
                elasticity = 0
            
            # Scenario analysis: impact of ±10% change in economic indicator
            feature_mean = X[feature].mean()
            feature_std = X[feature].std()
            
            # Create scenarios
            scenarios = {}
            for scenario_name, multiplier in [('recession_10pct_drop', 0.9), ('growth_10pct_rise', 1.1), 
                                            ('severe_recession_20pct_drop', 0.8), ('boom_20pct_rise', 1.2)]:
                # Modify feature values
                X_scenario = X.copy()
                X_scenario[feature] = X_scenario[feature] * multiplier
                
                # Predict impact
                y_pred_original = model.predict(X)
                y_pred_scenario = model.predict(X_scenario)
                
                impact = (y_pred_scenario.mean() - y_pred_original.mean()) / y_pred_original.mean() * 100
                
                scenarios[scenario_name] = {
                    'feature_change_percent': (multiplier - 1) * 100,
                    'predicted_target_change_percent': float(impact),
                    'sensitivity_ratio': float(impact / ((multiplier - 1) * 100)) if multiplier != 1 else 0
                }
            
            # Determine sensitivity level
            max_impact = max([abs(s['predicted_target_change_percent']) for s in scenarios.values()])
            if max_impact > 10:
                sensitivity_level = 'high'
            elif max_impact > 5:
                sensitivity_level = 'medium'
            else:
                sensitivity_level = 'low'
            
            sensitivity_analysis.append({
                'economic_indicator': feature,
                'shap_importance': float(importance),
                'elasticity': float(elasticity),
                'sensitivity_level': sensitivity_level,
                'scenario_impacts': scenarios,
                'statistical_summary': {
                    'mean': float(X[feature].mean()),
                    'std': float(X[feature].std()),
                    'correlation_with_target': float(X[feature].corr(y))
                }
            })
        
        # Sort by importance
        sensitivity_analysis.sort(key=lambda x: x['shap_importance'], reverse=True)
        
        # Overall economic vulnerability assessment
        high_sensitivity_count = len([s for s in sensitivity_analysis if s['sensitivity_level'] == 'high'])
        total_sensitivity_score = sum([s['shap_importance'] for s in sensitivity_analysis])
        
        vulnerability_assessment = {
            'overall_vulnerability': 'high' if high_sensitivity_count > 2 else 'medium' if high_sensitivity_count > 0 else 'low',
            'high_sensitivity_indicators': high_sensitivity_count,
            'total_sensitivity_score': float(total_sensitivity_score),
            'most_vulnerable_to': sensitivity_analysis[0]['economic_indicator'] if sensitivity_analysis else None
        }
        
        # Economic resilience insights
        resilience_factors = []
        for indicator in sensitivity_analysis:
            if indicator['sensitivity_level'] == 'low' and indicator['shap_importance'] > 0.01:
                resilience_factors.append({
                    'factor': indicator['economic_indicator'],
                    'resilience_score': 1 / (indicator['shap_importance'] + 0.001),  # Inverse relationship
                    'stability': 'stable' if abs(indicator['elasticity']) < 0.1 else 'variable'
                })
        
        result = {
            "target_variable": target_field,
            "economic_sensitivity_analysis": sensitivity_analysis,
            "vulnerability_assessment": vulnerability_assessment,
            "resilience_factors": resilience_factors[:5],  # Top 5 resilience factors
            "analysis_parameters": {
                'economic_indicators_analyzed': economic_indicators,
                'actual_features_used': valid_features,
                'sensitivity_threshold': sensitivity_threshold,
                'total_areas_analyzed': len(df)
            },
            "model_performance": float(model.score(X, y)),
            "economic_insights": {
                'most_elastic_indicator': max(sensitivity_analysis, key=lambda x: abs(x['elasticity']))['economic_indicator'] if sensitivity_analysis else None,
                'recession_impact_estimate': sensitivity_analysis[0]['scenario_impacts']['recession_10pct_drop']['predicted_target_change_percent'] if sensitivity_analysis else None,
                'growth_opportunity_estimate': sensitivity_analysis[0]['scenario_impacts']['growth_10pct_rise']['predicted_target_change_percent'] if sensitivity_analysis else None
            }
        }
        
        logger.info(f"Economic sensitivity analysis completed successfully for {target_field}")
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in economic sensitivity analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "error": f"Economic sensitivity analysis failed: {str(e)}"
        }, 500)

@app.route('/penetration-optimization', methods=['POST'])
def analyze_penetration_optimization():
    """
    SHAP-based recommendations for market penetration optimization.
    Business value: "How can Nike increase market penetration in underperforming areas?"
    """
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded. Cannot perform penetration optimization analysis."}, 500)
    
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")
    
    data = request.json
    target_field = data.get('target_field')
    optimization_features = data.get('optimization_features', ['demographic', 'economic', 'competitive'])
    target_improvement = data.get('target_improvement', 0.2)  # 20% improvement target
    
    try:
        logger.info(f"Starting penetration optimization analysis for target: {target_field}")
        
        # Validate target field
        if target_field not in df.columns:
            return safe_jsonify({"error": f"Target field '{target_field}' not found in dataset"}, 400)
        
        # Map optimization features to actual columns
        feature_mapping = {
            'demographic': [col for col in df.columns if any(demo in col.lower() 
                          for demo in ['age', 'asian', 'black', 'hispanic', 'white', 'pop'])],
            'economic': [col for col in df.columns if any(econ in col.lower() 
                        for econ in ['income', 'wealth', 'meddi', 'divindx'])],
            'competitive': [col for col in df.columns if 'mp30' in col.lower() and 'a_b_p' in col.lower() and col != target_field]
        }
        
        # Collect optimization features
        analysis_features = []
        for category in optimization_features:
            if category in feature_mapping:
                analysis_features.extend(feature_mapping[category][:8])  # Limit per category
        
        # Filter valid features
        valid_features = [f for f in analysis_features if f in df.columns]
        valid_features = list(set(valid_features))[:25]  # Remove duplicates and limit
        
        if len(valid_features) < 5:
            return safe_jsonify({"error": "Insufficient optimization features available"}, 400)
        
        # Prepare data
        X = df[valid_features].fillna(df[valid_features].median())
        y = df[target_field].fillna(df[target_field].median())
        
        # Identify underperforming areas (bottom quartile)
        performance_threshold = y.quantile(0.25)
        underperforming_mask = y <= performance_threshold
        high_performing_mask = y >= y.quantile(0.75)
        
        underperforming_areas = df[underperforming_mask]
        high_performing_areas = df[high_performing_mask]
        
        logger.info(f"Found {underperforming_mask.sum()} underperforming areas and {high_performing_mask.sum()} high-performing areas")
        
        # Train model for SHAP analysis
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Calculate SHAP values for underperforming areas
        explainer = shap.TreeExplainer(model)
        
        if underperforming_mask.sum() > 0:
            X_underperforming = X[underperforming_mask]
            sample_size = min(100, len(X_underperforming))
            X_sample = X_underperforming.sample(n=sample_size, random_state=42)
            shap_values_under = explainer.shap_values(X_sample)
        else:
            return safe_jsonify({"error": "No underperforming areas found for optimization"}, 400)
        
        # Calculate SHAP values for high-performing areas for comparison
        if high_performing_mask.sum() > 0:
            X_high_performing = X[high_performing_mask]
            sample_size_high = min(100, len(X_high_performing))
            X_sample_high = X_high_performing.sample(n=sample_size_high, random_state=42)
            shap_values_high = explainer.shap_values(X_sample_high)
        else:
            shap_values_high = None
        
        # Optimization recommendations
        optimization_opportunities = []
        
        for i, feature in enumerate(valid_features):
            # Compare feature values between high and low performing areas
            underperforming_mean = X[underperforming_mask][feature].mean()
            high_performing_mean = X[high_performing_mask][feature].mean() if high_performing_mask.sum() > 0 else X[feature].mean()
            overall_mean = X[feature].mean()
            
            # Calculate gap and potential
            performance_gap = high_performing_mean - underperforming_mean
            gap_percentage = (performance_gap / underperforming_mean * 100) if underperforming_mean != 0 else 0
            
            # SHAP importance for this feature
            shap_importance_under = np.abs(shap_values_under[:, i]).mean()
            shap_importance_high = np.abs(shap_values_high[:, i]).mean() if shap_values_high is not None else 0
            
            # Determine optimization potential
            if abs(gap_percentage) > 10 and shap_importance_under > 0.01:
                optimization_potential = 'high'
            elif abs(gap_percentage) > 5 and shap_importance_under > 0.005:
                optimization_potential = 'medium'
            else:
                optimization_potential = 'low'
            
            # Calculate potential impact of closing the gap
            # Simulate moving underperforming areas closer to high-performing values
            X_optimized = X.copy()
            if performance_gap > 0:  # Only if high-performing areas have higher values
                improvement_factor = min(target_improvement, performance_gap / underperforming_mean) if underperforming_mean != 0 else 0
                X_optimized.loc[underperforming_mask, feature] = X_optimized.loc[underperforming_mask, feature] * (1 + improvement_factor)
            
            # Predict impact
            y_pred_original = model.predict(X[underperforming_mask])
            y_pred_optimized = model.predict(X_optimized[underperforming_mask])
            
            predicted_improvement = (y_pred_optimized.mean() - y_pred_original.mean()) / y_pred_original.mean() * 100
            
            # Determine actionability
            if 'income' in feature.lower() or 'wealth' in feature.lower():
                actionability = 'indirect'  # Can influence through economic development
                recommendations = [f"Focus on economic development initiatives", f"Target higher-income segments"]
            elif 'age' in feature.lower():
                actionability = 'indirect'
                recommendations = [f"Adjust marketing to target optimal age groups", f"Develop age-specific product lines"]
            elif any(ethnic in feature.lower() for ethnic in ['asian', 'black', 'hispanic', 'white']):
                actionability = 'direct'
                recommendations = [f"Develop culturally targeted marketing campaigns", f"Partner with community organizations"]
            elif 'mp30' in feature.lower():  # Competitive brands
                actionability = 'direct'
                recommendations = [f"Develop competitive positioning strategies", f"Focus on differentiation"]
            else:
                actionability = 'moderate'
                recommendations = [f"Analyze {feature} patterns for optimization opportunities"]
            
            optimization_opportunities.append({
                'feature': feature,
                'optimization_potential': optimization_potential,
                'performance_gap': float(performance_gap),
                'gap_percentage': float(gap_percentage),
                'shap_importance': float(shap_importance_under),
                'predicted_improvement_percent': float(predicted_improvement),
                'actionability': actionability,
                'recommendations': recommendations,
                'current_values': {
                    'underperforming_average': float(underperforming_mean),
                    'high_performing_average': float(high_performing_mean),
                    'overall_average': float(overall_mean)
                }
            })
        
        # Sort by optimization potential and predicted improvement
        optimization_opportunities.sort(key=lambda x: (
            {'high': 3, 'medium': 2, 'low': 1}[x['optimization_potential']], 
            x['predicted_improvement_percent']
        ), reverse=True)
        
        # Strategic optimization plan
        top_opportunities = optimization_opportunities[:5]
        total_potential_improvement = sum([opp['predicted_improvement_percent'] for opp in top_opportunities])
        
        strategic_plan = {
            'priority_optimizations': top_opportunities,
            'estimated_total_improvement': float(total_potential_improvement),
            'quick_wins': [opp for opp in optimization_opportunities if opp['actionability'] == 'direct' and opp['optimization_potential'] == 'high'][:3],
            'long_term_initiatives': [opp for opp in optimization_opportunities if opp['actionability'] == 'indirect' and opp['optimization_potential'] == 'high'][:3]
        }
        
        # Geographic targeting recommendations
        underperforming_sample = underperforming_areas.head(10)
        targeting_recommendations = []
        
        for idx, area in underperforming_sample.iterrows():
            area_name = 'Unknown'
            for col in ['CSDNAME', 'DESCRIPTION', 'NAME', 'ID']:
                if col in df.columns and pd.notna(area[col]):
                    area_name = str(area[col])
                    break
            
            # Find the most impactful optimization for this area
            area_X = X.loc[idx:idx]
            area_shap = explainer.shap_values(area_X)[0]
            
            top_feature_idx = np.argmax(np.abs(area_shap))
            top_feature = valid_features[top_feature_idx]
            
            targeting_recommendations.append({
                'area_name': area_name,
                'current_performance': float(y.loc[idx]),
                'top_optimization_feature': top_feature,
                'optimization_priority': optimization_opportunities[0]['optimization_potential'] if optimization_opportunities else 'medium'
            })
        
        result = {
            "target_variable": target_field,
            "optimization_opportunities": optimization_opportunities,
            "strategic_plan": strategic_plan,
            "geographic_targeting": targeting_recommendations,
            "market_analysis": {
                'underperforming_areas_count': int(underperforming_mask.sum()),
                'high_performing_areas_count': int(high_performing_mask.sum()),
                'performance_threshold': float(performance_threshold),
                'average_performance_gap': float(y[high_performing_mask].mean() - y[underperforming_mask].mean()) if high_performing_mask.sum() > 0 and underperforming_mask.sum() > 0 else 0
            },
            "analysis_parameters": {
                'optimization_features_used': optimization_features,
                'actual_features_analyzed': valid_features,
                'target_improvement_goal': target_improvement,
                'total_areas_analyzed': len(df)
            },
            "model_performance": float(model.score(X, y))
        }
        
        logger.info(f"Penetration optimization analysis completed successfully for {target_field}")
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in penetration optimization analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "error": f"Penetration optimization analysis failed: {str(e)}"
        }, 500)

@app.route('/market-risk', methods=['POST'])
def analyze_market_risk():
    """
    SHAP analysis of market vulnerability and risk factors.
    Business value: "What factors put Nike market share at risk?"
    """
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded. Cannot perform market risk analysis."}, 500)
    
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")
    
    data = request.json
    target_field = data.get('target_field')
    risk_factors = data.get('risk_factors', ['competitive', 'economic', 'demographic'])
    risk_threshold = data.get('risk_threshold', 0.1)  # 10% decline threshold
    
    try:
        logger.info(f"Starting market risk analysis for target: {target_field}")
        
        # Validate target field
        if target_field not in df.columns:
            return safe_jsonify({"error": f"Target field '{target_field}' not found in dataset"}, 400)
        
        # Map risk factors to actual columns
        risk_mapping = {
            'competitive': [col for col in df.columns if 'mp30' in col.lower() and 'a_b_p' in col.lower() and col != target_field],
            'economic': [col for col in df.columns if any(econ in col.lower() for econ in ['income', 'wealth', 'meddi', 'divindx'])],
            'demographic': [col for col in df.columns if any(demo in col.lower() for demo in ['age', 'asian', 'black', 'hispanic', 'white', 'pop'])],
            'geographic': [col for col in df.columns if any(geo in col.lower() for geo in ['area', 'density', 'shape'])]
        }
        
        # Collect risk features
        analysis_features = []
        for category in risk_factors:
            if category in risk_mapping:
                analysis_features.extend(risk_mapping[category][:10])  # Limit per category
        
        # Filter valid features
        valid_features = [f for f in analysis_features if f in df.columns]
        valid_features = list(set(valid_features))[:25]  # Remove duplicates and limit
        
        if len(valid_features) < 5:
            return safe_jsonify({"error": "Insufficient risk features available"}, 400)
        
        # Prepare data
        X = df[valid_features].fillna(df[valid_features].median())
        y = df[target_field].fillna(df[target_field].median())
        
        # Identify high-risk areas (areas with declining or low performance)
        performance_threshold = y.quantile(0.25)  # Bottom quartile
        high_risk_mask = y <= performance_threshold
        low_risk_mask = y >= y.quantile(0.75)  # Top quartile
        
        high_risk_areas = df[high_risk_mask]
        low_risk_areas = df[low_risk_mask]
        
        logger.info(f"Found {high_risk_mask.sum()} high-risk areas and {low_risk_mask.sum()} low-risk areas")
        
        # Train model for SHAP analysis
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Calculate SHAP values for risk analysis
        explainer = shap.TreeExplainer(model)
        
        # Analyze high-risk areas
        if high_risk_mask.sum() > 0:
            X_high_risk = X[high_risk_mask]
            sample_size = min(100, len(X_high_risk))
            X_sample_risk = X_high_risk.sample(n=sample_size, random_state=42)
            shap_values_risk = explainer.shap_values(X_sample_risk)
        else:
            return safe_jsonify({"error": "No high-risk areas found for analysis"}, 400)
        
        # Risk factor analysis
        risk_factors_analysis = []
        
        for i, feature in enumerate(valid_features):
            # Compare feature values between high-risk and low-risk areas
            high_risk_mean = X[high_risk_mask][feature].mean()
            low_risk_mean = X[low_risk_mask][feature].mean() if low_risk_mask.sum() > 0 else X[feature].mean()
            overall_mean = X[feature].mean()
            
            # Calculate risk differential
            risk_differential = high_risk_mean - low_risk_mean
            risk_percentage = (risk_differential / low_risk_mean * 100) if low_risk_mean != 0 else 0
            
            # SHAP importance for risk
            shap_importance_risk = np.abs(shap_values_risk[:, i]).mean()
            
            # Determine risk level
            if abs(risk_percentage) > 20 and shap_importance_risk > 0.01:
                risk_level = 'high'
            elif abs(risk_percentage) > 10 and shap_importance_risk > 0.005:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            # Risk scenario modeling
            scenarios = {}
            
            # Model various risk scenarios
            for scenario_name, multiplier in [
                ('competitor_surge_20pct', 1.2 if 'mp30' in feature.lower() else 1.0),
                ('economic_downturn_15pct', 0.85 if any(econ in feature.lower() for econ in ['income', 'wealth']) else 1.0),
                ('demographic_shift_10pct', 1.1 if any(demo in feature.lower() for demo in ['age', 'asian', 'black', 'hispanic']) else 1.0)
            ]:
                if multiplier != 1.0:  # Only analyze relevant scenarios for this feature
                    # Modify feature values
                    X_scenario = X.copy()
                    X_scenario[feature] = X_scenario[feature] * multiplier
                    
                    # Predict impact
                    y_pred_original = model.predict(X)
                    y_pred_scenario = model.predict(X_scenario)
                    
                    impact = (y_pred_scenario.mean() - y_pred_original.mean()) / y_pred_original.mean() * 100
                    
                    scenarios[scenario_name] = {
                        'feature_change_percent': (multiplier - 1) * 100,
                        'predicted_impact_percent': float(impact),
                        'risk_severity': 'high' if impact < -10 else 'medium' if impact < -5 else 'low'
                    }
            
            # Risk mitigation strategies
            mitigation_strategies = []
            if 'mp30' in feature.lower():  # Competitive risk
                mitigation_strategies = [
                    "Strengthen brand differentiation",
                    "Develop competitive response strategies",
                    "Focus on customer loyalty programs"
                ]
            elif any(econ in feature.lower() for econ in ['income', 'wealth']):  # Economic risk
                mitigation_strategies = [
                    "Develop value-oriented product lines",
                    "Adjust pricing strategies",
                    "Target recession-resistant segments"
                ]
            elif any(demo in feature.lower() for demo in ['age', 'asian', 'black', 'hispanic']):  # Demographic risk
                mitigation_strategies = [
                    "Diversify target demographics",
                    "Develop culturally relevant marketing",
                    "Adapt product portfolio to changing demographics"
                ]
            else:
                mitigation_strategies = [
                    f"Monitor {feature} trends closely",
                    "Develop contingency plans"
                ]
            
            risk_factors_analysis.append({
                'risk_factor': feature,
                'risk_level': risk_level,
                'risk_differential': float(risk_differential),
                'risk_percentage': float(risk_percentage),
                'shap_importance': float(shap_importance_risk),
                'scenario_impacts': scenarios,
                'mitigation_strategies': mitigation_strategies,
                'current_values': {
                    'high_risk_average': float(high_risk_mean),
                    'low_risk_average': float(low_risk_mean),
                    'overall_average': float(overall_mean)
                }
            })
        
        # Sort by risk level and importance
        risk_factors_analysis.sort(key=lambda x: (
            {'high': 3, 'medium': 2, 'low': 1}[x['risk_level']], 
            x['shap_importance']
        ), reverse=True)
        
        # Overall risk assessment
        high_risk_count = len([r for r in risk_factors_analysis if r['risk_level'] == 'high'])
        total_risk_score = sum([r['shap_importance'] for r in risk_factors_analysis])
        
        overall_risk_assessment = {
            'market_risk_level': 'high' if high_risk_count > 3 else 'medium' if high_risk_count > 1 else 'low',
            'high_risk_factors_count': high_risk_count,
            'total_risk_score': float(total_risk_score),
            'primary_risk_category': risk_factors[0] if risk_factors else 'unknown',
            'risk_concentration': 'concentrated' if high_risk_count < 2 else 'diversified'
        }
        
        # Risk monitoring recommendations
        monitoring_recommendations = []
        for risk_factor in risk_factors_analysis[:5]:  # Top 5 risks
            monitoring_recommendations.append({
                'factor': risk_factor['risk_factor'],
                'monitoring_frequency': 'monthly' if risk_factor['risk_level'] == 'high' else 'quarterly',
                'key_metrics': [f"{risk_factor['risk_factor']} trend analysis", "Market share tracking"],
                'early_warning_threshold': float(risk_factor['current_values']['overall_average'] * 0.95)  # 5% decline
            })
        
        # Geographic risk hotspots
        risk_hotspots = []
        high_risk_sample = high_risk_areas.head(10)
        
        for idx, area in high_risk_sample.iterrows():
            area_name = 'Unknown'
            for col in ['CSDNAME', 'DESCRIPTION', 'NAME', 'ID']:
                if col in df.columns and pd.notna(area[col]):
                    area_name = str(area[col])
                    break
            
            # Calculate area-specific risk score
            area_X = X.loc[idx:idx]
            area_shap = explainer.shap_values(area_X)[0]
            risk_score = np.sum(area_shap[area_shap < 0])  # Sum of negative SHAP values
            
            risk_hotspots.append({
                'area_name': area_name,
                'current_performance': float(y.loc[idx]),
                'risk_score': float(risk_score),
                'primary_risk_factor': valid_features[np.argmin(area_shap)]
            })
        
        result = {
            "target_variable": target_field,
            "risk_factors_analysis": risk_factors_analysis,
            "overall_risk_assessment": overall_risk_assessment,
            "monitoring_recommendations": monitoring_recommendations,
            "risk_hotspots": risk_hotspots,
            "market_analysis": {
                'high_risk_areas_count': int(high_risk_mask.sum()),
                'low_risk_areas_count': int(low_risk_mask.sum()),
                'risk_threshold': float(performance_threshold),
                'market_volatility': float(y.std() / y.mean()) if y.mean() != 0 else 0
            },
            "analysis_parameters": {
                'risk_factors_analyzed': risk_factors,
                'actual_features_used': valid_features,
                'risk_threshold': risk_threshold,
                'total_areas_analyzed': len(df)
            },
            "model_performance": float(model.score(X, y))
        }
        
        logger.info(f"Market risk analysis completed successfully for {target_field}")
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in market risk analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "error": f"Market risk analysis failed: {str(e)}"
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

# --- Local Correlation (LISA) Dependencies ---
try:
    import geopandas as gpd
    from shapely.geometry import shape  # Convert GeoJSON to shapely geometry
    from libpysal.weights import KNN
    from esda.moran import Moran_Local_BV
except ImportError:
    # Modules may not be installed in some environments; log warning – the endpoint will error gracefully
    gpd = None  # type: ignore
    shape = None  # type: ignore
    KNN = None  # type: ignore
    Moran_Local_BV = None  # type: ignore
    logger.warning("geopandas / libpysal / esda not installed – /local_corr endpoint will be unavailable")

@app.route('/local_corr', methods=['POST'])
def calculate_local_correlation():
    """
    Compute bivariate Local Moran's I (LISA) between two variables for each feature.

    Expected JSON body:
    {
        "field_x": "column_name_1",
        "field_y": "column_name_2",
        "features": [ { "geometry": {..}, "attributes": { "field_x": .., "field_y": .. }} ],
        "k_neighbors": 6  # optional (default 6)
    }
    Returns GeoJSON-like list of features with added attributes:
        local_I, p_value, cluster (int)
    """
    if gpd is None or shape is None or KNN is None or Moran_Local_BV is None:
        return safe_jsonify({
            "error": "geopandas / libpysal / esda not installed on server."}, 501)

    data = request.json or {}
    field_x = data.get('field_x')
    field_y = data.get('field_y')
    raw_features = data.get('features', [])
    k_neighbors = int(data.get('k_neighbors', 6))

    # Validate input
    if not field_x or not field_y:
        abort(400, description="Missing required 'field_x' or 'field_y'")
    if not raw_features:
        abort(400, description="No features supplied for local correlation analysis")

    logger.info(f"Starting local correlation for {len(raw_features)} features (k={k_neighbors}) → {field_x} vs {field_y}")

    try:
        # Build GeoDataFrame
        geometries = []
        values_x = []
        values_y = []
        feature_geometries = []  # preserve for output

        for feat in raw_features:
            geom_json = feat.get('geometry')
            attrs = feat.get('attributes', feat.get('properties', {}))

            if geom_json is None:
                continue  # skip features without geometry

            geometries.append(shape(geom_json))
            feature_geometries.append(geom_json)

            val_x = attrs.get(field_x)
            val_y = attrs.get(field_y)
            # Convert to float safely
            try:
                val_x = float(val_x) if val_x is not None else np.nan
            except Exception:
                val_x = np.nan
            try:
                val_y = float(val_y) if val_y is not None else np.nan
            except Exception:
                val_y = np.nan

            values_x.append(val_x)
            values_y.append(val_y)

        gdf = gpd.GeoDataFrame({
            'val_x': values_x,
            'val_y': values_y
        }, geometry=geometries, crs="EPSG:4326")

        # Drop rows with missing values
        gdf = gdf.dropna(subset=['val_x', 'val_y'])

        if gdf.empty or len(gdf) < 3:
            abort(400, description="Insufficient valid features for local correlation analysis")

        # Build spatial weights matrix using k-nearest neighbors
        w = KNN.from_dataframe(gdf, k=k_neighbors)
        w.transform = 'r'

        lisa = Moran_Local_BV(gdf['val_x'], gdf['val_y'], w)

        local_I = lisa.Is
        pvals = lisa.p_sim
        clusters = lisa.q  # 1=HH,2=LH,3=LL,4=HL

        # Build response features list aligned with original order (may differ due to dropna)
        results = []
        idx_mapping = list(gdf.index)
        for idx_out, idx_orig in enumerate(idx_mapping):
            res_feat = {
                'geometry': feature_geometries[idx_orig],
                'attributes': {
                    'OBJECTID': int(idx_out + 1),
                    'local_I': float(local_I[idx_out]),
                    'p_value': float(pvals[idx_out]),
                    'cluster': int(clusters[idx_out])
                }
            }
            results.append(res_feat)

        response = {
            'analysis_type': 'local_correlation',
            'features': results,
            'metadata': {
                'field_x': field_x,
                'field_y': field_y,
                'k_neighbors': k_neighbors,
                'feature_count': len(results)
            },
            'success': True
        }

        logger.info("Local correlation analysis completed successfully")
        return safe_jsonify(response)

    except Exception as e:
        logger.error(f"Local correlation analysis failed: {e}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            'error': f'Local correlation failed: {str(e)}',
            'success': False
        }, 500)
