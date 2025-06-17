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
        """Map field names from frontend to actual column names in the dataset."""
        if not field_name:
            return field_name
            
        # Direct field mappings for common cases
        field_mappings = {
            'condo_ownership_pct': '2024 Condominium Status - In Condo (%)',
            'condominium_pct': '2024 Condominium Status - In Condo (%)',
            'single_detached_house_pct': '2024 Structure Type Single-Detached House (%)',
            'visible_minority_population_pct': '2024 Visible Minority Total Population (%)',
            'median_income': '2024 Household Average Income (Current Year $)',
            'disposable_income': '2024 Household Discretionary Aggregate Income',
            'mortgage_approvals': 'SUM_FUNDED',
            'conversion_rate': 'CONVERSION_RATE'
        }
        
        if field_name in field_mappings:
            logger.info(f"Direct mapping found: '{field_name}' -> '{field_mappings[field_name]}'")
            return field_mappings[field_name]
        
        # If no direct mapping, try the existing logic
        field_lower = field_name.lower()
        
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
