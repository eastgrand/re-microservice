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

# Import the master schema and data processing function
from map_nesto_data import MASTER_SCHEMA, TARGET_VARIABLE, load_and_preprocess_data
# Import the real analysis function
from query_aware_analysis import enhanced_query_aware_analysis

# --- Redis/RQ Imports for Async Jobs ---
import redis

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


# --- Redis/RQ Setup ---
# Use the REDIS_URL from environment variables, which is standard for Render deployments


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

        # Use the query-aware analysis function from the original implementation
        result = enhanced_query_aware_analysis(
            df=df,
            model=model,
            feature_names=feature_names,
            analysis_type=analysis_request.get('analysis_type', 'correlation'),
            target_variable=analysis_request.get('target_variable'),
            demographic_filters=analysis_request.get('demographic_filters', {}),
            matched_fields=analysis_request.get('matched_fields', []),
            min_applications=analysis_request.get('min_applications', 1),
            query=analysis_request.get('query', ''),
            top_n=analysis_request.get('top_n', 0)
        )
        
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
    Exposes the master data schema to the frontend.
    This is the single source of truth for all field names, aliases, and descriptions.
    """
    if not MASTER_SCHEMA:
        return jsonify({"error": "Schema is not available or failed to load"}), 500

    # Prepare a list of all canonical names for convenience
    known_fields = [details['canonical_name'] for _, details in MASTER_SCHEMA.items()]

    return jsonify({
        "fields": MASTER_SCHEMA,
        "known_fields": known_fields
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint. It validates the request against available data columns
    and returns a mock analysis result.
    """
    if df is None:
        abort(500, description="Dataset not loaded. Cannot perform analysis.")

    # --- Request Validation ---
    if not request.json:
        abort(400, description="Invalid request: Missing JSON body.")

    data = request.json
    target_variable = data.get('target_variable')
    matched_fields = data.get('matched_fields', [])
    
    # Validate that the requested fields are present in our dataset
    all_requested_fields = [target_variable] + matched_fields
    unknown_fields = [field for field in all_requested_fields if field and field not in AVAILABLE_COLUMNS]

    if unknown_fields:
        error_msg = f"Unknown metric fields requested: {', '.join(unknown_fields)}. Please use fields available in the schema."
        abort(400, description=error_msg)

    logger.info(f"Received analysis request with target '{target_variable}' and fields {matched_fields}")

    # --- Mock Analysis ---
    # In a real scenario, this is where the SHAP analysis would run.
    # We will return a structured mock response.
    analysis_results = {
        'success': True,
        'message': 'Analysis complete',
        'target_variable': target_variable,
        'matched_fields': matched_fields,
        'summary': {
            'base_value': 0.5,
            'feature_impacts': {field: np.random.rand() * 0.1 for field in matched_fields}
        },
        'results': [
            {'geo_id': 'A0A', 'shap_values': {field: np.random.rand() for field in matched_fields}, 'prediction': np.random.rand()}
            for i in range(10) # Return top 10 mock results
        ]
    }
    
    # Return the analysis results
    return jsonify(analysis_results)

# === Error Handlers ===

@app.errorhandler(400)
def bad_request(error):
    logger.error(f"Bad Request: {error.description}")
    response = jsonify({'error': 'Bad Request', 'message': error.description})
    response.status_code = 400
    return response

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal Server Error: {traceback.format_exc()}")
    response = jsonify({'error': 'Internal Server Error', 'message': 'An unexpected error occurred.'})
    response.status_code = 500
    return response

if __name__ == '__main__':
    # Note: Running in debug mode is not recommended for production
    app.run(debug=True, port=5001)
