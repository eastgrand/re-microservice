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
from enhanced_analysis_worker import enhanced_analysis_worker

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
        return jsonify({"error": "Dataset not loaded. Schema unavailable."}), 500

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
    
    return jsonify({
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
        return jsonify(analysis_results)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        logger.error(traceback.format_exc())
        abort(500, description=f"Analysis failed: {str(e)}")

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
