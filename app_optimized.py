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
import time
from datetime import datetime
import threading
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import asyncio
from threading import Semaphore

# Import optimization modules
from model_cache import model_cache, get_or_train_model
from async_job_manager import job_manager, JobStatus

# Import the master schema and data processing function
from map_nesto_data import MASTER_SCHEMA, TARGET_VARIABLE, load_and_preprocess_data, initialize_schema
# Import the real analysis function
from enhanced_analysis_worker import enhanced_analysis_worker

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
CORS(app)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Rate Limiting and Circuit Breaker ---
class RateLimiter:
    def __init__(self, max_requests=10, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
        self.lock = threading.Lock()
    
    def allow_request(self, client_id):
        now = time.time()
        with self.lock:
            if client_id not in self.requests:
                self.requests[client_id] = []
            
            # Remove old requests outside the window
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if now - req_time < self.window_seconds
            ]
            
            # Check if under limit
            if len(self.requests[client_id]) >= self.max_requests:
                return False
            
            # Add current request
            self.requests[client_id].append(now)
            return True

# Global rate limiter and semaphore for concurrency control
rate_limiter = RateLimiter(max_requests=20, window_seconds=60)
analysis_semaphore = Semaphore(4)  # Max 4 concurrent analysis jobs

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
            logger.warning(f"Model file not found at {MODEL_PATH}. Using cached models only.")

        if os.path.exists(FEATURE_NAMES_PATH):
            with open(FEATURE_NAMES_PATH, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            logger.info(f"Successfully loaded {len(feature_names)} feature names from {FEATURE_NAMES_PATH}")
        else:
            logger.warning(f"Feature names file not found at {FEATURE_NAMES_PATH}.")

    except Exception as e:
        logger.error(f"An error occurred during model loading: {e}")
        logger.error(traceback.format_exc())

# Load data and model on startup
try:
    logger.info("--- Loading optimized SHAP microservice ---")
    load_and_preprocess_data()
    
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, 'data', 'cleaned_data.csv')
    df = pd.read_csv(data_path)
    AVAILABLE_COLUMNS = set(df.columns)
    logger.info(f"Successfully loaded data with {len(df)} rows and {len(AVAILABLE_COLUMNS)} columns")
    
    initialize_schema(df)
    logger.info("Dynamic schema initialized")
    
    # Clear expired cache entries
    model_cache.clear_expired_cache()
    
except Exception as e:
    logger.error(f"FATAL: Could not load data on startup: {e}")
    df = None
    AVAILABLE_COLUMNS = set()

# Load the model on startup
load_model_and_features()

# --- Middleware ---
@app.before_request
def before_request():
    """Apply rate limiting and security checks."""
    # Skip rate limiting for health checks
    if request.endpoint in ['health', 'ping']:
        return
    
    # Get client identifier (IP address or API key)
    client_id = request.headers.get('X-API-Key', request.remote_addr)
    
    # Apply rate limiting
    if not rate_limiter.allow_request(client_id):
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': 'Too many requests. Please wait before trying again.',
            'retry_after': 60
        }), 429

# --- Optimized Analysis Worker ---
def optimized_analysis_worker(analysis_request, progress_callback=None):
    """
    Optimized analysis worker that uses cached models and reduced payloads.
    """
    try:
        if progress_callback:
            progress_callback(10, "Initializing analysis...")
        
        if df is None:
            raise ValueError("Dataset not loaded")
        
        # Extract analysis parameters
        target_variable = analysis_request.get('target_variable', TARGET_VARIABLE)
        analysis_type = analysis_request.get('analysis_type', 'correlation')
        matched_fields = analysis_request.get('matched_fields', [])
        
        if progress_callback:
            progress_callback(30, "Preparing data...")
        
        # Prepare features (limit to reduce memory usage)
        max_features = 50  # Limit features for performance
        features = matched_fields[:max_features] if matched_fields else []
        
        if not features:
            features = [col for col in df.columns if col != target_variable][:max_features]
        
        # Prepare training data
        X = df[features].fillna(0)
        y = df[target_variable].fillna(0)
        
        if progress_callback:
            progress_callback(50, "Loading or training model...")
        
        # Get or train model using cache
        model, explainer = get_or_train_model(target_variable, features, analysis_type, X, y)
        
        if progress_callback:
            progress_callback(70, "Calculating SHAP values...")
        
        # Calculate SHAP values for a sample (reduce computation)
        sample_size = min(1000, len(X))  # Limit sample size
        X_sample = X.sample(n=sample_size, random_state=42)
        
        shap_values = explainer(X_sample)
        
        if progress_callback:
            progress_callback(90, "Preparing results...")
        
        # Prepare optimized results (reduced payload)
        feature_importance = []
        for i, feature in enumerate(features):
            importance = np.abs(shap_values.values[:, i]).mean()
            feature_importance.append({
                'feature': feature,
                'importance': float(importance)
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        # Return only essential data
        result = {
            'success': True,
            'analysis_type': analysis_type,
            'target_variable': target_variable,
            'sample_size': sample_size,
            'feature_importance': feature_importance[:10],  # Top 10 only
            'model_info': {
                'model_type': 'XGBoost',
                'features_used': len(features),
                'cached': True  # Indicate this was from cache
            },
            'processing_time': time.time(),
            'summary': f"Analysis completed for {target_variable} using {len(features)} features"
        }
        
        if progress_callback:
            progress_callback(100, "Analysis complete")
        
        return result
        
    except Exception as e:
        logger.error(f"Optimized analysis worker failed: {e}")
        raise

# === API Endpoints ===

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    cache_stats = model_cache.get_cache_stats()
    queue_stats = job_manager.get_queue_stats()
    
    return safe_jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'cache_stats': cache_stats,
        'queue_stats': queue_stats,
        'data_loaded': df is not None,
        'model_loaded': model is not None
    })

@app.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint."""
    return safe_jsonify({'status': 'OK', 'timestamp': datetime.now().isoformat()})

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Optimized analysis endpoint with asynchronous processing.
    Returns job ID immediately for long-running analyses.
    """
    if df is None:
        return safe_jsonify({'error': 'Dataset not loaded'}, 500)
    
    if not request.json:
        return safe_jsonify({'error': 'Invalid request: Missing JSON body'}, 400)
    
    try:
        # Acquire semaphore for concurrency control
        if not analysis_semaphore.acquire(blocking=False):
            return safe_jsonify({
                'error': 'Service busy',
                'message': 'Too many concurrent analyses. Please try again shortly.',
                'retry_after': 30
            }, 503)
        
        # Submit job to async manager
        job_id = job_manager.submit_job('analysis', request.json)
        
        # Release semaphore (job manager will handle concurrency)
        analysis_semaphore.release()
        
        return safe_jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Analysis job submitted successfully'
        }, 202)
        
    except Exception as e:
        analysis_semaphore.release()
        logger.error(f"Analysis submission failed: {e}")
        return safe_jsonify({
            'error': 'Failed to submit analysis',
            'message': str(e)
        }, 500)

@app.route('/job_status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get the status of an analysis job."""
    status = job_manager.get_job_status(job_id)
    
    if not status:
        return safe_jsonify({'error': 'Job not found'}, 404)
    
    return safe_jsonify(status)

@app.route('/job_result/<job_id>', methods=['GET'])
def get_job_result(job_id):
    """Get the result of a completed analysis job."""
    result = job_manager.get_job_result(job_id)
    
    if result is None:
        status = job_manager.get_job_status(job_id)
        if not status:
            return safe_jsonify({'error': 'Job not found'}, 404)
        elif status['status'] != 'completed':
            return safe_jsonify({'error': 'Job not completed', 'status': status['status']}, 400)
        else:
            return safe_jsonify({'error': 'No result available'}, 404)
    
    return safe_jsonify(result)

@app.route('/cancel_job/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """Cancel a running analysis job."""
    success = job_manager.cancel_job(job_id)
    
    if not success:
        return safe_jsonify({'error': 'Job not found or cannot be cancelled'}, 404)
    
    return safe_jsonify({'message': 'Job cancelled successfully'})

@app.route('/cache_stats', methods=['GET'])
def get_cache_stats():
    """Get model cache statistics."""
    stats = model_cache.get_cache_stats()
    return safe_jsonify(stats)

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the model cache."""
    model_cache.clear_cache()
    return safe_jsonify({'message': 'Cache cleared successfully'})

@app.route('/api/v1/schema', methods=['GET'])
def get_schema():
    """Get the data schema."""
    if df is None:
        return safe_jsonify({"error": "Dataset not loaded"}, 500)
    
    # Generate lightweight schema
    schema = {}
    for col in df.columns:
        schema[col] = {
            'name': col,
            'type': 'numeric' if df[col].dtype in ['int64', 'float64'] else 'string',
            'description': f'Data field: {col}'
        }
    
    return safe_jsonify({
        'fields': schema,
        'field_count': len(schema)
    })

# --- Error Handlers ---
@app.errorhandler(429)
def rate_limit_handler(e):
    return safe_jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please wait before trying again.'
    }, 429)

@app.errorhandler(503)
def service_unavailable(e):
    return safe_jsonify({
        'error': 'Service temporarily unavailable',
        'message': 'The service is currently busy. Please try again shortly.'
    }, 503)

@app.errorhandler(500)
def internal_server_error(e):
    return safe_jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred. Please try again.'
    }, 500)

# --- Cleanup and Shutdown ---
def cleanup_on_shutdown():
    """Cleanup resources on shutdown."""
    logger.info("Shutting down optimized SHAP microservice...")
    job_manager.shutdown()
    model_cache.clear_expired_cache()

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    cleanup_on_shutdown()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- Background Tasks ---
def background_maintenance():
    """Background task for maintenance."""
    while True:
        try:
            # Clean up old jobs every hour
            job_manager.cleanup_old_jobs()
            model_cache.clear_expired_cache()
            time.sleep(3600)  # 1 hour
        except Exception as e:
            logger.error(f"Background maintenance error: {e}")
            time.sleep(300)  # 5 minutes on error

# Start background maintenance thread
maintenance_thread = threading.Thread(target=background_maintenance, daemon=True)
maintenance_thread.start()

if __name__ == '__main__':
    # Register the optimized analysis worker
    job_manager.register_handler('analysis', optimized_analysis_worker)
    
    # Run with multiple workers for better concurrency
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True) 