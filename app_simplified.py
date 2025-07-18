#!/usr/bin/env python3
"""
SHAP Microservice - Simplified Redis-Free Version
Synchronous dataset generation without Redis dependencies
"""

import os
import sys
import logging
import traceback
import gc
import pickle
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import json
import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional

# Import custom modules
from map_nesto_data import MASTER_SCHEMA, TARGET_VARIABLE, load_and_preprocess_data, initialize_schema
from enhanced_analysis_worker import enhanced_analysis_worker
from field_utils import resolve_field_name
from memory_utils import (
    get_memory_usage, force_garbage_collection,
    batch_shap_calculation, memory_safe_shap_wrapper,
    get_endpoint_config
)

# --- FLASK APP SETUP ---
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-API-Key"],
        "supports_credentials": True
    }
})

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("shap-microservice")

# --- ENVIRONMENT VARIABLES ---
load_dotenv()
API_KEY = os.getenv('API_KEY')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
REQUIRE_AUTH = API_KEY is not None

# --- PATHS ---
MODEL_PATH = "models/xgboost_model.pkl"
FEATURE_NAMES_PATH = "models/feature_names.txt"
TRAINING_DATASET_PATH = "data/nesto_merge_0.csv"
JOINED_DATASET_PATH = "data/joined_data.csv"

# --- GLOBAL DATA STORAGE ---
model = None
feature_names = []
training_data = None
joined_data = None
schema_initialized = False

# --- CUSTOM JSON HANDLER ---
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
        elif pd.isna(obj):
            return None
        return obj
    
    safe_data = convert_nan(data)
    response = jsonify(safe_data)
    response.status_code = status_code
    return response

# --- AUTHENTICATION DECORATOR ---
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'OPTIONS':
            return f(*args, **kwargs)
        if not REQUIRE_AUTH:
            return f(*args, **kwargs)
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != API_KEY:
            return safe_jsonify({"success": False, "error": "Unauthorized"}, 401)
        return f(*args, **kwargs)
    return decorated_function

# --- INITIALIZATION FUNCTIONS ---
def load_model_and_data():
    """Load model, features, and datasets on startup"""
    global model, feature_names, training_data, joined_data, schema_initialized
    
    try:
        logger.info("Loading model and data...")
        
        # Load XGBoost model
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            logger.info("✅ XGBoost model loaded successfully")
        else:
            logger.warning(f"⚠️ Model file not found: {MODEL_PATH}")
        
        # Load feature names
        if os.path.exists(FEATURE_NAMES_PATH):
            with open(FEATURE_NAMES_PATH, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            logger.info(f"✅ Feature names loaded: {len(feature_names)} features")
        else:
            logger.warning(f"⚠️ Feature names file not found: {FEATURE_NAMES_PATH}")
        
        # Load training dataset
        if os.path.exists(TRAINING_DATASET_PATH):
            training_data = pd.read_csv(TRAINING_DATASET_PATH)
            logger.info(f"✅ Training data loaded: {training_data.shape}")
        else:
            logger.warning(f"⚠️ Training dataset not found: {TRAINING_DATASET_PATH}")
        
        # Load joined dataset
        if os.path.exists(JOINED_DATASET_PATH):
            joined_data = pd.read_csv(JOINED_DATASET_PATH)
            logger.info(f"✅ Joined data loaded: {joined_data.shape}")
        else:
            logger.warning(f"⚠️ Joined dataset not found: {JOINED_DATASET_PATH}")
        
        # Initialize schema
        initialize_schema()
        schema_initialized = True
        logger.info("✅ Schema initialized")
        
        # Force garbage collection
        force_garbage_collection()
        logger.info(f"✅ Initialization complete. Memory usage: {get_memory_usage():.1f}MB")
        
    except Exception as e:
        logger.error(f"❌ Error during initialization: {str(e)}")
        logger.error(traceback.format_exc())

# --- CORE ANALYSIS FUNCTION ---
def run_analysis_sync(data: Dict[str, Any]) -> Dict[str, Any]:
    """Run SHAP analysis synchronously without Redis"""
    start_time = time.time()
    
    try:
        logger.info("Starting synchronous SHAP analysis...")
        
        # Use enhanced analysis worker for the heavy lifting
        result = enhanced_analysis_worker(data)
        
        # Add timing information
        result['processing_time'] = time.time() - start_time
        result['timestamp'] = datetime.now().isoformat()
        result['success'] = True
        
        # Memory cleanup
        force_garbage_collection()
        
        logger.info(f"✅ Analysis completed in {result['processing_time']:.2f}s")
        return result
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Analysis failed: {error_msg}")
        logger.error(traceback.format_exc())
        
        return {
            'success': False,
            'error': error_msg,
            'error_type': type(e).__name__,
            'processing_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }

# --- API ENDPOINTS ---

@app.route('/')
def health_check():
    """Basic health check endpoint"""
    return safe_jsonify({
        "message": "SHAP Microservice (Redis-Free) is running",
        "status": "healthy",
        "version": "2.0.0-simplified",
        "redis_enabled": False,
        "model_loaded": model is not None,
        "schema_initialized": schema_initialized,
        "memory_usage_mb": get_memory_usage()
    })

@app.route('/ping', methods=['GET'])
def ping():
    """Simple ping endpoint"""
    return safe_jsonify({"pong": True, "timestamp": datetime.now().isoformat()})

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return safe_jsonify({
        "status": "healthy",
        "redis_enabled": False,
        "model_status": "loaded" if model else "not_loaded",
        "feature_count": len(feature_names),
        "training_data_rows": len(training_data) if training_data is not None else 0,
        "joined_data_rows": len(joined_data) if joined_data is not None else 0,
        "schema_initialized": schema_initialized,
        "memory_usage_mb": get_memory_usage(),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
@require_api_key
def analyze():
    """
    Main analysis endpoint - now synchronous
    Returns results directly instead of job ID
    """
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return safe_jsonify({
                "success": False,
                "error": "No JSON data provided"
            }, 400)
        
        logger.info(f"Received analysis request: {data.keys()}")
        
        # Check if model is loaded
        if model is None:
            return safe_jsonify({
                "success": False,
                "error": "Model not loaded"
            }, 503)
        
        # Run analysis synchronously
        result = run_analysis_sync(data)
        
        # Return results directly
        if result.get('success', False):
            return safe_jsonify(result)
        else:
            return safe_jsonify(result, 500)
        
    except Exception as e:
        logger.error(f"❌ Analyze endpoint error: {str(e)}")
        logger.error(traceback.format_exc())
        return safe_jsonify({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat()
        }, 500)

# --- SIMPLIFIED ENDPOINT IMPLEMENTATIONS ---

@app.route('/outlier-detection', methods=['POST'])
@require_api_key
def outlier_detection():
    """Outlier detection with SHAP explanations"""
    try:
        data = request.get_json()
        if not data:
            return safe_jsonify({"success": False, "error": "No data provided"}, 400)
        
        # Set analysis type
        data['analysis_type'] = 'outlier-detection'
        
        # Run analysis
        result = run_analysis_sync(data)
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Outlier detection error: {str(e)}")
        return safe_jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, 500)

@app.route('/feature-interactions', methods=['POST'])
@require_api_key
def feature_interactions():
    """Feature interaction analysis"""
    try:
        data = request.get_json()
        if not data:
            return safe_jsonify({"success": False, "error": "No data provided"}, 400)
        
        # Set analysis type
        data['analysis_type'] = 'feature-interactions'
        
        # Run analysis
        result = run_analysis_sync(data)
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Feature interactions error: {str(e)}")
        return safe_jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, 500)

@app.route('/spatial-clusters', methods=['POST'])
@require_api_key
def spatial_clusters():
    """Spatial clustering analysis"""
    try:
        data = request.get_json()
        if not data:
            return safe_jsonify({"success": False, "error": "No data provided"}, 400)
        
        # Set analysis type
        data['analysis_type'] = 'spatial-clusters'
        
        # Run analysis
        result = run_analysis_sync(data)
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Spatial clusters error: {str(e)}")
        return safe_jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, 500)

@app.route('/competitive-analysis', methods=['POST'])
@require_api_key
def competitive_analysis():
    """Competitive brand analysis"""
    try:
        data = request.get_json()
        if not data:
            return safe_jsonify({"success": False, "error": "No data provided"}, 400)
        
        # Set analysis type
        data['analysis_type'] = 'competitive-analysis'
        
        # Run analysis
        result = run_analysis_sync(data)
        return safe_jsonify(result)
        
    except Exception as e:
        logger.error(f"Competitive analysis error: {str(e)}")
        return safe_jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }, 500)

# --- ERROR HANDLERS ---
@app.errorhandler(404)
def not_found(error):
    return safe_jsonify({
        "success": False,
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /",
            "GET /ping", 
            "GET /health",
            "POST /analyze",
            "POST /outlier-detection",
            "POST /feature-interactions", 
            "POST /spatial-clusters",
            "POST /competitive-analysis"
        ]
    }, 404)

@app.errorhandler(500)
def internal_error(error):
    return safe_jsonify({
        "success": False,
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }, 500)

# --- STARTUP ---
if __name__ == '__main__':
    # Load model and data
    load_model_and_data()
    
    # Run the app
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"🚀 Starting SHAP Microservice (Redis-Free) on port {port}")
    app.run(host='0.0.0.0', port=port, debug=DEBUG) 