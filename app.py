import os
import sys
import logging
import traceback
import gc
import pickle
import platform
import shutil
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from functools import wraps
from dotenv import load_dotenv
from data_versioning import DataVersionTracker
import uuid
from collections import defaultdict
import time
import psutil
import threading
import signal
import xgboost as xgb
import json
import math
from typing import List, Dict, Any, Optional
from worker_process_fix import apply_all_worker_patches

# Attempt to import query-aware analysis functions
try:
    from query_aware_analysis import enhanced_query_aware_analysis, generate_intent_aware_summary
    QUERY_AWARE_AVAILABLE = True
    print("✅ Successfully imported query-aware analysis module.")
except ImportError as e:
    QUERY_AWARE_AVAILABLE = False
    print(f"⚠️ Query-aware analysis module not found or failed to import: {e}. Running in standard mode.")
    # Define dummy functions if the import fails to prevent runtime errors
    def enhanced_query_aware_analysis(*args, **kwargs):
        return {"summary": "Query-aware analysis is not available.", "results": []}
    def generate_intent_aware_summary(*args, **kwargs):
        return "Query-aware analysis is not available."

# Import field mappings and target variable
from map_nesto_data import FIELD_MAPPINGS, TARGET_VARIABLE

# --- FLASK APP SETUP (must come after imports) ---
app = Flask(__name__)

# Initialize CORS with proper configuration
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-API-KEY"],
        "supports_credentials": True
    }
})

@app.route('/')
def hello_world():
    return jsonify({
        "message": "SHAP Microservice is running", 
        "status": "healthy",
        "query_aware_available": QUERY_AWARE_AVAILABLE
    }), 200

print(f"🚀 Flask app created successfully - Query-aware: {QUERY_AWARE_AVAILABLE}")

# --- /analyze GET handler (added for friendly error) ---
@app.route('/analyze', methods=['GET'])
def analyze_get():
    """GET handler for /analyze to provide a helpful message instead of 404."""
    return jsonify({
        "message": "Use POST to /analyze to submit a job for SHAP/XGBoost analysis. This endpoint only accepts POST for analysis jobs.",
        "usage": {
            "POST /analyze": "Submit a JSON body with analysis_type, target_variable, and demographic_filters to start an async job.",
            "GET /job_status/<job_id>": "Poll for job status/results."
        },
        "status": 405
    }), 405


# --- REDIS/RQ IMPORTS FOR ASYNC JOBS ---
import redis
from rq import Queue, get_current_job

# --- PATHS FOR MODEL, FEATURE NAMES, AND DATASET ---
MODEL_PATH = "models/xgboost_model.pkl"  # Update if your model file is named differently
FEATURE_NAMES_PATH = "models/feature_names.txt"  # Update if your feature names file is named differently
TRAINING_DATASET_PATH = "data/cleaned_data.csv"  # Canonical training dataset (mapped columns)
JOINED_DATASET_PATH = "data/joined_data.csv"  # Joined dataset for analysis
PRIMARY_DATASET_PATH = TRAINING_DATASET_PATH  # Alias for clarity

# --- DEFAULTS FOR ANALYSIS TYPE AND TARGET VARIABLE ---
DEFAULT_ANALYSIS_TYPE = 'correlation'

# --- DATASET SCHEMA (column list) -----------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    # Always use the canonical dataset for schema detection
    _schema_path = os.path.join(BASE_DIR, PRIMARY_DATASET_PATH)
    if not os.path.exists(_schema_path):
        raise FileNotFoundError(f"Primary dataset not found at {_schema_path}. Ensure data/cleaned_data.csv is present.")

    _df_schema = pd.read_csv(_schema_path, nrows=1)
    AVAILABLE_COLUMNS = set(_df_schema.columns)
    logging.getLogger("schema").info(f"[schema] Loaded {len(AVAILABLE_COLUMNS)} columns from {_schema_path}")
except Exception as schema_err:
    AVAILABLE_COLUMNS = set()
    logging.getLogger("schema").error(f"[schema] Could not load dataset schema: {schema_err}")

# ------------- SCHEMA ENDPOINT -------------
@cross_origin(origins="*", methods=['GET'])
@app.route('/schema', methods=['GET'])
def get_schema():
    """Return the list of available data columns so the UI can validate queries dynamically."""
    if not AVAILABLE_COLUMNS:
        return jsonify({"success": False, "error": "Schema not loaded"}), 500
    return jsonify({"success": True, "columns": sorted(AVAILABLE_COLUMNS)})

# Logging setup (must be before any use of logger)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("shap-microservice")
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
numeric_level = getattr(logging, LOG_LEVEL.upper(), None)
if not isinstance(numeric_level, int):
    numeric_level = getattr(logging, 'INFO')
logger.setLevel(numeric_level)


# --- EARLY FLASK APP DEFINITION AND DECORATORS (fixes NameError and require_api_key) ---


# --- REDIS/RQ SETUP ---
REDIS_URL = os.getenv('REDIS_URL', 'rediss://default:AVnAAAIjcDEzZjMwMjdiYWI5MjA0NjY3YTQ4ZjRjZjZjNWZhNTdmM3AxMA@ruling-stud-22976.upstash.io:6379')
logger = logging.getLogger("shap-microservice")
logger.info(f"[DEBUG] Using Redis URL: {REDIS_URL}")

# Initialize Redis connection and queue more gracefully
redis_conn = None
queue = None

# First create a Flask application context
with app.app_context():
    try:
        # Apply Redis connection patches for better stability
        apply_all_worker_patches(app)
        
        # Create Redis connection with improved parameters
        logger.info("Attempting to connect to Redis...")
        redis_conn = redis.from_url(
            REDIS_URL,
            socket_timeout=10,
            socket_connect_timeout=10,
            socket_keepalive=True,
            health_check_interval=30,
            retry_on_timeout=True,
            ssl_cert_reqs=None # Don't verify SSL certificate
        )
        
        # Test the connection with timeout
        redis_conn.ping()
        logger.info("Successfully connected to Redis")
        
        # Store Redis connection in app config for endpoint access
        app.config['redis_conn'] = redis_conn
        
        # Initialize the job queue with the connection
        logger.info("Initializing Redis Queue for job processing...")
        queue = Queue('shap-jobs', connection=redis_conn)
        
    except Exception as e:
        logger.error(f"Failed to initialize Redis connection: {str(e)}")
        logger.error("App will start without Redis - some features may be unavailable")
        # Don't raise the exception - let the app start without Redis
        app.config['redis_conn'] = None

load_dotenv()
API_KEY = os.getenv('API_KEY')
REQUIRE_AUTH = True  # Re-enable API key requirement for Render

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Allow OPTIONS requests to pass through without API key check for CORS preflight
        if request.method == 'OPTIONS':
            return f(*args, **kwargs)
        if not REQUIRE_AUTH:
            return f(*args, **kwargs)
        
        # Check for standard 'Authorization: Bearer <token>' header first
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            api_key = auth_header.split(' ')[1]
        else:
            # Fallback to 'X-API-KEY' for backward compatibility
            api_key = request.headers.get('X-API-KEY')

        if not api_key or api_key != API_KEY:
            return jsonify({"success": False, "error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function




# --- ASYNC ANALYSIS WORKER FUNCTION FOR RQ ---
def create_memory_optimized_explainer(model, data, max_rows=10):
    """Create a memory-optimized explainer that processes data in batches."""
    try:
        # Enable garbage collection
        gc.enable()
        
        # Log initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Validate data
        if 'ID' not in data.columns:
            # Try alternative ID fields
            id_candidates = ['OBJECTID', 'FID', 'id', 'objectid']
            id_field = None
            for candidate in id_candidates:
                if candidate in data.columns:
                    id_field = candidate
                    logger.info(f"Using {candidate} as ID field")
                    break
            
            if not id_field:
                logger.warning("No ID field found - creating synthetic IDs")
                # Create synthetic ID field
                data = data.copy()
                data['ID'] = range(len(data))
            else:
                # Rename the ID field to 'ID' for consistency
                data = data.copy()
                data['ID'] = data[id_field]
            
        if data['ID'].duplicated().any():
            logger.warning("Data contains duplicate IDs - removing duplicates")
            data = data.drop_duplicates(subset=['ID'])
            
        # Log data info
        logger.info(f"Processing data with {len(data)} rows and {len(data.columns)} columns")
        
        # Create explainer once outside the loop
        logger.info("Creating TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        logger.info("TreeExplainer created successfully")
        
        # Process in batches
        total_rows = len(data)
        num_batches = (total_rows + max_rows - 1) // max_rows
        all_shap_values = []
        
        start_time = time.time()
        timeout = 180  # Increase timeout to 3 minutes for 128-feature model on standard plan
        
        for i in range(0, total_rows, max_rows):
            # Check for timeout
            if time.time() - start_time > timeout:
                logger.warning("Processing timeout reached, stopping batch processing")
                break
                
            batch_start = time.time()
            batch_end = min(i + max_rows, total_rows)
            batch = data.iloc[i:batch_end].copy()  # Create a copy to avoid memory leaks
            
            logger.info(f"Processing batch {i//max_rows + 1}/{num_batches} ({len(batch)} rows)")
            
            try:
                # Calculate SHAP values for batch
                batch_shap = explainer.shap_values(batch, check_additivity=False)
                all_shap_values.append(batch_shap)
                
                # Log batch progress
                batch_time = time.time() - batch_start
                logger.info(f"Batch {i//max_rows + 1} completed in {batch_time:.2f} seconds")
                
                # Force garbage collection
                del batch
                gc.collect()
                
                # Log memory usage after batch
                current_memory = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory usage after batch: {current_memory:.2f} MB")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//max_rows + 1}: {str(e)}")
                continue  # Continue with next batch even if this one fails
        
        # Combine SHAP values
        if not all_shap_values:
            logger.error("No SHAP values were computed successfully")
            return None
            
        try:
            if len(all_shap_values) > 1:
                shap_values = np.concatenate(all_shap_values, axis=0)
            else:
                shap_values = all_shap_values[0]
                
            # Log final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024
            logger.info(f"Final memory usage: {final_memory:.2f} MB")
            
            return shap_values
            
        except Exception as e:
            logger.error(f"Error combining SHAP values: {str(e)}")
            return None
        
    except Exception as e:
        logger.error(f"Error in create_memory_optimized_explainer: {str(e)}")
        raise

from map_nesto_data import FIELD_MAPPINGS, TARGET_VARIABLE

def generate_analysis_summary(query: str, target_field: str, results: List[Dict], feature_importance: List[Dict], query_type: str = 'unknown') -> str:
    """Generate a natural language summary of the analysis results"""
    
    if not results or len(results) == 0:
        return "No results found matching the specified criteria."
    
    # Special handling for application count queries
    if target_field.lower() == 'frequency' and ('application' in query.lower() or query_type == 'topN'):
        top_areas = [result.get('FSA_ID', result.get('ID', 'Unknown Area')) for result in results[:5]]
        top_counts = [int(result.get('FREQUENCY', 0)) for result in results[:5]]
        
        summary = f"The areas with the most mortgage applications are {top_areas[0]} ({top_counts[0]} applications)"
        if len(top_areas) > 1:
            summary += f", followed by {top_areas[1]} ({top_counts[1]} applications)"
        if len(top_areas) > 2:
            summary += f", and {top_areas[2]} ({top_counts[2]} applications)"
        summary += "."
        
        # Only add feature importance if it's relevant to applications
        relevant_features = [f for f in feature_importance if 
            any(term in f['feature'].lower() for term in ['household', 'income', 'population', 'density'])]
        
        if relevant_features:
            summary += f" These areas tend to have higher {relevant_features[0]['feature'].lower().replace('_', ' ')}"
            if len(relevant_features) > 1:
                summary += f" and {relevant_features[1]['feature'].lower().replace('_', ' ')}"
            summary += "."
        
        return summary
    
    # Handle other query types
    # ... existing code for other analysis types ...

def calculate_feature_importance(data: pd.DataFrame, target: str) -> List[Dict]:
    """Generic Pearson correlation-based importance for numeric target variables."""
    if target not in data.columns:
        return []

    # Consider numeric columns only (exclude the target itself)
    numeric_cols = [c for c in data.select_dtypes(include=['number']).columns if c != target]

    importances: List[Dict] = []
    for col in numeric_cols:
        corr = data[col].corr(data[target])
        if pd.notna(corr) and abs(corr) > 0.1:
            importances.append({
                'feature': col,
                'importance': corr,
                'correlation_type': 'positive' if corr > 0 else 'negative'
            })

    importances.sort(key=lambda x: abs(x['importance']), reverse=True)
    return importances

def sanitize_for_json(obj):
    """Recursively replace NaN, Infinity, -Infinity with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    else:
        return obj

# ----------------------- REQUEST VALIDATION ---------------------------
ALLOWED_ANALYSIS_TYPES = {
    'jointHigh', 'joint_high', 'correlation', 'ranking', 'distribution', 'trends', 'topN'
}


def validate_analysis_request(req_json: Dict[str, Any]):
    """Raise APIError if the request payload is invalid."""
    if not isinstance(req_json, dict):
        raise APIError("Request body must be a JSON object", 400)

    # analysis_type
    analysis_type = req_json.get('analysis_type') or req_json.get('analysisType')
    if not analysis_type or analysis_type not in ALLOWED_ANALYSIS_TYPES:
        raise APIError(f"'analysis_type' must be one of {sorted(ALLOWED_ANALYSIS_TYPES)}", 400)

    # target_variable
    target_variable = req_json.get('target_variable') or req_json.get('targetVariable')
    if not target_variable:
        raise APIError("'target_variable' is required", 400)
    if target_variable not in AVAILABLE_COLUMNS:
        raise APIError(f"target_variable '{target_variable}' not found in dataset", 400)

    # matched_fields / metrics
    matched_fields = req_json.get('matched_fields') or req_json.get('matchedFields') or []
    if isinstance(matched_fields, str):
        matched_fields = [matched_fields]
    bad_fields = [m for m in matched_fields if m not in AVAILABLE_COLUMNS]
    if bad_fields:
        raise APIError(f"Unknown metric fields: {', '.join(bad_fields)}", 400)

    # All good – normalized payload can be returned if needed
    return {
        'analysis_type': analysis_type,
        'target_variable': target_variable,
        'matched_fields': matched_fields,
    }

# --- ASYNC /analyze ENDPOINT FOR RENDER ---
@cross_origin(origins="*", methods=['POST', 'OPTIONS'], headers=['Content-Type', 'X-API-KEY'], supports_credentials=True)
@app.route('/analyze', methods=['POST'])
@require_api_key
def analyze():
    logger.info("/analyze endpoint called (ASYNC POST)")
    query = request.json
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Check if Redis/queue is available
    if queue is None:
        logger.error("Redis queue not available - cannot process async jobs")
        return jsonify({
            "success": False, 
            "error": "Analysis service temporarily unavailable. Redis connection failed during startup.",
            "retry_suggestion": "Please try again in a few minutes as the service may be initializing."
        }), 503
    
    try:
        # Validate the request early
        try:
            validate_analysis_request(query)
        except APIError as ve:
            return handle_api_error(ve)

        # Ensure 'analysis_worker' is defined later in the file
        job = queue.enqueue(analysis_worker, query, job_timeout=600)
        logger.info(f"Enqueued job {job.id}")
        return jsonify({"job_id": job.id, "status": "queued"}), 202
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"[ANALYZE ENQUEUE ERROR] Exception: {e}\nTraceback:\n{tb}")
        return jsonify({"success": False, "error": str(e), "traceback": tb}), 500


# --- JOB STATUS ENDPOINT ---
@app.route('/job_status/<job_id>', methods=['GET'])
@require_api_key
def job_status(job_id):
    # Check if Redis/queue is available
    if queue is None:
        logger.error("Redis queue not available - cannot check job status")
        return jsonify({
            "success": False, 
            "error": "Job status unavailable. Redis connection failed during startup."
        }), 503
    
    try:
        job = queue.fetch_job(job_id)
        if job is None:
            return jsonify({"success": False, "error": "Job not found"}), 404

        # Normalize job status for frontend compatibility
        if job.is_finished:
            result = job.result
            if result is None:
                return jsonify({"success": False, "error": "Job finished but no result found."}), 500

            # The frontend expects the string 'completed' to indicate success.
            # Preserve backwards-compatibility by also including the legacy 'finished'.
            return jsonify({
                "success": True,
                "status": "completed",
                "legacy_status": "finished",
                "result": result
            })
        elif job.is_failed:
            return jsonify({"success": False, "status": "failed", "error": str(job.exc_info)})
        else:
            return jsonify({"success": True, "status": job.get_status()})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"[JOB STATUS ERROR] Exception: {e}\nTraceback:\n{tb}")
        return jsonify({"success": False, "error": str(e), "traceback": tb}), 500

# --- LOGS ENDPOINT ---
@app.route('/logs', methods=['GET'])
@require_api_key
def get_logs():
    """Get recent logs from the application"""
    try:
        # Get Redis connection
        redis_conn = get_redis_connection()
        
        if not redis_conn:
            return jsonify({
                'success': False,
                'error': 'Redis connection not available'
            }), 500
        
        # Get worker logs from Redis if available
        logs_key = 'shap-service:logs'
        logs = redis_conn.lrange(logs_key, -100, -1)  # Get last 100 log entries
        
        log_entries = []
        for log_entry in logs:
            try:
                log_entries.append(log_entry.decode('utf-8'))
            except Exception:
                log_entries.append(str(log_entry))
        
        # If no logs in Redis, return a message
        if not log_entries:
            log_entries = ["No recent logs available in Redis"]
        
        return jsonify({
            'success': True,
            'logs': log_entries,
            'count': len(log_entries)
        })
        
    except Exception as e:
        logger.error(f"Error getting logs: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

from data_versioning import DataVersionTracker


## --- REMOVED DUPLICATE DEFINITIONS OF app, require_api_key, job_store ---

# Timeout handler
def timeout_handler(timeout=25):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'RENDER' not in os.environ:
                return f(*args, **kwargs)
            result = {"value": None, "error": None}
            def target():
                try:
                    result["value"] = f(*args, **kwargs)
                except Exception as e:
                    result["error"] = str(e)
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout)
            if thread.is_alive():
                logger.warning(f"Function {f.__name__} timed out after {timeout} seconds")
                return jsonify({
                    "success": False,
                    "error": "Request timed out. The operation is still processing in the background.",
                    "retry_suggestion": "This request is taking longer than expected. Please try again in a few moments."
                }), 503
            if result["error"] is not None:
                raise Exception(result["error"])
            return result["value"]
        return decorated_function
    return decorator

# Version tracker
version_tracker = DataVersionTracker()

# Lazy model/data loading
model = None
dataset = None
feature_names = None
def ensure_model_loaded():
    global model, feature_names
    try:
        if model is None or feature_names is None:  # Remove dataset check since it's created per query
            logger.info("Lazy-loading model and feature names...")
            model_, feature_names_ = load_model()
            if model_ is None:
                raise RuntimeError("Model failed to load (model_ is None)")
            if not feature_names_:
                raise RuntimeError("Feature names failed to load (feature_names_ is empty or None)")
            model = model_
            feature_names = feature_names_
            logger.info(f"Model and feature names loaded successfully (lazy): model={type(model)}, features={len(feature_names)}")
        return model, feature_names
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"[ENSURE_MODEL_LOADED ERROR] Exception: {e}\nTraceback:\n{tb}")
        print(f"[ENSURE_MODEL_LOADED ERROR] Exception: {e}\nTraceback:\n{tb}")
        sys.stdout.flush()
        raise

def load_model():
    try:
        logger.info("Loading model and feature names...")
        import xgboost as xgb
        is_render = 'RENDER' in os.environ
        if is_render:
            gc.enable()
            
        # Prefer modern JSON/UBJ model if present (version-agnostic)
        json_model_path = os.path.splitext(MODEL_PATH)[0] + ".json"
        ubj_model_path  = os.path.splitext(MODEL_PATH)[0] + ".ubj"

        if os.path.exists(json_model_path):
            model = xgb.Booster()
            model.load_model(json_model_path)
        elif os.path.exists(ubj_model_path):
            model = xgb.Booster()
            model.load_model(ubj_model_path)
        elif os.path.exists(MODEL_PATH):
            # Fallback to legacy pickle (may warn on newer XGBoost)
            model = pickle.load(open(MODEL_PATH, 'rb'))
        else:
            raise FileNotFoundError("No model file found (.json, .ubj, or .pkl). Please train or re-export the model.")

        # Load feature names if available
            if os.path.exists(FEATURE_NAMES_PATH):
                with open(FEATURE_NAMES_PATH, 'r') as f:
                    feature_names = [line.strip() for line in f.readlines()]
            else:
                feature_names = []
                
            return model, feature_names  # Return None for dataset as it will be created per query
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.route('/ping', methods=['GET'])
def ping():
    """Simple endpoint to check if service is running - no auth required."""
    import datetime
    return jsonify({
        "status": "ok",
        "message": "SHAP Microservice is running",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "version": "1.2.0"  # Update this when making significant changes
    })

@app.route('/versions', methods=['GET'])
@require_api_key
def list_versions():
    ensure_model_loaded()
    try:
        versions = version_tracker.list_all_versions()
        simplified_versions = {"datasets": {}, "models": {}}
        for version_id, info in versions.get("datasets", {}).items():
            simplified_versions["datasets"][version_id] = {
                "timestamp": info.get("timestamp"),
                "description": info.get("description"),
                "source": info.get("source"),
                "row_count": info.get("row_count"),
                "column_count": info.get("column_count"),
                "columns": info.get("columns")
            }
        for version_id, info in versions.get("models", {}).items():
            simplified_versions["models"][version_id] = {
                "timestamp": info.get("timestamp"),
                "dataset_version_id": info.get("dataset_version_id"),
                "metrics": info.get("metrics", {}),
                "feature_names": info.get("feature_names")
            }
        return jsonify({"success": True, "versions": simplified_versions})
    except Exception as e:
        logger.error(f"Error listing versions: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
@require_api_key
def health_check():
    try:
        # Load model and features to ensure they're available
        model, feature_names = ensure_model_loaded()
        
        # Calculate memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            memory_usage = "psutil not installed"
        
        # Check Redis connection
        redis_status = {"connected": False, "error": None}
        try:
            redis_conn = app.config.get('redis_conn')
            if redis_conn:
                redis_status["connected"] = redis_conn.ping()
            else:
                redis_status["error"] = "Redis connection not found in app config"
        except Exception as e:
            redis_status["error"] = str(e)
        
        # Get version info
        model_version = version_tracker.get_latest_model()
        dataset_version = version_tracker.get_latest_dataset()
        
        model_version_info = None
        if model_version:
            model_version_id, model_info = model_version
            model_version_info = {
                "id": model_version_id,
                "created_at": model_info.get("timestamp"),
                "metrics": model_info.get("metrics", {})
            }
        
        dataset_version_info = None
        if dataset_version:
            dataset_version_id, dataset_info = dataset_version
            dataset_version_info = {
                "id": dataset_version_id,
                "created_at": dataset_info.get("timestamp"),
                "record_count": dataset_info.get("row_count"),
                "description": dataset_info.get("description"),
                "source": dataset_info.get("source")
            }
        
        # Try to load dataset info (optional)
        dataset_info = None
        try:
            dataset_path = PRIMARY_DATASET_PATH
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Primary dataset not found at {dataset_path}")
            
            # Just get basic info without loading full dataset
            import pandas as pd
            sample_data = pd.read_csv(dataset_path, nrows=1)  # Read just one row to get columns
            dataset_info = {
                "columns": list(sample_data.columns),
                "column_count": len(sample_data.columns),
                "path": dataset_path
            }
        except Exception as e:
            logger.warning(f"Could not load dataset info: {str(e)}")
            dataset_info = None
        
        # Get XGBoost version safely
        xgboost_version = "unknown"
        try:
            import xgboost
            xgboost_version = xgboost.__version__
        except ImportError:
            xgboost_version = "not installed"
        
        return jsonify({
            "status": "healthy",
            "model": {
                "type": "xgboost",
                "version": xgboost_version,
                "feature_count": len(feature_names) if feature_names else 0,
                "features": feature_names[:10] if feature_names else [],  # Limit to first 10 features
                "version_info": model_version_info
            },
            "dataset": dataset_info,
            "redis_connected": redis_status["connected"],
            "redis_status": redis_status,
            "system_info": {
                "python_version": platform.python_version(),
                "system": platform.system(),
                "memory_usage_mb": memory_usage
            },
            "shap_version": shap.__version__
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500



@app.route('/metadata', methods=['GET'])
@require_api_key
def get_metadata():
    try:
        # Load model and features to ensure they're available
        model, feature_names = ensure_model_loaded()
        
        # Load dataset for metadata
        try:
            dataset_path = PRIMARY_DATASET_PATH
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Primary dataset not found at {dataset_path}")
            
            if not os.path.exists(dataset_path):
                raise APIError("Dataset file not found", 500)
            
            dataset = pd.read_csv(dataset_path)
            logger.info(f"Successfully loaded dataset from {dataset_path} for metadata")
        except Exception as e:
            logger.error(f"Error loading dataset for metadata: {str(e)}")
            raise APIError(f"Dataset not available: {str(e)}", 500)
        
        # Calculate summary statistics
        summary_stats = {}
        for column in dataset.columns:
            if column in ['zip_code']:
                continue
            if np.issubdtype(dataset[column].dtype, np.number):
                try:
                    column_stats = {
                        "mean": float(dataset[column].mean()),
                        "median": float(dataset[column].median()),
                        "min": float(dataset[column].min()),
                        "max": float(dataset[column].max()),
                        "std": float(dataset[column].std())
                    }
                    summary_stats[column] = column_stats
                except Exception as e:
                    logger.warning(f"Could not calculate stats for column {column}: {str(e)}")
        
        # Calculate correlations with target variable
        correlations = None
        if TARGET_VARIABLE in dataset.columns:
            correlations = {}
            for column in dataset.columns:
                if column != TARGET_VARIABLE and np.issubdtype(dataset[column].dtype, np.number):
                    try:
                        correlation = dataset[column].corr(dataset[TARGET_VARIABLE])
                        if not np.isnan(correlation):
                            correlations[column] = float(correlation)
                    except Exception as e:
                        logger.warning(f"Could not calculate correlation for column {column}: {str(e)}")
        
        # Get version info
        dataset_version = version_tracker.get_latest_dataset()
        version_info = None
        if dataset_version:
            dataset_version_id, dataset_info = dataset_version
            version_info = {
                "id": dataset_version_id,
                "created_at": dataset_info.get("timestamp"),
                "description": dataset_info.get("description"),
                "source": dataset_info.get("source")
            }
        
        return jsonify({
            "success": True,
            "columns": list(dataset.columns),
            "record_count": len(dataset),
            "statistics": summary_stats,
            "correlations_with_target": correlations,
            "target_variable": TARGET_VARIABLE,
            "version_info": version_info
        })
    except APIError:
        raise  # Re-raise API errors as-is
    except Exception as e:
        logger.error(f"Error getting metadata: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

# Add worker status endpoint
@app.route('/worker-status', methods=['GET'])
def worker_status():
    """Check status of RQ workers"""
    try:
        # Get Redis connection
        redis_conn = get_redis_connection() 
        
        if not redis_conn:
            return jsonify({
                'status': 'error',
                'message': 'Redis connection not available'
            }), 500
        
        # Get all workers
        workers_key = 'rq:workers'
        worker_keys = redis_conn.smembers(workers_key)
        
        # Check if there are any workers
        if not worker_keys:
            return jsonify({
                'status': 'warning',
                'message': 'No workers registered',
                'workers': [],
                'active_workers': 0,
                'total_workers': 0
            })
            
        workers_info = []
        active_workers = 0
        for worker_key in worker_keys:
            try:
                # Get worker info
                worker_name = worker_key.decode('utf-8').replace('rq:worker:', '')
                
                # Check worker heartbeat
                heartbeat_key = f"{worker_key.decode('utf-8')}:heartbeat"
                last_heartbeat = redis_conn.get(heartbeat_key)
                
                if last_heartbeat:
                    # Calculate time since last heartbeat
                    last_beat_time = float(last_heartbeat.decode('utf-8'))
                    time_since_beat = time.time() - last_beat_time
                    is_active = time_since_beat < 60  # Consider active if heartbeat in last 60 seconds
                    if is_active:
                        active_workers += 1
                else:
                    time_since_beat = None
                    is_active = False
                
                # Get current jobs
                current_job_id = redis_conn.get(f"{worker_key.decode('utf-8')}:current_job")
                if current_job_id:
                    current_job = current_job_id.decode('utf-8')
                else:
                    current_job = None
                
                # Add to workers info
                workers_info.append({
                    'name': worker_name,
                    'active': is_active,
                    'last_heartbeat_seconds_ago': time_since_beat if time_since_beat else None,
                    'current_job': current_job
                })
            except Exception as e:
                logger.error(f"Error getting info for worker {worker_key}: {str(e)}")
                workers_info.append({
                    'name': worker_key.decode('utf-8'),
                    'error': str(e)
                })
        
        # Return worker status
        return jsonify({
            'status': 'ok' if active_workers > 0 else 'warning',
            'active_workers': active_workers,
            'total_workers': len(worker_keys),
            'workers': workers_info
        })
        
    except Exception as e:
        logger.error(f"Error checking worker status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# --- ADMIN ENDPOINTS ---
@app.route('/admin/queue_status', methods=['GET'])
@require_api_key
def admin_queue_status():
    """Get detailed queue status for admin purposes"""
    try:
        # Get Redis connection
        redis_conn = get_redis_connection()
        
        if not redis_conn:
            return jsonify({
                'success': False,
                'error': 'Redis connection not available',
                'redis_connected': False
            }), 500
        
        # Test Redis connection
        try:
            redis_conn.ping()
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Redis connection failed: {str(e)}',
                'redis_connected': False
            }), 500
        
        # Get queue information
        from rq import Queue
        try:
            queue_instance = Queue('shap-jobs', connection=redis_conn)
            
            # Get job counts with error handling
            try:
                queued_jobs = len(queue_instance)
            except Exception:
                queued_jobs = 0
            
            try:
                started_jobs = len(queue_instance.started_job_registry)
            except Exception:
                started_jobs = 0
            
            try:
                finished_jobs = len(queue_instance.finished_job_registry)
            except Exception:
                finished_jobs = 0
            
            try:
                failed_jobs = len(queue_instance.failed_job_registry)
            except Exception:
                failed_jobs = 0
            
            # Get recent job IDs with error handling
            recent_queued = []
            recent_started = []
            recent_finished = []
            recent_failed = []
            
            try:
                recent_queued = queue_instance.job_ids[:5]
            except Exception:
                pass
            
            try:
                recent_started = list(queue_instance.started_job_registry.get_job_ids())[:5]
            except Exception:
                pass
            
            try:
                recent_finished = list(queue_instance.finished_job_registry.get_job_ids())[:5]
            except Exception:
                pass
            
            try:
                recent_failed = list(queue_instance.failed_job_registry.get_job_ids())[:5]
            except Exception:
                pass
            
            return jsonify({
                'success': True,
                'redis_connected': True,
                'queue_name': 'shap-jobs',
                'queued_jobs': queued_jobs,
                'in_progress_jobs': started_jobs,
                'completed_jobs': finished_jobs,
                'failed_jobs': failed_jobs,
                'recent_jobs': {
                    'queued': recent_queued,
                    'started': recent_started,
                    'finished': recent_finished,
                    'failed': recent_failed
                }
            })
            
        except Exception as e:
            logger.error(f"Error creating queue instance: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'Failed to access queue: {str(e)}',
                'redis_connected': True
            }), 500
        
    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'redis_connected': False
        }), 500

# Error handlers
class APIError(Exception):
    def __init__(self, message, status_code=400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

@app.errorhandler(APIError)
def handle_api_error(error):
    response = jsonify({"success": False, "error": error.message})
    response.status_code = error.status_code
    return response

@app.errorhandler(Exception)
def handle_generic_exception(error):
    logger.error(f"Unhandled exception: {str(error)}")
    logger.error(traceback.format_exc())
    response = jsonify({"success": False, "error": "An internal server error occurred. Please try again later."})
    response.status_code = 500
    return response

def get_redis_connection():
    """Get Redis connection from app config or create a new one"""
    if hasattr(app, 'config') and 'redis_conn' in app.config:
        return app.config['redis_conn']
    
    # Try to create a new connection
    try:
        import redis
        import os
        
        redis_url = os.environ.get('REDIS_URL')
        if not redis_url:
            logger.error("REDIS_URL environment variable not set")
            return None
            
        # Create Redis connection with improved parameters
        redis_conn = redis.from_url(
            redis_url,
            socket_timeout=int(os.environ.get('REDIS_TIMEOUT', '10')),
            socket_connect_timeout=int(os.environ.get('REDIS_CONNECT_TIMEOUT', '10')),
            socket_keepalive=os.environ.get('REDIS_SOCKET_KEEPALIVE', 'true').lower() == 'true',
            health_check_interval=int(os.environ.get('REDIS_HEALTH_CHECK_INTERVAL', '30')),
            retry_on_timeout=True
        )
        
        # Store in app config for future use
        if hasattr(app, 'config'):
            app.config['redis_conn'] = redis_conn
            
        return redis_conn
    except Exception as e:
        logger.error(f"Error creating Redis connection: {str(e)}")
        return None

def analysis_worker(query):
    """Worker function to process analysis requests"""
    try:
        # Extract query parameters
        user_query = query.get('query', '')
        analysis_type = query.get('analysis_type', 'unknown')
        target_field = query['target_variable']
        demographic_filters = query.get('demographic_filters', [])
        conversation_context = query.get('conversationContext', '')
        min_applications = query.get('minApplications', 1)

        # Use the preprocessed global dataframe; reload if it failed earlier
        global data
        if data is None or data.empty:
            logger.warning("Global dataframe 'data' is empty – reloading via load_and_preprocess_data()")
            data = load_and_preprocess_data()
            if data is None or data.empty:
                raise APIError("Failed to load analysis dataset.", 500)

        # The data is now loaded and columns are renamed.
        # We can directly use the clean conceptual names from the request.

        # Normalize analysis type
        if analysis_type == 'topN':
            logger.info("Performing SHAP analysis for 'topN' type...")
            analysis_type = 'ranking'

        # --- General Purpose Filtering ---
        data_to_process = data.copy()
        if demographic_filters:
            for filt in demographic_filters:
                field = filt.get('field')
                op = filt.get('op')
                value = filt.get('value')

                if analysis_type == 'jointHigh' and field:
                    # (Filtering logic as previously implemented)
                    pass
                elif field and op and value is not None:
                    if field not in data_to_process.columns:
                        logger.warning(f"Field '{field}' from filter not in data columns. Skipping.")
                        continue
                    try:
                        if op == '>': data_to_process = data_to_process[data_to_process[field] > value]
                        elif op == '>=': data_to_process = data_to_process[data_to_process[field] >= value]
                        elif op == '<': data_to_process = data_to_process[data_to_process[field] < value]
                        elif op == '<=': data_to_process = data_to_process[data_to_process[field] <= value]
                        elif op == '==':
                            # Case-insensitive comparison for strings
                            if isinstance(data_to_process[field].dtype, object):
                                data_to_process = data_to_process[data_to_process[field].str.lower() == str(value).lower()]
                            else:
                                data_to_process = data_to_process[data_to_process[field] == value]
                        elif op == '!=':
                            if isinstance(data_to_process[field].dtype, object):
                                data_to_process = data_to_process[data_to_process[field].str.lower() != str(value).lower()]
                            else:
                                data_to_process = data_to_process[data_to_process[field] != value]
                        else:
                            logger.warning(f"Unsupported operator '{op}' in filter. Skipping.")
                    except Exception as e:
                        logger.error(f"Could not apply filter {filt}: {e}")

        # --- Analysis Logic ---
        results: List[Dict] = []
        feature_importance: List[Dict] = []
        analysis_summary = ""
        confidence = 0.85

        # ---------------- JOINT HIGH IMPLEMENTATION ----------------
        if analysis_type in ("joint_high", "jointHigh"):
            try:
                # Collect metric fields: target + matched_fields if provided
                matched_fields = query.get("matched_fields") or query.get("matchedFields") or []
                if isinstance(matched_fields, str):
                    matched_fields = [matched_fields]

                metrics: List[str] = [target_field] + [f for f in matched_fields if f and f != target_field]

                # Ensure metrics exist in dataframe
                metrics = [m for m in metrics if m in data_to_process.columns]
                if len(metrics) < 2:
                    logger.warning("[joint_high] Fewer than 2 valid metric fields found – returning empty result")
                    return {
                        'success': True,
                        'results': [],
                        'summary': 'Not enough data for joint-high analysis',
                        'feature_importance': [],
                        'confidence': 0.5,
                        'visualizationData': []
                    }

                # --- 2025-06-13: Percentile mask disabled to allow larger result set ---
                # thresholds: Dict[str, float] = {}
                # for m in metrics:
                #     thresholds[m] = data_to_process[m].quantile(0.75)
                # logger.info(f"[joint_high] Thresholds: {thresholds}")
                # mask = np.ones(len(data_to_process), dtype=bool)
                # for m in metrics:
                #     mask &= data_to_process[m] >= thresholds[m]
                # high_df = data_to_process.loc[mask].copy()

                # For now, evaluate combined score over the **entire** dataset
                high_df = data_to_process.copy()

                # Compute combined score and take top 100 rows
                for m in metrics:
                    norm_col = f"{m}_norm"
                    high_df[norm_col] = (high_df[m] - high_df[m].min()) / (high_df[m].max() - high_df[m].min() + 1e-9)
                high_df['combined_score'] = high_df[[f"{m}_norm" for m in metrics]].mean(axis=1)

                # honour optional top_n parameter – if omitted, return **all rows**
                top_n = query.get('top_n')
                if top_n is None:
                    top_n = query.get('limit')
                if top_n is None:
                    top_n = -1  # Default: no limit

                try:
                    top_n = int(top_n)
                except (TypeError, ValueError):
                    top_n = -1

                high_df = high_df.sort_values('combined_score', ascending=False)
                if top_n > 0:
                    high_df = high_df.head(top_n)

                # Dynamically choose an identifier column
                id_col = None
                if 'geo_id' in high_df.columns:
                    id_col = 'geo_id'
                elif 'province_code' in high_df.columns:
                    id_col = 'province_code'
                elif 'ID' in high_df.columns:
                    id_col = 'ID'
                elif 'OBJECTID' in high_df.columns:
                    id_col = 'OBJECTID'

                # Fall back to using the dataframe index when no ID column is present
                if id_col is None:
                    high_df = high_df.reset_index().rename(columns={'index': 'row_index'})
                    id_col = 'row_index'

                # Guarantee a canonical 'ID' column for downstream joins
                if id_col != 'ID':
                    high_df['ID'] = high_df[id_col]

                # Always include canonical 'geo_id' and legacy 'ID'
                if 'geo_id' not in high_df.columns and id_col != 'geo_id':
                    high_df['geo_id'] = high_df[id_col]
                if 'ID' not in high_df.columns:
                    high_df['ID'] = high_df['geo_id']

                # Return **all** available columns (features) for each matched area
                output_cols = high_df.columns.tolist()

                results = high_df[output_cols].to_dict(orient='records')

                # --------------------- FEATURE IMPORTANCE ---------------------
                feature_importance = calculate_feature_importance(high_df, target_field)

                # --------------------- SUMMARY CONSTRUCTION ------------------
                analysis_summary = generate_simple_summary(results, target_field, metrics[1:])
                if feature_importance:
                    top_feats = ', '.join([fi['feature'].replace('_', ' ') for fi in feature_importance[:3]])
                    analysis_summary += f" Key factors influencing {target_field.replace('_', ' ')} include {top_feats}."

            except Exception as je:
                logger.error(f"[joint_high] Error: {je}")
                results = []
                analysis_summary = f"Joint-high analysis failed: {je}"

        # ---------------- OTHER ANALYSIS TYPES (default SHAP) ----------------
        else:
            # --- Use the enhanced pre-calculated SHAP analysis pipeline ---
            try:
                from enhanced_analysis_worker import enhanced_analysis_worker as _enhanced
                shap_results = _enhanced(query)

                # Validate success flag (the helper returns {'success': bool, …})
                if shap_results and shap_results.get('success') is not False:
                    # Ensure required keys exist and normalise identifiers for the join step
                    raw_results = shap_results.get('results', [])

                    for rec in raw_results:
                        # Promote common identifier keys to canonical names expected by the frontend join
                        if 'geo_id' not in rec:
                            if 'ID' in rec:
                                rec['geo_id'] = rec['ID']
                            elif 'zip_code' in rec:
                                rec['geo_id'] = rec['zip_code']
                        if 'ID' not in rec and 'geo_id' in rec:
                            rec['ID'] = rec['geo_id']

                    results = raw_results
                    feature_importance = shap_results.get('feature_importance', [])
                    analysis_summary = shap_results.get('summary', 'Analysis complete.')
                else:
                    raise ValueError("Enhanced SHAP pipeline returned empty or failed result")
            except Exception as shap_err:
                logger.error(f"[SHAP] Fallback to generic analysis due to error: {shap_err}")
                # Generic statistical fallback (as before)
                results = data_to_process.to_dict(orient='records')
                feature_importance = calculate_feature_importance(data_to_process, target_field)
                analysis_summary = generate_simple_summary(results, target_field, [])


        return {
            'success': True,
            'results': sanitize_for_json(results),
            'summary': analysis_summary,
            'feature_importance': sanitize_for_json(feature_importance),
            'confidence': confidence,
            'visualizationData': []
        }
    except Exception as e:
        logger.error(f"Error in analysis worker: {str(e)}")
        logger.error(traceback.format_exc())
        raise APIError(f"Analysis failed: {str(e)}")

def load_and_preprocess_data():
    """
    Loads and preprocesses the training data from a CSV file.
    This function now uses the correct data mapping script.
    """
    try:
        base_dir = os.path.dirname(__file__)
        raw_path = os.path.join(base_dir, 'data', 'nesto_merge_0.csv')
        mapped_path = os.path.join(base_dir, 'data', 'cleaned_data.csv')

        # Honour Render's container path layout as a secondary lookup
        if not os.path.exists(raw_path):
            alt_raw = '/opt/render/project/src/data/nesto_merge_0.csv'
            raw_path = alt_raw if os.path.exists(alt_raw) else raw_path

        def regenerate_mapped():
            logger.info("Generating cleaned_data.csv from the latest nesto_merge_0.csv …")
            from map_nesto_data import map_nesto_data as _map
            _map(raw_path, mapped_path)

        # Determine if we need to (re)generate the mapped file
        need_regen = False
        if not os.path.exists(mapped_path):
            logger.info("cleaned_data.csv is missing – will generate a fresh one.")
            need_regen = True
        elif os.path.getmtime(mapped_path) < os.path.getmtime(raw_path):
            logger.info("cleaned_data.csv is older than nesto_merge_0.csv – will regenerate.")
            need_regen = True

        if need_regen:
            if not os.path.exists(raw_path):
                raise FileNotFoundError(
                    f"Raw dataset not found at {raw_path}; cannot generate cleaned_data.csv"
                )
            try:
                regenerate_mapped()
            except Exception as map_err:
                logger.error(f"Failed to map raw data: {map_err}")
                raise

        # Load the cleaned (schema-standardised) dataset
        processed_df = pd.read_csv(mapped_path, low_memory=False)

        # Quick sanity-check: ensure canonical identifier exists
        if not ({'ID', 'province_code'} & set(processed_df.columns)):
            logger.warning("cleaned_data.csv missing 'ID'/'province_code' – forcing regeneration once.")
            regenerate_mapped()
            processed_df = pd.read_csv(mapped_path, low_memory=False)

        # Log the columns after mapping to confirm correctness
        logger.info(
            "Loaded %d records with %d columns from cleaned_data.csv. Sample columns: %s",
            len(processed_df), len(processed_df.columns), processed_df.columns[:10].tolist()
        )

        return processed_df
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading: {e}")
        return None

# Load data and model at startup
data = load_and_preprocess_data()
if data is None:
    logger.critical("Failed to load data. The application may not function correctly.")

# Load the model and feature names
try:
    # Construct paths relative to the current script's location for robustness
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'models', 'xgboost_model.pkl')
    features_path = os.path.join(base_dir, 'models', 'feature_names.txt')

    # Load the XGBoost model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Successfully loaded the XGBoost model.")

    # Load feature names
    with open(features_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    logger.info("Successfully loaded feature names.")

except FileNotFoundError as e:
    logger.critical(f"Error loading model or feature names: {e}")
    model = None
    feature_names = []
except Exception as e:
    logger.critical(f"An unexpected error occurred during model loading: {e}")
    model = None
    feature_names = []

class ShapAnalyzer:
    def __init__(self, model_path, feature_names_path):
        self.model = self.load_pickle(model_path)
        self.feature_names = self.load_feature_names(feature_names_path)
        self.explainer = shap.TreeExplainer(self.model)

    def load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def load_feature_names(self, path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def get_shap_analysis(self, query, analysis_type):
        # This is a placeholder for the actual SHAP analysis logic.
        # In a real implementation, you would use the query and analysis_type
        # to filter and process data before running the explainer.
        
        # For now, we'll just return some mock data.
        return {
            "feature_importance": [{"feature": "income", "importance": 0.5}],
            "summary": "This is a summary of the analysis.",
            "results": [{"ID": "A1A1A1", "value": 123}]
        }

def generate_simple_summary(results: List[Dict], target: str, metrics: List[str]) -> str:
    if not results:
        return "No results found matching the specified criteria."

    top = results[:3]
    # Cast identifiers to string to ensure safe joining
    areas = [str(r.get('geo_id', r.get('ID', r.get('FSA_ID', 'Unknown')))) for r in top]
    summary = f"Top areas by {target.replace('_', ' ')}: " + ", ".join(areas)
    if metrics:
        summary += f" (evaluated jointly with {', '.join(metrics)})"
    summary += "."
    return summary

# ---------------- REQUEST-ID OBSERVABILITY ----------------

class RequestIDFilter(logging.Filter):
    """Attach the request_id stored on flask.g (if any) to log records."""
    def filter(self, record):
        from flask import g
        record.request_id = getattr(g, 'request_id', '-')
        return True

# Add the filter to the root logger
logging.getLogger().addFilter(RequestIDFilter())

@app.before_request
def attach_request_id():
    from flask import g, request
    rid = request.headers.get('X-Request-ID') or str(uuid.uuid4())
    g.request_id = rid

@app.after_request
def propagate_request_id(resp):
    from flask import g
    if hasattr(g, 'request_id'):
        resp.headers['X-Request-ID'] = g.request_id
    return resp

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}...")
    try:
        app.run(host='0.0.0.0', port=port)
    except Exception as startup_error:
        logger.error(f"Error starting application: {str(startup_error)}")
        sys.exit(1)