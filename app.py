# --- EARLY LOGGING SETUP (fixes NameError: logger not defined) ---
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


# Redis connection patch for better stability
from redis_connection_patch import apply_all_patches
from worker_process_fix import apply_all_worker_patches
# Memory optimization for SHAP analysis
try:
    from shap_memory_fix import apply_memory_patches
except ImportError:
    print("SHAP memory optimization not available. For better performance, run ./deploy_shap_fix.sh")

# --- FLASK APP SETUP (must come after imports) ---
app = Flask(__name__)

# Initialize CORS with proper configuration
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "X-API-KEY"],
        "supports_credentials": True
    }
})

@app.route('/')
def hello_world():
    return jsonify({"message": "SHAP Microservice is running", "status": "healthy"}), 200

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
TRAINING_DATASET_PATH = "data/nesto_merge_0.csv"  # Training dataset
JOINED_DATASET_PATH = "data/joined_data.csv"  # Joined dataset for analysis

# --- DEFAULTS FOR ANALYSIS TYPE AND TARGET VARIABLE ---
DEFAULT_ANALYSIS_TYPE = 'correlation'
DEFAULT_TARGET = 'Mortgage_Approvals'  # Set to actual default target column name for production



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

# First create a Flask application context
with app.app_context():
    try:
        # Apply Redis connection patches for better stability
        apply_all_worker_patches(app)
        
        # Create Redis connection with improved parameters
        redis_conn = redis.from_url(
            REDIS_URL,
            socket_timeout=10,
            socket_connect_timeout=10,
            socket_keepalive=True,
            health_check_interval=30,
            retry_on_timeout=True,
            ssl_cert_reqs=None  # Don't verify SSL certificate
        )
        
        # Test the connection
        if redis_conn.ping():
            logger.info("Successfully connected to Redis")
        else:
            logger.error("Failed to ping Redis server")
            raise Exception("Redis connection test failed")
        
        # Store Redis connection in app config for endpoint access
        app.config['redis_conn'] = redis_conn
        
        # Initialize the job queue with the connection
        logger.info("Initializing Redis Queue for job processing...")
        queue = Queue('shap-jobs', connection=redis_conn)
        
    except Exception as e:
        logger.error(f"Failed to initialize Redis connection: {str(e)}")
        raise

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
        api_key = request.headers.get('X-API-KEY')
        if not api_key or api_key != API_KEY:
            return jsonify({"success": False, "error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function




# --- ASYNC ANALYSIS WORKER FUNCTION FOR RQ ---
def create_memory_optimized_explainer(model, data, max_rows=200):
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
            raise ValueError("Data missing required ID field")
            
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
        timeout = 90  # Set timeout to 90 seconds to allow for cleanup
        
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

def analysis_worker(query):
    import time
    import shap
    import gc
    import psutil
    from fetch_data import fetch_arcgis_data, process_features
    
    logger.info(f"[RQ WORKER] analysis_worker called with query: {query}")
    ensure_model_loaded()
    
    try:
        # Log initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Fetch and join data based on query
        logger.info("Fetching and joining data based on query...")
        all_data = []
        
        # Read layer configurations
        try:
            layers_df = pd.read_csv('nesto.csv')
            logger.info(f"Successfully loaded nesto.csv with {len(layers_df)} layers")
        except Exception as e:
            logger.error(f"Error reading nesto.csv: {str(e)}")
            return {"success": False, "error": f"Failed to read layer configurations: {str(e)}"}
        
        # Process each layer
        for _, row in layers_df.iterrows():
            layer_name = row['Layer Name']
            url = row['URL']
            
            logger.info(f"Processing layer: {layer_name} with URL: {url}")
            try:
                features = fetch_arcgis_data(url)
                logger.info(f"Fetched {len(features)} features from {layer_name}")
                df = process_features(features)
                logger.info(f"Processed features into dataframe with shape: {df.shape}")
            except Exception as e:
                logger.error(f"Error processing layer {layer_name}: {str(e)}")
                continue
            
            if not df.empty:
                # Validate ID field exists
                if 'ID' not in df.columns:
                    logger.error(f"Layer {layer_name} missing required ID field. Available columns: {list(df.columns)}")
                    continue
                    
                # Validate ID field is unique
                if df['ID'].duplicated().any():
                    logger.error(f"Layer {layer_name} has duplicate IDs. Total rows: {len(df)}, Unique IDs: {df['ID'].nunique()}")
                    continue
                    
                # Add layer name as a prefix to columns except ID
                rename_dict = {col: f"{layer_name}_{col}" for col in df.columns if col != 'ID'}
                df = df.rename(columns=rename_dict)
                all_data.append(df)
                logger.info(f"Added {layer_name} data with shape: {df.shape}")
        
        # Merge all data using ID as join key
        if all_data:
            # Start with first dataset
            dataset = all_data[0]
            logger.info(f"Starting merge with {len(dataset)} rows from first layer")
            
            # Join subsequent datasets
            for i, df in enumerate(all_data[1:], 1):
                logger.info(f"Merging layer {i+1} with {len(df)} rows")
                try:
                    dataset = dataset.merge(df, on='ID', how='inner')
                    logger.info(f"After merge: {len(dataset)} rows")
                except Exception as e:
                    logger.error(f"Error merging layer {i+1}: {str(e)}")
                    return {"success": False, "error": f"Failed to merge data: {str(e)}"}
                
                # Validate merge
                if len(dataset) == 0:
                    logger.error("Merge resulted in empty dataset - no matching IDs")
                    return {"success": False, "error": "No matching IDs found across layers"}
                    
            # Validate final dataset
            logger.info(f"Final dataset has {len(dataset)} rows and {len(dataset.columns)} columns")
            if len(dataset) > 2000:  # Sanity check
                logger.warning(f"Final dataset has more rows than expected: {len(dataset)}")
                
            # Save joined data
            output_path = 'data/joined_data.csv'
            try:
                dataset.to_csv(output_path, index=False)
                logger.info(f"Joined data saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving joined data: {str(e)}")
                return {"success": False, "error": f"Failed to save joined data: {str(e)}"}
            
            # Load the saved data for analysis
            logger.info("Loading saved joined data for analysis...")
            try:
                dataset = pd.read_csv(output_path)
                logger.info(f"Successfully loaded data with shape: {dataset.shape}")
            except Exception as e:
                logger.error(f"Error loading saved data: {str(e)}")
                return {"success": False, "error": f"Failed to load saved data: {str(e)}"}
        else:
            logger.error("No data was collected from any layer")
            return {"success": False, "error": "No data collected from any layer"}
        
        # Log dataset info
        logger.info(f"Dataset shape: {dataset.shape}")
        logger.info(f"Dataset columns: {list(dataset.columns)}")
        logger.info(f"Dataset memory usage: {dataset.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
        
        analysis_type = query.get('analysis_type', DEFAULT_ANALYSIS_TYPE)
        target_variable = query.get('target_variable', query.get('target', DEFAULT_TARGET))
        filters = query.get('demographic_filters', [])
        
        logger.info(f"Analysis parameters - type: {analysis_type}, target: {target_variable}, filters: {filters}")
        
        # Validate target variable exists
        if target_variable not in dataset.columns:
            logger.error(f"Target variable {target_variable} not found in dataset. Available columns: {list(dataset.columns)}")
            return {"success": False, "error": f"Target variable {target_variable} not found in dataset"}
        
        # Optimize data loading and filtering
        logger.info("Loading and filtering data...")
        filtered_data = dataset.copy()
        
        # Apply filters more efficiently
        for filter_item in filters:
            if isinstance(filter_item, str):
                if '>' in filter_item:
                    feature, value = filter_item.split('>')
                    feature = feature.strip()
                    value = float(value.strip())
                    filtered_data = filtered_data[filtered_data[feature] > value]
                elif '<' in filter_item:
                    feature, value = filter_item.split('<')
                    feature = feature.strip()
                    value = float(value.strip())
                    filtered_data = filtered_data[filtered_data[feature] < value]
                elif 'high' in filter_item.lower():
                    feature = filter_item.lower().replace('high', '').strip()
                    feature = ''.join([w.capitalize() for w in feature.split(' ')])
                    if feature in filtered_data.columns:
                        threshold = filtered_data[feature].quantile(0.75)
                        filtered_data = filtered_data[filtered_data[feature] > threshold]
        
        logger.info(f"Data filtered. Shape: {filtered_data.shape}")
        
        # Limit the number of rows for analysis
        MAX_ANALYSIS_ROWS = 2000
        if len(filtered_data) > MAX_ANALYSIS_ROWS:
            logger.warning(f"Limiting analysis to {MAX_ANALYSIS_ROWS} rows")
            filtered_data = filtered_data.sample(n=MAX_ANALYSIS_ROWS, random_state=42)
        
        # Optimize data preparation
        top_data = filtered_data.sort_values(by=target_variable, ascending=False)
        X = top_data.copy()
        
        # Drop unnecessary columns more efficiently
        columns_to_drop = ['zip_code', 'latitude', 'longitude', target_variable]
        X = X.drop(columns=[col for col in columns_to_drop if col in X.columns], axis=1)
        
        # Align columns with model features more efficiently
        X = X.reindex(columns=model_features, fill_value=0)
        
        logger.info(f"Data prepared. Shape: {X.shape}")
        
        # Replace the direct SHAP computation with the memory-optimized version
        logger.info("Starting SHAP computation with memory optimization")
        shap_values = create_memory_optimized_explainer(model, X)
        
        if shap_values is None:
            logger.error("SHAP computation failed")
            return {
                "success": False,
                "error": "SHAP computation failed due to timeout or memory constraints",
                "results": [],
                "summary": "Analysis could not be completed due to resource constraints."
            }
            
        logger.info("SHAP computation completed successfully")
        
        feature_importance = []
        for i, feature in enumerate(model_features):
            importance = abs(shap_values[:, i]).mean()
            feature_importance.append({'feature': feature, 'importance': float(importance)})
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        results = []
        for idx, row in top_data.iterrows():
            result = {}
            if 'zip_code' in row:
                result['zip_code'] = str(row['zip_code'])
            if 'latitude' in row and 'longitude' in row:
                result['latitude'] = float(row['latitude'])
                result['longitude'] = float(row['longitude'])
            target_var_lower = target_variable.lower()
            if target_variable in row:
                result[target_var_lower] = float(row[target_variable])
            for col in row.index:
                if col not in ['zip_code', 'latitude', 'longitude', target_variable]:
                    try:
                        result[col.lower()] = float(row[col])
                    except (ValueError, TypeError):
                        if isinstance(row[col], str):
                            result[col.lower()] = row[col]
                        else:
                            result[col.lower()] = str(row[col])
            results.append(result)
        if analysis_type == 'correlation':
            if len(feature_importance) > 0:
                summary = f"Analysis shows a strong correlation between {target_variable} and {feature_importance[0]['feature']}."
            else:
                summary = f"Analysis complete for {target_variable}, but no clear correlations found."
        elif analysis_type == 'ranking':
            if len(results) > 0:
                summary = f"The top area for {target_variable} has a value of {results[0][target_variable.lower()]:.2f}."
            else:
                summary = f"No results found for {target_variable} with the specified filters."
        else:
            summary = f"Analysis complete for {target_variable}."
        if len(feature_importance) >= 3:
            summary += f" The top 3 factors influencing {target_variable} are {feature_importance[0]['feature']}, "
            summary += f"{feature_importance[1]['feature']}, and {feature_importance[2]['feature']}."
        shap_values_dict = {}
        for i, feature in enumerate(model_features):
            shap_values_dict[feature] = shap_values[:, i].tolist()[:10]
        model_version = version_tracker.get_latest_model()
        dataset_version = version_tracker.get_latest_dataset()
        version_info = {}
        if model_version:
            version_info["model_version"] = model_version[0]
        if dataset_version:
            version_info["dataset_version"] = dataset_version[0]
        return {
            "success": True,
            "results": results,
            "summary": summary,
            "feature_importance": feature_importance,
            "shap_values": shap_values_dict,
            "version_info": version_info
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"[ANALYSIS JOB ERROR] Exception: {e}\nTraceback:\n{tb}")
        return {"success": False, "error": str(e), "traceback": tb}



# --- ASYNC /analyze ENDPOINT FOR RENDER ---
@cross_origin(origins="*", methods=['POST', 'OPTIONS'], headers=['Content-Type', 'X-API-KEY'], supports_credentials=True)
@app.route('/analyze', methods=['POST'])
@require_api_key
def analyze():
    logger.info("/analyze endpoint called (ASYNC POST)")
    query = request.json
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
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
    try:
        job = queue.fetch_job(job_id)
        if job is None:
            return jsonify({"success": False, "error": "Job not found"}), 404
        if job.is_finished:
            result = job.result
            if result is None:
                return jsonify({"success": False, "error": "Job finished but no result found."}), 500
            return jsonify({"success": True, "status": "finished", "result": result})
        elif job.is_failed:
            return jsonify({"success": False, "status": "failed", "error": str(job.exc_info)})
        else:
            return jsonify({"success": True, "status": job.get_status()})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"[JOB STATUS ERROR] Exception: {e}\nTraceback:\n{tb}")
        return jsonify({"success": False, "error": str(e), "traceback": tb}), 500


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
    global model, dataset, feature_names
    try:
        if model is None or dataset is None or feature_names is None:
            logger.info("Lazy-loading model and dataset...")
            model_, dataset_, feature_names_ = load_model()
            if model_ is None:
                raise RuntimeError("Model failed to load (model_ is None)")
            if dataset_ is None:
                raise RuntimeError("Dataset failed to load (dataset_ is None)")
            if not feature_names_:
                raise RuntimeError("Feature names failed to load (feature_names_ is empty or None)")
            model = model_
            dataset = dataset_
            feature_names = feature_names_
            logger.info(f"Model and dataset loaded successfully (lazy): model={type(model)}, dataset_shape={dataset.shape}, features={len(feature_names)}")
            # Immediately set job_store status to 'running' if called from a job thread
            # (This is a no-op here, but you can add more logic if needed)
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
            
        # Try to load model
        if os.path.exists(MODEL_PATH):
            model = pickle.load(open(MODEL_PATH, 'rb'))
            if os.path.exists(FEATURE_NAMES_PATH):
                with open(FEATURE_NAMES_PATH, 'r') as f:
                    feature_names = [line.strip() for line in f.readlines()]
            else:
                feature_names = []
                
            return model, None, feature_names  # Return None for dataset as it will be created per query
        else:
            raise FileNotFoundError("Model not found, please train the model first.")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Routes
@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "message": "SHAP/XGBoost Analytics API",
        "endpoints": {
            "/analyze": "POST - Run analysis with structured query",
            "/health": "GET - Check system health",
            "/metadata": "GET - Get dataset metadata and statistics",
            "/versions": "GET - List all tracked versions of datasets and models"
        }
    })

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
    ensure_model_loaded()
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
    return jsonify({
        "status": "healthy",
        "model": {
            "type": "xgboost",
            "version": xgboost.__version__,
            "feature_count": len(feature_names) if feature_names else 0,
            "features": feature_names,
            "version_info": model_version_info
        },
        "dataset": {
            "shape": f"{dataset.shape[0]} rows, {dataset.shape[1]} columns" if dataset is not None else None,
            "columns": list(dataset.columns) if dataset is not None else None,
            "version_info": dataset_version_info
        },
        "redis_connected": redis_status["connected"],
        "redis_status": redis_status,
        "system_info": {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "memory_usage_mb": memory_usage
        },
        "shap_version": shap.__version__
    })



@app.route('/metadata', methods=['GET'])
@require_api_key
def get_metadata():
    ensure_model_loaded()
    try:
        if dataset is None:
            raise APIError("Dataset not available", 500)
        summary_stats = {}
        for column in dataset.columns:
            if column in ['zip_code']:
                continue
            if np.issubdtype(dataset[column].dtype, np.number):
                column_stats = {
                    "mean": float(dataset[column].mean()),
                    "median": float(dataset[column].median()),
                    "min": float(dataset[column].min()),
                    "max": float(dataset[column].max()),
                    "std": float(dataset[column].std())
                }
                summary_stats[column] = column_stats
        if DEFAULT_TARGET in dataset.columns:
            correlations = {}
            for column in dataset.columns:
                if column != DEFAULT_TARGET and np.issubdtype(dataset[column].dtype, np.number):
                    correlations[column] = float(dataset[column].corr(dataset[DEFAULT_TARGET]))
        else:
            correlations = None
        dataset_version = version_tracker.get_latest_dataset()
        if dataset_version:
            dataset_version_id, dataset_info = dataset_version
            version_info = {
                "id": dataset_version_id,
                "created_at": dataset_info.get("timestamp"),
                "description": dataset_info.get("description"),
                "source": dataset_info.get("source")
            }
        else:
            version_info = None
        return jsonify({
            "success": True,
            "columns": list(dataset.columns),
            "record_count": len(dataset),
            "statistics": summary_stats,
            "correlations_with_target": correlations,
            "version_info": version_info
        })
    except Exception as e:
        logger.error(f"Error getting metadata: {str(e)}")
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
                'workers': []
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}...")
    try:
        # Apply SHAP memory optimization fix if available
        try:
            apply_memory_patches(app)
            logger.info("✅ Applied SHAP memory optimization patches")
        except NameError:
            logger.warning("⚠️ SHAP memory optimization not available")
        except Exception as e:
            logger.error(f"❌ Error applying memory patches: {str(e)}")
        
        # Apply Redis connection patches
        apply_all_patches(app)
        logger.info("✅ Applied Redis connection patches")
        
        # Apply worker process fixes
        apply_all_worker_patches(app)
        logger.info("✅ Applied worker process fixes")
        
        app.run(host='0.0.0.0', port=port)
    except Exception as startup_error:
        logger.error(f"Error starting application: {str(startup_error)}")
        sys.exit(1)