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

# Import field mappings and target variable
from map_nesto_data import FIELD_MAPPINGS, TARGET_VARIABLE

# Import query-aware analysis functions
from query_aware_analysis import enhanced_query_aware_analysis, analyze_query_intent

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
            ssl_cert_reqs=None  # Don't verify SSL certificate
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

def analysis_worker(query):
    import time
    import shap
    import gc
    import psutil
    
    logger.info(f"[RQ WORKER] analysis_worker called with query: {query}")
    model, model_features = ensure_model_loaded()
    
    try:
        # Log initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Load the dataset
        logger.info("Loading dataset...")
        try:
            # Try nesto_merge_0.csv first
            dataset_path = 'data/nesto_merge_0.csv'
            if not os.path.exists(dataset_path):
                dataset_path = 'data/cleaned_data.csv'
            
            dataset = pd.read_csv(dataset_path)
            logger.info(f"Successfully loaded dataset from {dataset_path} with shape: {dataset.shape}")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return {"success": False, "error": f"Failed to load dataset: {str(e)}"}
        
        # Log dataset info
        logger.info(f"Dataset shape: {dataset.shape}")
        logger.info(f"Dataset columns: {list(dataset.columns)}")
        logger.info(f"Dataset memory usage: {dataset.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")
        
        analysis_type = query.get('analysis_type', DEFAULT_ANALYSIS_TYPE)
        target_variable = query.get('target_variable', query.get('target', TARGET_VARIABLE))
        
        # Map target variable to dataset field name
        target_field = None
        for orig_field, mapped_field in FIELD_MAPPINGS.items():
            if mapped_field == target_variable:
                target_field = orig_field
                break
        
        # If no mapping found, try to find the field directly or with common variations
        if not target_field:
            # Check if the target variable exists directly in the dataset
            if target_variable in dataset.columns:
                target_field = target_variable
            # Try common variations for CONVERSIONRATE
            elif target_variable == 'CONVERSIONRATE' and 'CONVERSION_RATE' in dataset.columns:
                target_field = 'CONVERSION_RATE'
            elif target_variable == 'CONVERSION_RATE' and 'CONVERSIONRATE' in dataset.columns:
                target_field = 'CONVERSIONRATE'
            else:
                target_field = target_variable  # Use original if no mapping found
        
        filters = query.get('demographic_filters', [])
        
        logger.info(f"Analysis parameters - type: {analysis_type}, target: {target_field}, filters: {filters}")
        
        # Validate target variable exists
        if target_field not in dataset.columns:
            logger.error(f"Target variable {target_field} not found in dataset. Available columns: {list(dataset.columns)}")
            return {"success": False, "error": f"Target variable {target_field} not found in dataset"}
        
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
        
        # Optimize data preparation
        top_data = filtered_data.sort_values(by=target_field, ascending=False)
        
        # Limit data size for faster processing (extremely conservative for 128+ feature model)
        max_analysis_rows = 25  # Very small sample to test 128-feature SHAP feasibility
        if len(top_data) > max_analysis_rows:
            logger.info(f"Limiting analysis to top {max_analysis_rows} rows (from {len(top_data)} total)")
            top_data = top_data.head(max_analysis_rows)
        
        X = top_data.copy()
        
        # Apply field mappings to match model expectations
        logger.info("Checking if dataset columns match model features directly...")
        
        # Check if dataset columns already match model features (they should!)
        matching_features = [feat for feat in model_features if feat in X.columns]
        logger.info(f"Direct matches between model and dataset: {len(matching_features)} of {len(model_features)}")
        
        if len(matching_features) >= len(model_features) * 0.8:  # If 80%+ features match directly
            logger.info("Dataset columns match model features directly - using without mapping")
            
            # Use ALL model features that are available in the dataset
            available_features = matching_features.copy()
            
            # Preserve additional non-model columns like ID for results processing
            preserved_cols = ['ID', 'OBJECTID', 'Shape__Area', 'Shape__Length']
            for col in preserved_cols:
                if col in X.columns and col not in available_features:
                    available_features.append(col)
            
            X = X[available_features]
            
            # Set model_only_features to be the matched model features (not the preserved columns)
            model_only_features = matching_features
            preserved_only_cols = [col for col in available_features if col not in model_features]
        else:
            # Apply field mappings only if direct match fails
            logger.info("Applying field mappings to match model features...")
            mapped_data = {}
            
            # Apply field mappings from FIELD_MAPPINGS
            for orig_field, mapped_field in FIELD_MAPPINGS.items():
                if orig_field in X.columns:
                    mapped_data[mapped_field] = X[orig_field]
                    logger.info(f"Mapped '{orig_field}' to '{mapped_field}'")
            
            # Create DataFrame with mapped columns
            X_mapped = pd.DataFrame(mapped_data)
            logger.info(f"Created mapped dataset with columns: {list(X_mapped.columns)}")
            
            # Only keep columns that are used by the model
            available_features = [feat for feat in model_features if feat in X_mapped.columns]
            missing_features = [feat for feat in model_features if feat not in X_mapped.columns]
            
            if missing_features:
                logger.warning(f"Missing model features: {missing_features}")
            if not available_features:
                logger.error("No model features available in mapped data")
                return {
                    "success": False,
                    "error": f"No model features found. Model expects: {model_features}, Available: {list(X_mapped.columns)}",
                    "results": [],
                    "summary": "Analysis failed due to feature mismatch."
                }
            
            X = X_mapped[available_features]
            
            # Set model_only_features and preserved_only_cols for the mapping case
            model_only_features = available_features  # For mapped case, available_features are model features
            preserved_only_cols = []  # No preserved columns in mapping case
        
        logger.info(f"Data prepared. Shape: {X.shape}")
        
        logger.info(f"Model features for SHAP: {len(model_only_features)} features")
        logger.info(f"Preserved columns: {preserved_only_cols}")
        
        # Only pass model features to SHAP
        X_for_shap = X[model_only_features] if model_only_features else X
        
        # Replace the direct SHAP computation with pre-calculated SHAP lookup
        logger.info("Loading pre-calculated SHAP values for instant analysis")
        try:
            # Load pre-calculated SHAP values
            if os.path.exists('precalculated/shap_values.pkl.gz'):
                logger.info("Loading pre-calculated SHAP data...")
                precalc_df = pd.read_pickle('precalculated/shap_values.pkl.gz', compression='gzip')
                logger.info(f"Pre-calculated data loaded: {precalc_df.shape}")
                
                # Filter to matching IDs from our analysis data
                analysis_ids = top_data['ID'].values
                precalc_subset = precalc_df[precalc_df['ID'].isin(analysis_ids)]
                
                if len(precalc_subset) == 0:
                    logger.warning("No matching IDs found in pre-calculated data")
                    return {
                        "success": False,
                        "error": "No pre-calculated SHAP values found for the filtered data",
                        "results": [],
                        "summary": "Analysis could not be completed - no matching data in pre-calculated values."
                    }
                
                logger.info(f"Found {len(precalc_subset)} matching rows in pre-calculated data")
                
                # Extract SHAP values for model features
                shap_columns = [col for col in precalc_df.columns if col.startswith('shap_')]
                shap_feature_names = [col.replace('shap_', '') for col in shap_columns]
                
                # Create SHAP values array matching our model features
                matching_shap_features = []
                shap_values_list = []
                
                for feature in model_only_features:
                    shap_col = f'shap_{feature}'
                    if shap_col in precalc_df.columns:  # Check against full dataframe, not subset
                        matching_shap_features.append(feature)
                        shap_values_list.append(precalc_subset[shap_col].values)
                
                if len(matching_shap_features) == 0:
                    logger.error("No matching SHAP features found")
                    logger.error(f"Model features: {model_only_features[:5]}...")
                    logger.error(f"Available SHAP columns: {[col for col in precalc_df.columns if col.startswith('shap_')][:5]}...")
                    return {
                        "success": False,
                        "error": "No matching SHAP features found in pre-calculated data",
                        "results": [],
                        "summary": "SHAP features do not match between model and pre-calculated data."
                    }
                
                # Create SHAP values array (rows x features)
                shap_values = np.column_stack(shap_values_list)
                logger.info(f"SHAP values loaded: {shap_values.shape} for {len(matching_shap_features)} features")
                
            else:
                logger.warning("Pre-calculated SHAP file not found, falling back to on-demand computation")
                # Fallback to original SHAP computation
                X_for_shap = X[model_only_features] if model_only_features else X
                shap_values = create_memory_optimized_explainer(model, X_for_shap)
                matching_shap_features = model_only_features
                
                if shap_values is None:
                    logger.error("SHAP computation failed")
                    return {
                        "success": False,
                        "error": "SHAP computation failed due to timeout or memory constraints",
                        "results": [],
                        "summary": "Analysis could not be completed due to resource constraints."
                    }
            
        except Exception as e:
            logger.error(f"Error loading pre-calculated SHAP values: {str(e)}")
            logger.info("Falling back to on-demand SHAP computation")
            # Fallback to original SHAP computation
            X_for_shap = X[model_only_features] if model_only_features else X
            shap_values = create_memory_optimized_explainer(model, X_for_shap)
            matching_shap_features = model_only_features
            
            if shap_values is None:
                logger.error("SHAP computation failed")
                return {
                    "success": False,
                    "error": "SHAP computation failed due to timeout or memory constraints",
                    "results": [],
                    "summary": "Analysis could not be completed due to resource constraints."
                }
            
        logger.info("SHAP analysis completed successfully (using pre-calculated values)")
        
        feature_importance = []
        for i, feature in enumerate(matching_shap_features):  # Use matching_shap_features for SHAP results
            importance = abs(shap_values[:, i]).mean()
            feature_importance.append({'feature': feature, 'importance': float(importance)})
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        results = []
        for idx, row in top_data.iterrows():
            result = {}
            # Use ID field as zip_code if available
            if 'ID' in row:
                result['zip_code'] = str(row['ID'])
            # Add latitude and longitude if available (from Shape fields)
            if 'Shape__Area' in row and 'Shape__Length' in row:
                # Create proxy coordinates based on shape data
                result['latitude'] = float(row['Shape__Area']) / 1000000 + 45.0  # Rough Canadian latitude
                result['longitude'] = float(row['Shape__Length']) / 1000000 - 75.0  # Rough Canadian longitude
            # Add the target variable
            if target_field in row:
                result[target_field.lower()] = float(row[target_field])
            # Add other key demographic fields
            key_fields = [
                '2024 Total Population', '2024 Household Average Income (Current Year $)',
                '2024 Visible Minority Total Population (%)', 'FREQUENCY', 'SUM_FUNDED'
            ]
            for field in key_fields:
                if field in row:
                    try:
                        result[field.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('$', '')] = float(row[field])
                    except (ValueError, TypeError):
                        result[field.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('$', '')] = str(row[field])
            results.append(result)
        
        # Apply query-aware analysis enhancement
        user_query = query.get('query', '')
        if user_query:
            logger.info(f"Applying query-aware analysis for: {user_query}")
            try:
                # Use the pre-calculated data for enhanced analysis
                enhanced_analysis = enhanced_query_aware_analysis(
                    user_query, precalc_subset, feature_importance, results, target_field
                )
                
                # Use enhanced results
                summary = enhanced_analysis['intent_aware_summary']
                enhanced_feature_importance = enhanced_analysis['enhanced_feature_importance']
                query_intent = enhanced_analysis['query_intent']
                
                logger.info(f"Query intent detected: {query_intent}")
                
            except Exception as e:
                logger.warning(f"Query-aware analysis failed, using standard analysis: {str(e)}")
                # Fallback to standard summary
                if analysis_type == 'correlation':
                    if len(feature_importance) > 0:
                        summary = f"Analysis shows a strong correlation between {target_field} and {feature_importance[0]['feature']}."
                    else:
                        summary = f"Analysis complete for {target_field}, but no clear correlations found."
                elif analysis_type == 'ranking':
                    if len(results) > 0:
                        summary = f"The top area for {target_field} has a value of {results[0][target_field.lower()]:.2f}."
                    else:
                        summary = f"No results found for {target_field} with the specified filters."
                else:
                    summary = f"Analysis complete for {target_field}."
                
                enhanced_feature_importance = feature_importance
                query_intent = None
        else:
            # No query provided, use standard analysis
            if analysis_type == 'correlation':
                if len(feature_importance) > 0:
                    summary = f"Analysis shows a strong correlation between {target_field} and {feature_importance[0]['feature']}."
                else:
                    summary = f"Analysis complete for {target_field}, but no clear correlations found."
            elif analysis_type == 'ranking':
                if len(results) > 0:
                    summary = f"The top area for {target_field} has a value of {results[0][target_field.lower()]:.2f}."
                else:
                    summary = f"No results found for {target_field} with the specified filters."
            else:
                summary = f"Analysis complete for {target_field}."
            
            enhanced_feature_importance = feature_importance
            query_intent = None
        
        # Add top 3 factors to summary if available
        if len(enhanced_feature_importance) >= 3 and 'top 3 factors' not in summary.lower():
            summary += f" The top 3 factors influencing {target_field} are {enhanced_feature_importance[0]['feature']}, "
            summary += f"{enhanced_feature_importance[1]['feature']}, and {enhanced_feature_importance[2]['feature']}."
        
        shap_values_dict = {}
        for i, feature in enumerate(matching_shap_features):  # Use matching_shap_features for SHAP results
            shap_values_dict[feature] = shap_values[:, i].tolist()[:10]
        model_version = version_tracker.get_latest_model()
        dataset_version = version_tracker.get_latest_dataset()
        version_info = {}
        if model_version:
            version_info["model_version"] = model_version[0]
        if dataset_version:
            version_info["dataset_version"] = dataset_version[0]
        
        # Build final response with query-aware enhancements
        response = {
            "success": True,
            "results": results,
            "summary": summary,
            "feature_importance": enhanced_feature_importance,
            "shap_values": shap_values_dict,
            "version_info": version_info
        }
        
        # Add query intent information if available
        if query_intent:
            response["query_analysis"] = {
                "intent": query_intent,
                "analysis_focus": query_intent.get('focus_areas', []),
                "key_concepts": query_intent.get('key_concepts', []),
                "detected_analysis_type": query_intent.get('analysis_type', 'correlation')
            }
        
        return response
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
    
    # Check if Redis/queue is available
    if queue is None:
        logger.error("Redis queue not available - cannot process async jobs")
        return jsonify({
            "success": False, 
            "error": "Analysis service temporarily unavailable. Redis connection failed during startup.",
            "retry_suggestion": "Please try again in a few minutes as the service may be initializing."
        }), 503
    
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
            
        # Try to load model
        if os.path.exists(MODEL_PATH):
            model = pickle.load(open(MODEL_PATH, 'rb'))
            if os.path.exists(FEATURE_NAMES_PATH):
                with open(FEATURE_NAMES_PATH, 'r') as f:
                    feature_names = [line.strip() for line in f.readlines()]
            else:
                feature_names = []
                
            return model, feature_names  # Return None for dataset as it will be created per query
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
            dataset_path = 'data/nesto_merge_0.csv'
            if not os.path.exists(dataset_path):
                dataset_path = 'data/cleaned_data.csv'
            
            if os.path.exists(dataset_path):
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
            dataset_path = 'data/nesto_merge_0.csv'
            if not os.path.exists(dataset_path):
                dataset_path = 'data/cleaned_data.csv'
            
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