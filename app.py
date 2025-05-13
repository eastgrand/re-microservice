
# --- REWRITE STARTS HERE ---
import os
import sys
import logging
import traceback
import threading
import gc
import pickle
import platform
import shutil
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
from dotenv import load_dotenv
from data_versioning import DataVersionTracker

# Load environment variables
load_dotenv()

# Configuration
PORT = int(os.getenv('PORT', 5000))
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
MODEL_PATH = os.getenv('MODEL_PATH', 'models/xgboost_model.pkl')
FEATURE_NAMES_PATH = os.getenv('FEATURE_NAMES_PATH', 'models/feature_names.txt')
DATASET_PATH = os.getenv('DATASET_PATH', 'data/nesto_merge_0.csv')
ENABLE_CORS = os.getenv('ENABLE_CORS', 'true').lower() == 'true'
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
MAX_RESULTS = int(os.getenv('MAX_RESULTS', 100))
DEFAULT_ANALYSIS_TYPE = os.getenv('DEFAULT_ANALYSIS_TYPE', 'ranking')
DEFAULT_TARGET = os.getenv('DEFAULT_TARGET', 'Mortgage_Approvals')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("shap-microservice")
numeric_level = getattr(logging, LOG_LEVEL.upper(), None)
if not isinstance(numeric_level, int):
    numeric_level = getattr(logging, 'INFO')
logger.setLevel(numeric_level)

# Flask app
app = Flask(__name__)
if ENABLE_CORS:
    CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})

# Authentication
API_KEY = os.getenv('API_KEY')
REQUIRE_AUTH = os.getenv('REQUIRE_AUTH', 'false').lower() == 'true' or API_KEY is not None

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not REQUIRE_AUTH:
            return f(*args, **kwargs)
        api_key = request.headers.get('X-API-KEY')
        logger.info(f"[DEBUG] Header X-API-KEY: {repr(api_key)}, Config API_KEY: {repr(API_KEY)}")
        if not api_key or api_key != API_KEY:
            logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
            return jsonify({"success": False, "error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

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
    if model is None or dataset is None or feature_names is None:
        logger.info("Lazy-loading model and dataset...")
        model_, dataset_, feature_names_ = load_model()
        model = model_
        dataset = dataset_
        feature_names = feature_names_
        logger.info("Model and dataset loaded successfully (lazy)")

def load_model():
    try:
        logger.info("Loading model and dataset...")
        import xgboost as xgb
        from train_model import DATA_PATHS
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
            cleaned_data_path = 'data/cleaned_data.csv'
            if os.path.exists(cleaned_data_path):
                dataset = pd.read_csv(cleaned_data_path, nrows=10000 if is_render else 20000)
            elif os.path.exists(DATASET_PATH):
                dataset = pd.read_csv(DATASET_PATH, nrows=10000 if is_render else 20000)
            else:
                dataset = pd.DataFrame()
            return model, dataset, feature_names
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
    import sys
    return jsonify({
        "status": "OK",
        "message": "SHAP microservice is responding",
        "python_version": sys.version,
        "is_render": 'RENDER' in os.environ,
        "environment": {k: v for k, v in os.environ.items() if not k.startswith('_') and k.isupper()}
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
    import xgboost
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        memory_usage = "psutil not installed"
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
        "system_info": {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "memory_usage_mb": memory_usage
        },
        "shap_version": shap.__version__
    })

@app.route('/analyze', methods=['POST'])
@require_api_key
@timeout_handler(timeout=25)
def analyze():
    ensure_model_loaded()
    try:
        query = request.json
        if not query:
            return jsonify({"error": "No query provided"}), 400
        analysis_type = query.get('analysis_type', DEFAULT_ANALYSIS_TYPE)
        target_variable = query.get('target_variable', query.get('target', DEFAULT_TARGET))
        filters = query.get('demographic_filters', [])
        filtered_data = dataset.copy()
        for filter_item in filters:
            if isinstance(filter_item, str) and '>' in filter_item:
                feature, value = filter_item.split('>')
                feature = feature.strip()
                value = float(value.strip())
                filtered_data = filtered_data[filtered_data[feature] > value]
            elif isinstance(filter_item, str) and '<' in filter_item:
                feature, value = filter_item.split('<')
                feature = feature.strip()
                value = float(value.strip())
                filtered_data = filtered_data[filtered_data[feature] < value]
            elif isinstance(filter_item, str):
                if 'high' in filter_item.lower():
                    feature = filter_item.lower().replace('high', '').strip()
                    feature = ''.join([w.capitalize() for w in feature.split(' ')])
                    if feature in filtered_data.columns:
                        threshold = filtered_data[feature].quantile(0.75)
                        filtered_data = filtered_data[filtered_data[feature] > threshold]
        if 'top' in query.get('output_format', '').lower():
            try:
                count = int(''.join(filter(str.isdigit, query.get('output_format', 'top_10').lower())))
                if count == 0:
                    count = 10
            except:
                count = 10
            top_data = filtered_data.sort_values(by=target_variable, ascending=False).head(count)
        else:
            top_data = filtered_data.sort_values(by=target_variable, ascending=False).head(10)
        X = filtered_data.copy()
        for col in ['zip_code', 'latitude', 'longitude']:
            if col in X.columns:
                X = X.drop(col, axis=1)
        if target_variable in X.columns:
            X = X.drop(target_variable, axis=1)
        model_features = feature_names
        X_cols = list(X.columns)
        for col in X_cols:
            if col not in model_features:
                X = X.drop(col, axis=1)
        for feature in model_features:
            if feature not in X.columns:
                X[feature] = 0
        X = X[model_features]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        feature_importance = []
        for i, feature in enumerate(model_features):
            importance = abs(shap_values.values[:, i]).mean()
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
            shap_values_dict[feature] = shap_values.values[:, i].tolist()[:10]
        model_version = version_tracker.get_latest_model()
        dataset_version = version_tracker.get_latest_dataset()
        version_info = {}
        if model_version:
            version_info["model_version"] = model_version[0]
        if dataset_version:
            version_info["dataset_version"] = dataset_version[0]
        return jsonify({
            "success": True,
            "results": results,
            "summary": summary,
            "feature_importance": feature_importance,
            "shap_values": shap_values_dict,
            "version_info": version_info
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/metadata', methods=['GET'])
@require_api_key
def get_metadata():
    ensure_model_loaded()
    try:
        if dataset is None:
            raise Exception("Dataset not available")
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}...")
    try:
        app.run(host='0.0.0.0', port=port, threaded=True)
        logger.info(f"Flask app running on port {port}")
    except Exception as e:
        logger.error(f"Failed to start Flask app: {str(e)}")
        sys.exit(1)

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

# Add this after the root route but before loading any models
@app.route('/ping', methods=['GET'])
def ping():
    """Simple endpoint to test connectivity without loading models"""
    import sys
    return jsonify({
        "status": "OK", 
        "message": "SHAP microservice is responding",
        "python_version": sys.version,
        "is_render": 'RENDER' in os.environ,
        "environment": {k: v for k, v in os.environ.items() if not k.startswith('_') and k.isupper()}
    })

@app.route('/versions', methods=['GET'])
@require_api_key
def list_versions():
    """Return all tracked versions of datasets and models."""
    ensure_model_loaded()
    try:
        versions = version_tracker.list_all_versions()
        
        # Simplify the response structure
        simplified_versions = {
            "datasets": {},
            "models": {}
        }
        
        # Process dataset versions
        for version_id, info in versions.get("datasets", {}).items():
            simplified_versions["datasets"][version_id] = {
                "timestamp": info.get("timestamp"),
                "description": info.get("description"),
                "source": info.get("source"),
                "row_count": info.get("row_count"),
                "column_count": info.get("column_count"),
                "columns": info.get("columns")
            }
            
        # Process model versions
        for version_id, info in versions.get("models", {}).items():
            simplified_versions["models"][version_id] = {
                "timestamp": info.get("timestamp"),
                "dataset_version_id": info.get("dataset_version_id"),
                "metrics": info.get("metrics", {}),
                "feature_names": info.get("feature_names")
            }
            
        return jsonify({
            "success": True,
            "versions": simplified_versions
        })
    except Exception as e:
        logger.error(f"Error listing versions: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/health', methods=['GET'])
@require_api_key
def health_check():
    ensure_model_loaded()
    import xgboost  # Import locally for version information
    # Get memory usage
    try:
        import psutil  # Import inside function to avoid issues if library is missing
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    except ImportError:
        logger.warning("psutil not installed. Memory usage information not available.")
        memory_usage = "psutil not installed"
    # Get version information
    model_version = version_tracker.get_latest_model()
    dataset_version = version_tracker.get_latest_dataset()
    # Format version info
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
        "system_info": {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "memory_usage_mb": memory_usage
        },
        "shap_version": shap.__version__
    })

@app.route('/analyze', methods=['POST'])
@require_api_key
@timeout_handler(timeout=25)  # Set timeout to 25 seconds (Render has 30s limit)
def analyze():
    ensure_model_loaded()
    try:
        # Get the query from the request
        query = request.json
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Log the received query
        print(f"Received query: {query}")
        
        # Extract query parameters using environment defaults
        analysis_type = query.get('analysis_type', DEFAULT_ANALYSIS_TYPE)
        target_variable = query.get('target_variable', query.get('target', DEFAULT_TARGET))
        filters = query.get('demographic_filters', [])
        
        print(f"Analysis type: {analysis_type}")
        print(f"Target variable: {target_variable}")
        print(f"Filters: {filters}")
        
        # Apply filters to dataset
        filtered_data = dataset.copy()
        print(f"Starting with {len(filtered_data)} records")
        
        for filter_item in filters:
            if isinstance(filter_item, str) and '>' in filter_item:
                # Handle filters like "Income > 75000"
                feature, value = filter_item.split('>')
                feature = feature.strip()
                value = float(value.strip())
                filtered_data = filtered_data[filtered_data[feature] > value]
                print(f"Applied filter {feature} > {value}: {len(filtered_data)} records remaining")
            elif isinstance(filter_item, str) and '<' in filter_item:
                # Handle filters like "Age < 30"
                feature, value = filter_item.split('<')
                feature = feature.strip()
                value = float(value.strip())
                filtered_data = filtered_data[filtered_data[feature] < value]
                print(f"Applied filter {feature} < {value}: {len(filtered_data)} records remaining")
            elif isinstance(filter_item, str):
                # Handle filters referencing high values of a variable
                # e.g., "High Hispanic population"
                if 'high' in filter_item.lower():
                    feature = filter_item.lower().replace('high', '').strip()
                    feature = ''.join([w.capitalize() for w in feature.split(' ')])
                    if feature in filtered_data.columns:
                        # Filter to top 25%
                        threshold = filtered_data[feature].quantile(0.75)
                        filtered_data = filtered_data[filtered_data[feature] > threshold]
                        print(f"Applied filter high {feature} > {threshold}: {len(filtered_data)} records remaining")
        
        # Get top results for the target variable
        if 'top' in query.get('output_format', '').lower():
            try:
                count = int(''.join(filter(str.isdigit, query.get('output_format', 'top_10').lower())))
                if count == 0:
                    count = 10
            except:
                count = 10
            print(f"Getting top {count} results")
            top_data = filtered_data.sort_values(by=target_variable, ascending=False).head(count)
        else:
            print("Getting top 10 results by default")
            top_data = filtered_data.sort_values(by=target_variable, ascending=False).head(10)
        
        print(f"Selected {len(top_data)} top records")
        
        # Calculate SHAP values for the filtered dataset
        X = filtered_data.copy()
        for col in ['zip_code', 'latitude', 'longitude']:
            if col in X.columns:
                X = X.drop(col, axis=1)
                
        # Handle target variable
        if target_variable in X.columns:
            X = X.drop(target_variable, axis=1)
        
        # Only use columns that the model knows about
        model_features = feature_names
        X_cols = list(X.columns)
        for col in X_cols:
            if col not in model_features:
                X = X.drop(col, axis=1)
                print(f"Dropped unknown column: {col}")
        
        # If needed, add missing columns that the model expects
        for feature in model_features:
            if feature not in X.columns:
                X[feature] = 0  # Default value
                print(f"Added missing feature with default 0: {feature}")
        
        # Ensure column order matches the model's expected order
        X = X[model_features]
        
        print(f"Calculating SHAP values for {len(X)} records")
        
        # Calculate SHAP values using the real implementation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        
        # Convert SHAP values to a JSON-serializable format
        feature_importance = []
        for i, feature in enumerate(model_features):
            importance = abs(shap_values.values[:, i]).mean()
            feature_importance.append({
                'feature': feature,
                'importance': float(importance)
            })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        print("Generated feature importance")
        
        # Generate results
        results = []
        for idx, row in top_data.iterrows():
            result = {}
            
            # If we have a zip_code column, use it
            if 'zip_code' in row:
                result['zip_code'] = str(row['zip_code'])
            
            # Add geographic coordinates (either from data or generated)
            if 'latitude' in row and 'longitude' in row:
                result['latitude'] = float(row['latitude'])
                result['longitude'] = float(row['longitude'])
            
            # Add the target variable
            target_var_lower = target_variable.lower()
            if target_variable in row:
                result[target_var_lower] = float(row[target_variable])
                
            # Add other columns
            for col in row.index:
                if col not in ['zip_code', 'latitude', 'longitude', target_variable]:
                    try:
                        result[col.lower()] = float(row[col])
                    except (ValueError, TypeError):
                        # Handle non-numeric values
                        if isinstance(row[col], str):
                            result[col.lower()] = row[col]
                        else:
                            result[col.lower()] = str(row[col])
            
            results.append(result)
        
        print(f"Generated {len(results)} result items")
        
        # Create a summary based on analysis type
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
            
        # Add additional insights based on SHAP values
        if len(feature_importance) >= 3:
            summary += f" The top 3 factors influencing {target_variable} are {feature_importance[0]['feature']}, "
            summary += f"{feature_importance[1]['feature']}, and {feature_importance[2]['feature']}."
        
        print(f"Generated summary: {summary}")
        
        # Prepare SHAP values for output
        # Convert numpy arrays to lists for JSON serialization
        shap_values_dict = {}
        for i, feature in enumerate(model_features):
            shap_values_dict[feature] = shap_values.values[:, i].tolist()[:10]  # Limit to first 10 values
        
        # Get version information
        model_version = version_tracker.get_latest_model()
        dataset_version = version_tracker.get_latest_dataset()
        
        # Prepare version info for response
        version_info = {}
        if model_version:
            version_info["model_version"] = model_version[0]
        if dataset_version:
            version_info["dataset_version"] = dataset_version[0]
        
        return jsonify({
            "success": True,
            "results": results,
            "summary": summary,
            "feature_importance": feature_importance,
            "shap_values": shap_values_dict,
            "version_info": version_info
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during analysis: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# Add new endpoints for dataset metadata exploration
@app.route('/metadata', methods=['GET'])
@require_api_key
def get_metadata():
    """Return metadata about the dataset."""
    ensure_model_loaded()
    try:
        if dataset is None:
            raise APIError("Dataset not available", 500)
        
        # Extract basic statistics
        summary_stats = {}
        for column in dataset.columns:
            if column in ['zip_code']:  # Skip non-numeric columns
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
        
        # Get correlation with target variable
        if DEFAULT_TARGET in dataset.columns:
            correlations = {}
            for column in dataset.columns:
                if column != DEFAULT_TARGET and np.issubdtype(dataset[column].dtype, np.number):
                    correlations[column] = float(dataset[column].corr(dataset[DEFAULT_TARGET]))
        else:
            correlations = None
        
        # Get version information for the current dataset
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



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask app on port {port}...")
    try:
        # Use threaded=True to improve responsiveness to health checks
        app.run(host='0.0.0.0', port=port, threaded=True)
        logger.info(f"Flask app running on port {port}")
    except Exception as e:
        logger.error(f"Failed to start Flask app: {str(e)}")
        sys.exit(1)