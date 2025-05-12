from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import xgboost as xgb
# Import the real SHAP implementation
import shap
import traceback
import logging
import os
import sys
import platform
from functools import wraps

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("shap-microservice")
import os
import json
import pickle
from dotenv import load_dotenv
from data_versioning import DataVersionTracker

# Load environment variables
load_dotenv()

# Configuration from environment variables
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

# Authentication settings
API_KEY = os.getenv('API_KEY')
REQUIRE_AUTH = os.getenv('REQUIRE_AUTH', 'false').lower() == 'true' or API_KEY is not None

# Set up logging level
numeric_level = getattr(logging, LOG_LEVEL.upper(), None)
if not isinstance(numeric_level, int):
    numeric_level = getattr(logging, 'INFO')
logger.setLevel(numeric_level)

# Configure the app to use the correct target variable for Nesto data
if os.path.exists('data/nesto_merge_0.csv'):
    logger.info("Nesto mortgage data detected. Using Mortgage_Approvals as target.")
    DEFAULT_TARGET = os.getenv('DEFAULT_TARGET', 'Mortgage_Approvals')

# Flask app setup
app = Flask(__name__)
if ENABLE_CORS:
    CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})

# Authentication decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not REQUIRE_AUTH:
            return f(*args, **kwargs)
            
        api_key = request.headers.get('X-API-KEY')
        if not api_key or api_key != API_KEY:
            logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
            return jsonify({"success": False, "error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

# Error handler for API exceptions
class APIError(Exception):
    """Custom exception for API errors with status code and message."""
    def __init__(self, message, status_code=400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

@app.errorhandler(APIError)
def handle_api_error(error):
    """Return JSON response for API errors."""
    response = jsonify({"success": False, "error": error.message})
    response.status_code = error.status_code
    return response

@app.errorhandler(Exception)
def handle_generic_exception(error):
    """Handle any unhandled exception."""
    logger.error(f"Unhandled exception: {str(error)}")
    logger.error(traceback.format_exc())
    response = jsonify({
        "success": False, 
        "error": "An internal server error occurred. Please try again later."
    })
    response.status_code = 500
    return response

# Initialize version tracker
version_tracker = DataVersionTracker()

# Load the trained model or create a fallback
def load_model():
    try:
        logger.info("Loading model and dataset...")
        
        # Try to run the setup script to ensure all files are present
        try:
            import subprocess
            subprocess.run([sys.executable, 'setup_for_render.py'], 
                           capture_output=True, check=False)
            logger.info("Setup script completed")
        except Exception as setup_err:
            logger.warning(f"Setup script could not run: {setup_err}")
        
        # Try to load the saved model using environment variable path
        if os.path.exists(MODEL_PATH):
            logger.info(f"Loading trained model from {MODEL_PATH}...")
            model = pickle.load(open(MODEL_PATH, 'rb'))
            
            # Load feature names
            if os.path.exists(FEATURE_NAMES_PATH):
                logger.info(f"Loading feature names from {FEATURE_NAMES_PATH}...")
                with open(FEATURE_NAMES_PATH, 'r') as f:
                    feature_names = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(feature_names)} features")
            else:
                logger.warning(f"Feature names file not found at {FEATURE_NAMES_PATH}")
                feature_names = []
            
            # Load dataset
            cleaned_data_path = 'data/cleaned_data.csv'
            if os.path.exists(cleaned_data_path):
                logger.info(f"Loading dataset from {cleaned_data_path}...")
                dataset = pd.read_csv(cleaned_data_path)
                logger.info(f"Loaded dataset with {dataset.shape[0]} records and {dataset.shape[1]} columns")
            elif os.path.exists(DATASET_PATH):
                logger.info(f"Loading dataset from {DATASET_PATH}...")
                dataset = pd.read_csv(DATASET_PATH)
                logger.info(f"Loaded dataset with {dataset.shape[0]} records and {dataset.shape[1]} columns")
            else:
                error_msg = f"Dataset not found at {DATASET_PATH} or {cleaned_data_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # Get model and dataset version information
            model_version = version_tracker.get_latest_model()
            dataset_version = version_tracker.get_latest_dataset()
            
            # If model is not registered yet, register it now
            if model_version is None and os.path.exists(MODEL_PATH):
                logger.info("Model not registered in version tracker. Registering now...")
                if dataset_version:
                    dataset_version_id = dataset_version[0]
                else:
                    # Register dataset if not already registered
                    dataset_version_id = version_tracker.register_dataset(
                        DATASET_PATH, 
                        description="Dataset loaded on startup", 
                        source="Unknown (loaded on startup)"
                    )
                
                model_version_id = version_tracker.register_model(
                    MODEL_PATH, 
                    dataset_version_id,
                    feature_names_path=FEATURE_NAMES_PATH
                )
                logger.info(f"Registered model with ID: {model_version_id}")
            
            return model, dataset, feature_names
        else:
            raise FileNotFoundError("Model not found, please train the model first")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

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

@app.route('/versions', methods=['GET'])
@require_api_key
def list_versions():
    """Return all tracked versions of datasets and models."""
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
    import xgboost  # Import locally for version information
    
    # Get memory usage
    try:
        import psutil  # Import inside function to avoid issues if library is missing
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    except ImportError:
        # Handle case when psutil is not installed
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
def analyze():
    try:
        # Get the query from the request
        query = request.json
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Log the received query
        print(f"Received query: {query}")
        
        # Extract query parameters using environment defaults
        analysis_type = query.get('analysis_type', DEFAULT_ANALYSIS_TYPE)
        target_variable = query.get('target_variable', DEFAULT_TARGET)
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
                    result[col.lower()] = float(row[col])
            
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

# Load model and dataset when the app starts
logger.info("Loading model and dataset...")
model, dataset, feature_names = load_model()
logger.info("Model and dataset loaded successfully")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)