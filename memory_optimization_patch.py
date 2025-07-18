
# Memory optimization integration patch for app.py
# Add this import at the top of app.py after other imports

from memory_utils import (
    memory_safe_shap_wrapper, 
    get_endpoint_config,
    batch_shap_calculation,
    force_garbage_collection,
    get_memory_usage
)

# Add this function to replace direct SHAP calculations
def calculate_shap_with_memory_optimization(explainer, X, endpoint_path):
    """
    Memory-optimized SHAP calculation wrapper
    Replaces direct explainer.shap_values(X) calls
    """
    config = get_endpoint_config(endpoint_path)
    
    try:
        # Use batch processing for large datasets
        shap_values = batch_shap_calculation(
            explainer, X,
            batch_size=config['batch_size'],
            max_memory_mb=config['memory_limit_mb']
        )
        return shap_values
    except Exception as e:
        logger.error(f"Memory-optimized SHAP calculation failed: {str(e)}")
        # Fallback to smaller batch size
        try:
            smaller_batch = max(10, config['batch_size'] // 2)
            logger.info(f"Retrying with smaller batch size: {smaller_batch}")
            shap_values = batch_shap_calculation(
                explainer, X,
                batch_size=smaller_batch,
                max_memory_mb=config['memory_limit_mb'] // 2
            )
            return shap_values
        except Exception as e2:
            logger.error(f"Fallback SHAP calculation also failed: {str(e2)}")
            raise e2

# Example integration pattern for endpoints:
# Replace this pattern:
#   shap_values = explainer.shap_values(X_sample)
# 
# With this pattern:
#   shap_values = calculate_shap_with_memory_optimization(explainer, X_sample, request.path)

# Memory monitoring endpoint
@app.route('/memory-status', methods=['GET'])
def memory_status():
    """Get current memory usage and limits"""
    try:
        current_memory = get_memory_usage()
        
        return safe_jsonify({
            "current_memory_mb": current_memory,
            "memory_configs": ENDPOINT_MEMORY_CONFIGS,
            "status": "healthy" if current_memory < 800 else "high_memory"
        })
    except Exception as e:
        return safe_jsonify({"error": str(e)}, 500)
