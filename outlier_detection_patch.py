
# Outlier detection memory optimization patch

# In the /outlier-detection endpoint, replace:
# shap_values = explainer.shap_values(outlier_data)

# With:
from memory_utils import memory_safe_sample_selection, batch_shap_calculation, get_endpoint_config

# Sample data intelligently for outlier detection
config = get_endpoint_config('/outlier-detection')
sampled_df = memory_safe_sample_selection(
    df, target_field, 
    config['max_samples'], 
    'extremes'  # Focus on outliers
)

# Use batch processing for SHAP
outlier_data = sampled_df[outlier_indices] if len(outlier_indices) > 0 else sampled_df.head(20)
shap_values = batch_shap_calculation(
    explainer, outlier_data,
    batch_size=config['batch_size'],
    max_memory_mb=config['memory_limit_mb']
)
