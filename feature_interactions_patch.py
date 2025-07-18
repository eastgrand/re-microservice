
# Feature interactions memory optimization patch

# In the /feature-interactions endpoint, replace:
# X_sample = X_test.sample(n=sample_size, random_state=42)
# interaction_values = explainer.shap_interaction_values(X_sample)

# With:
from memory_utils import memory_safe_sample_selection, batch_shap_calculation, get_endpoint_config

config = get_endpoint_config('/feature-interactions')

# Use intelligent sampling for feature interactions
sampled_df_full = memory_safe_sample_selection(
    df, target_field,
    config['max_samples'],
    'random'  # Random sampling for interactions
)

# Prepare features from sampled data
X_sampled = sampled_df_full[top_features]
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y, test_size=0.2, random_state=42)

# Use batch processing for interaction calculation
interaction_values = batch_shap_calculation(
    explainer, X_test,
    batch_size=config['batch_size'] // 2,  # Interactions are more memory intensive
    max_memory_mb=config['memory_limit_mb']
)
