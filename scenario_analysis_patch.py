
# Scenario analysis memory optimization patch

# In the /scenario-analysis endpoint, replace:
# scenario_shap = explainer.shap_values([modified_data])
# baseline_shap = explainer.shap_values([base_data])

# With:
from memory_utils import batch_shap_calculation, get_endpoint_config

config = get_endpoint_config('/scenario-analysis')

# Process scenarios in memory-safe batches
scenario_data = pd.DataFrame([modified_data])
baseline_data = pd.DataFrame([base_data])

scenario_shap = batch_shap_calculation(
    explainer, scenario_data,
    batch_size=config['batch_size'],
    max_memory_mb=config['memory_limit_mb']
)

baseline_shap = batch_shap_calculation(
    explainer, baseline_data,
    batch_size=config['batch_size'],
    max_memory_mb=config['memory_limit_mb']
)
