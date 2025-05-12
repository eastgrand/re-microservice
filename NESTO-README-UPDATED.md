# Nesto Mortgage Analytics Microservice

This microservice analyzes Canadian mortgage application data from Nesto, focusing on Forward Sortation Area (FSA) insights. The service provides analytics on regional mortgage application trends using XGBoost and SHAP for explainable AI.

## Key Updates

- **Descriptive Field Names**: Now using the clearer field names from `nesto_merge_0.csv` instead of cryptic field names
- **Python 3.13 Compatibility**: Fixed SHAP compatibility issues with Python 3.13
- **Flask/Werkzeug Compatibility**: Addressed dependency conflicts between Flask and Werkzeug
- **Data Processing Pipeline**: Streamlined pipeline that maps descriptive field names during model training
- **Target Variable**: Changed to `Mortgage_Approvals` to reflect the mortgage application focus
- **Memory Optimization**: Implemented memory usage improvements to run efficiently on Render's 512MB limit

## API Usage

The API allows you to:
1. Analyze mortgage trends by geographic region
2. Get SHAP values to explain model predictions
3. Query metadata about the dataset
4. Compare different geographic areas

Example query:
```bash
curl -X POST https://nesto-mortgage-analytics.onrender.com/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your-api-key" \
  -d '{
    "analysis_type": "ranking",
    "target": "Mortgage_Approvals",
    "filters": {
      "Income": "high",
      "Population": "high"
    },
    "output_format": "top_10"
  }'
```

## Deployment

The service is configured for deployment on Render.com using the provided `render.yaml` configuration.

To deploy:
1. Push the code to your GitHub repository
2. Connect your Render.com account to the repository
3. Deploy as a Web Service using the Blueprint

## Local Development

To run the service locally:
```bash
# Install dependencies
pip install -r requirements.txt

# Apply compatibility patches
python fix_flask_werkzeug.py
python patch_shap.py

# Enable memory optimization
export MEMORY_OPTIMIZATION=true
export MAX_MEMORY_MB=450

# Train the model
python train_model.py

# Run the API server
python app.py
```

## Verification

To verify the complete data processing pipeline:
```bash
python verify_nesto_data_pipeline.py
```

This script will check:
1. Data loading and field mapping
2. Model training
3. Prediction capabilities

## Memory Optimization

The service now includes comprehensive memory optimization features to stay below Render's 512MB memory limit:

1. **Data Type Optimization**: Reduces DataFrame memory usage by ~50% by using the smallest possible data types
2. **Adaptive Model Complexity**: Reduces model complexity when memory is constrained
3. **Dynamic Dataset Sampling**: Automatically reduces dataset size when memory usage becomes critical
4. **Memory Usage Monitoring**: Tracks and logs memory usage at key points in execution

To test memory optimizations:
```bash
./test_memory_optimization.sh
```

For detailed information about memory optimizations, see [MEMORY-OPTIMIZATIONS.md](./MEMORY-OPTIMIZATIONS.md).

## Field Mapping

See `data/NESTO_FIELD_MAPPING.md` for the complete mapping between descriptive field names and model feature names.
