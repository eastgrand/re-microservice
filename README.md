# SHAP/XGBoost Microservice

This microservice powers advanced analytics features using SHAP (SHapley Additive exPlanations) and XGBoost. It provides an API for analyzing sales data and demographic information with machine learning.

## Features

- XGBoost-based predictive modeling
- SHAP-based feature importance analysis
- Demographic filtering and analysis
- JSON API with structured query support
- Data and model versioning for reproducibility
- Authentication with API keys
- Cross-validation model training
- Detailed logging throughout the application

## Implementation Notes

This implementation uses the full SHAP library for calculating feature importance. We've applied multiple patches to the SHAP library to ensure compatibility with Python 3.13 and newer NumPy versions:

1. Fixed import handling for optional dependencies (pyspark, catboost)
2. Replaced deprecated NumPy APIs (bool8 → bool_, obj2sctype → dtype().type)
3. Added type checking exclusions to prevent false positive errors

For detailed information about these patches, see [SHAP-PATCHES.md](./SHAP-PATCHES.md).

These patches ensure we get accurate SHAP values that properly account for feature interactions and individual data point effects without runtime errors.

### Memory Optimization

This service includes memory optimization features to ensure it runs efficiently within Render's 512MB memory limit:

1. **Data Optimization**: Reduces memory footprint by optimizing DataFrame data types
2. **Adaptive Training**: Adjusts model complexity based on available memory
3. **Dynamic Sampling**: Reduces dataset size when memory usage becomes critical
4. **Memory Monitoring**: Logs memory usage at key points in execution

For detailed information about memory optimizations, see [MEMORY-OPTIMIZATIONS.md](./MEMORY-OPTIMIZATIONS.md).

Memory usage thresholds control different optimization strategies:
- 450MB+: Critical - Activates all optimizations
- 400MB+: High - Skips cross-validation and reduces data size
- 350MB+: Moderate - Reduces model complexity

## API Endpoints

### Health Check

```http
GET /health
```

Returns the status of the service, model information, dataset statistics, and version information.

### Root

```http
GET /
```

Returns information about available endpoints.

### Analysis

```http
POST /analyze
```

Accepts a structured query for data analysis:

```json
{
  "analysis_type": "correlation",
  "target_variable": "Nike_Sales",
  "demographic_filters": ["Income > 50000"],
  "output_format": "top_10_zip_codes"
}
```

### Metadata

```http
GET /metadata
```

Returns detailed metadata about the dataset including statistics, correlations, and version information.

### Versions

```http
GET /versions
```

Lists all tracked versions of datasets and models in the system.

Parameters:

- `analysis_type`: Type of analysis to perform (correlation, ranking)
- `target_variable`: The variable to analyze (e.g., "Nike_Sales")
- `demographic_filters`: List of filter conditions (e.g., ["Income > 50000", "Age < 30"])
- `output_format`: Format preference for results (e.g., "top_10_zip_codes")

Response:

```json
{
  "success": true,
  "results": [...],
  "summary": "Analysis shows a strong correlation between Nike_Sales and African_American_Population.",
  "feature_importance": [...],
  "shap_values": {...}
}
```

## Deployment

The service is configured for deployment on Render.com using the provided `render.yaml` file.

## Authentication

The API is protected using API keys. When authentication is enabled, you must include an `X-API-KEY` header with your requests:

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your_api_key" \
  -d '{"analysis_type":"correlation","target_variable":"Nike_Sales"}'
```

To configure authentication:

1. Copy `.env.example` to `.env`
2. Set `REQUIRE_AUTH=true`
3. Set `API_KEY` to a strong, unique value

## Data Versioning

The microservice includes a data versioning system that tracks:

- Dataset versions with metadata and statistics
- Model versions with performance metrics
- Relationships between models and the datasets they were trained on

This ensures reproducibility of results and helps track model performance over time. Version information is included in API responses.

## Local Development

To run the service locally:

1. Install dependencies: `pip install -r requirements.txt`
2. Apply SHAP patch: `python patch_shap.py`
3. Train the model: `python train_model.py`
4. Start the server: `python app.py`
5. Test locally:

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your_api_key" \
  -d '{"analysis_type":"correlation","target_variable":"Nike_Sales","demographic_filters":["Income > 50000"]}'
```

You can also use the provided test script: `bash test_api.sh`

## Adding and Deploying a New Dataset

This section provides step-by-step instructions for adding a new dataset to the microservice and deploying it properly.

### 1. Prepare Your Dataset

- **Format Requirements**:
  - CSV format with headers
  - Include the necessary geographic identifiers (`zip_code` or `Forward Sortation Area`)
  - Include a target variable (what you want to predict/analyze)
  - Include demographic and other predictive features
  - Percentage fields should be marked with a `(%)` suffix in the column name

- **Placement**:
  - Save your dataset as `data/nesto_merge_0.csv` (for Nesto data) or
  - Save a mapped dataset directly as `data/cleaned_data.csv`

### 2. Update Field Mappings

If your dataset uses custom field names that need to be mapped to standardized names:

- **Edit `map_nesto_data.py`**:

```bash
nano map_nesto_data.py
```

- **Update the `FIELD_MAPPINGS` dictionary**:
  - Add entries for each field in your dataset
  - Use the format: `'Original Column Name': 'Standardized_Name'`
  - For percentage fields, use the `_Pct` suffix in the standardized name

- **Update the `TARGET_VARIABLE`**:
  - Set this to the column name you want to predict/analyze
  - Example: `TARGET_VARIABLE = 'Mortgage_Approvals'`

- **Document Your Mappings**:
  - Update `data/NESTO_FIELD_MAPPING.md` with your new fields
  - Include descriptions for each field

### 3. Test Your Dataset Locally

Before deploying, verify your dataset works correctly:

- **Run the setup script**:

```bash
python setup_for_render.py
```

- **Train the model with your data**:

```bash
python train_model.py
```

- **Verify the model was created**:

```bash
ls -la models/
```

- **Run the application locally**:

```bash
python app.py
```

- **Test the API with your new target variable**:

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: your_api_key" \
  -d '{"analysis_type":"correlation","target_variable":"YOUR_TARGET_VARIABLE"}'
```

### 4. Prepare for Deployment

- **Run the pre-deployment verification script**:

```bash
bash train_before_deploy.sh
```

- **Commit Changes**:

```bash
git add map_nesto_data.py data/NESTO_FIELD_MAPPING.md
git commit -m "Update field mappings for new dataset"
```

- **Consider Model Size**:
  - If your model file is small (<100MB), you can commit it:

```bash
git add models/xgboost_model.pkl models/feature_names.txt
git commit -m "Add trained model for deployment"
```

  - For larger models, the `render.yaml` will handle model training during deployment

### 5. Deploy to Render

- **Push your changes**:

```bash
git push
```

- **Monitor the build process**:
  - Watch the build logs on Render for any errors
  - If the build fails, check the logs for missing fields or mapping errors

- **Verify Deployment**:
  - Test the deployed API endpoint
  - Check the `/health` endpoint to verify model information

### 6. Troubleshooting Deployment Issues

If you encounter issues during deployment:

- **Check Model Training Logs**:
  - Review the build logs on Render
  - Look for errors in the `python train_model.py` step

- **Verify Field Mappings**:
  - Ensure all required fields are properly mapped
  - Check that the `TARGET_VARIABLE` exists in your dataset

- **Fallback Options**:
  - If deployment fails, the system will attempt to:
    - Use the cleaned data if available
    - Use the raw data with on-the-fly mapping
    - Fall back to sample data as a last resort

- **Manual Model Training**:
  - You can pre-train the model locally and commit it to avoid training during deployment
  - Follow the steps in [MODEL_TRAINING.md](./MODEL_TRAINING.md)

For more detailed information on model training and deployment, refer to [MODEL_TRAINING.md](./MODEL_TRAINING.md).
