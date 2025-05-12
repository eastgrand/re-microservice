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
