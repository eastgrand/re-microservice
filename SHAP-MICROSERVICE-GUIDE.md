# Step-by-Step Guide: Building & Deploying a SHAP/XGBoost Microservice

This guide provides detailed instructions for creating, training, and deploying the SHAP/XGBoost microservice that powers advanced analytics features in the AI Analytics App.

## Prerequisites

- Python 3.9+
- Git
- GitHub account
- Render account ([https://render.com](https://render.com))
- Basic familiarity with machine learning concepts
- CSV dataset with sales and demographic data

## Step 1: Set Up Project Structure

Create the basic directory structure for the microservice:

```bash
# Create the project directory and subdirectories
mkdir -p shap-microservice/models shap-microservice/data
cd shap-microservice
```

## Step 2: Prepare Your Data

Place your CSV dataset in the data directory:

```bash
# Copy your existing CSV data to the data directory
cp /path/to/your/salesdata.csv shap-microservice/data/sales_data.csv
```

Your CSV should include:

- Geographic identifiers (zip_code or similar)
- Target variables (sales metrics)
- Demographic features (income, age, etc.)
- Geographic coordinates (latitude/longitude)

## Step 3: Create the Training Script

Create a Python script to train and save the XGBoost model:

```bash
# Create the training script file
touch train_model.py
```

Add the following code to `train_model.py`, which uses real CSV data:

```python
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("Starting model training process...")

# Load real data from CSV
data_path = os.path.join('data', 'sales_data.csv')
if os.path.exists(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {df.shape[0]} records and {df.shape[1]} columns")
else:
    raise FileNotFoundError(f"Data file not found at {data_path}. Please add your CSV file to the data directory.")

# Basic data cleaning
print("Cleaning data...")
df = df.dropna(subset=['zip_code', 'Nike_Sales'])  # Adjust column names to match your dataset

# Data validation - ensure all required columns exist
required_columns = ['zip_code', 'Nike_Sales', 'Income', 'Age', 'Hispanic_Population']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Required columns missing from dataset: {missing_columns}")

# Make sure numeric columns are properly typed
numeric_columns = ['Income', 'Age', 'Hispanic_Population', 'Nike_Sales']
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle any additional NAs created by type conversion
df = df.dropna(subset=numeric_columns)

print(f"After cleaning: {df.shape[0]} records and {df.shape[1]} columns")

# Save cleaned data for reference
df.to_csv('data/cleaned_data.csv', index=False)
print("Cleaned data saved")

# Prepare features and target
print("Preparing data for training...")
# Exclude non-feature columns - adjust based on your dataset structure
exclude_cols = ['zip_code', 'Nike_Sales']
if 'latitude' in df.columns:
    exclude_cols.append('latitude')
if 'longitude' in df.columns:
    exclude_cols.append('longitude')

X = df.drop(exclude_cols, axis=1)
y = df['Nike_Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training XGBoost model...")
# Train XGBoost model
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Model evaluation - RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Save model
print("Saving model...")
pickle.dump(model, open('models/xgboost_model.pkl', 'wb'))

# Save feature names
with open('models/feature_names.txt', 'w') as f:
    for feature in X.columns:
        f.write(f"{feature}\n")

print("Done! Model and feature names saved.")
```

## Step 4: Create the Flask API Application

Create the main application file that will serve API requests:

```bash
# Create the app file
touch app.py
```

Add the following code to `app.py`:

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import os
import json
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model or create a fallback
def load_model():
    try:
        # Try to load the saved model
        model_path = os.path.join('models', 'xgboost_model.pkl')
        if os.path.exists(model_path):
            print("Loading trained model...")
            model = pickle.load(open(model_path, 'rb'))
            
            # Load feature names
            feature_names_path = os.path.join('models', 'feature_names.txt')
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'r') as f:
                    feature_names = [line.strip() for line in f.readlines()]
            else:
                feature_names = None
            
            # Load dataset
            data_path = os.path.join('data', 'cleaned_data.csv')
            if os.path.exists(data_path):
                dataset = pd.read_csv(data_path)
                print(f"Loaded dataset with {dataset.shape[0]} records")
            else:
                raise FileNotFoundError("Cleaned dataset not found")
            
            return model, dataset, feature_names
        else:
            raise FileNotFoundError("Model not found, please train the model first")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        "message": "SHAP/XGBoost Analytics API",
        "endpoints": {
            "/analyze": "POST - Run analysis with structured query",
            "/health": "GET - Check system health"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model": "xgboost",
        "features": feature_names,
        "data_shape": f"{dataset.shape[0]} rows, {dataset.shape[1]} columns" if dataset is not None else None
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get the query from the request
        query = request.json
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Log the received query
        print(f"Received query: {query}")
        
        # Extract query parameters
        analysis_type = query.get('analysis_type', 'correlation')
        target_variable = query.get('target_variable', 'Nike_Sales')
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
        
        # Calculate SHAP values
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
        
        return jsonify({
            "success": True,
            "results": results,
            "summary": summary,
            "feature_importance": feature_importance,
            "shap_values": shap_values_dict
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during analysis: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# Load model and dataset when the app starts
print("Loading model and dataset...")
model, dataset, feature_names = load_model()
print("Model and dataset loaded successfully")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
```

## Step 5: Create Dependencies File

Create a requirements.txt file with all dependencies:

```bash
# Create requirements.txt
touch requirements.txt
```

Add the following to requirements.txt:

```txt
flask==2.2.3
flask-cors==3.0.10
gunicorn==20.1.0
numpy==1.23.5
pandas==1.5.3
xgboost==1.7.4
shap==0.41.0
scikit-learn==1.2.2
python-dotenv==1.0.0
```

## Step 6: Create Configuration Files for Render

Create a render.yaml file for Render's Blueprint deployment:

```bash
touch render.yaml
```

Add the following to render.yaml:

```yaml
services:
  - type: web
    name: shap-analytics
    env: python
    plan: starter
    buildCommand: pip install -r requirements.txt && python train_model.py
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.9
```

Create a .gitignore file:

```bash
touch .gitignore
```

Add the following to .gitignore:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# Model files
models/*.pkl

# Jupyter Notebooks
.ipynb_checkpoints

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
```

## Step 7: Test Locally

Test the microservice locally before deploying:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Make sure your CSV data is in the data directory
# data/sales_data.csv should contain your actual data

# Train the model
python train_model.py

# Start the service
python app.py
```

Test the API with a tool like curl or Postman:

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_type": "correlation",
    "target_variable": "Nike_Sales",
    "demographic_filters": ["Income > 50000"],
    "output_format": "top_10_zip_codes"
  }'
```

## Step 8: Initialize Git Repository

Initialize a Git repository and commit all files:

```bash
# Initialize Git repository
git init

# Add all files
git add .

# Don't commit actual data files, but commit .gitkeep to maintain directory structure
touch data/.gitkeep
git add data/.gitkeep
touch models/.gitkeep
git add models/.gitkeep

# Commit changes
git commit -m "Initial commit: SHAP/XGBoost microservice"
```

## Step 9: Create GitHub Repository

1. Go to GitHub.com and log in
2. Click "New" to create a new repository
3. Name it "shap-microservice"
4. Keep it public (or private if you prefer)
5. Do not initialize with README, .gitignore, or license
6. Click "Create repository"

Connect your local repository to GitHub:

```bash
# Add the GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/shap-microservice.git

# Push your code to GitHub
git branch -M main
git push -u origin main
```

## Step 10: Deploy to Render

Before deploying, you need to ensure your data is accessible during build. There are several approaches:

### Option 1: Include a small sample dataset in your repository

Create a small sample of your data (e.g., 100 records) and include it in your repository:

```bash
# Create a small sample of your data
head -n 100 data/sales_data.csv > data/sample_data.csv

# Update your train_model.py to use sample_data.csv if in production
# Update the relevant part of train_model.py
```

### Option 2: Download data during build

If your data is publicly accessible, modify train_model.py to download the data during build:

```python
# In train_model.py
import requests

# Download data if not exists
if not os.path.exists(data_path):
    print("Downloading dataset...")
    url = "https://your-data-hosting-url/sales_data.csv"
    response = requests.get(url)
    with open(data_path, 'wb') as f:
        f.write(response.content)
    print("Download complete")
```

Now, deploy the service to Render:

1. Sign up or log in to Render ([https://render.com](https://render.com))
2. In the Render dashboard, click "New +"
3. Select "Web Service"
4. Connect your GitHub account if you haven't already
5. Select the "shap-microservice" repository you just created
6. Click "Connect"
7. Configure the service:
   - Name: "shap-analytics"
   - Environment: "Python"
   - Region: Choose the region closest to your users
   - Branch: "main"
   - Build Command: `pip install -r requirements.txt && python train_model.py`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - Plan: "Free" (for testing)
   - Click "Create Web Service"
8. Wait for the deployment to complete (this may take several minutes)
9. Once deployed, note the URL Render provides (e.g., [https://shap-analytics.onrender.com](https://shap-analytics.onrender.com))

## Step 11: Update Your Next.js App Configuration

Update your AI Analytics App's environment variables:

```bash
# Edit .env.local file
nano .env.local
```

Add or update these environment variables:

```env
# SHAP Integration
NEXT_PUBLIC_ENABLE_SHAP=true
NEXT_PUBLIC_SHAP_SERVICE_URL=https://your-service-name.onrender.com/analyze
```

Update your shapClient.ts:

```typescript
// filepath: /Users/voldeck/code/ai-analytics-app/src/app/lib/shapClient.ts
/**
 * Client for communicating with the XGBoost/SHAP microservice
 */

import { StructuredQuery, AnalysisResult } from '../types';
import { trackApiCall } from './monitoring';

const microserviceUrl = process.env.NEXT_PUBLIC_SHAP_SERVICE_URL || 'http://localhost:5000/analyze';

/**
 * Send structured query to the XGBoost/SHAP microservice
 * @param query The structured query object
 * @returns Analysis results from the XGBoost/SHAP microservice
 */
export async function runShapAnalysis(query: StructuredQuery): Promise<AnalysisResult> {
  const startTime = Date.now();
  
  try {
    const response = await fetch(microserviceUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(query)
    });

    // Track the API call for monitoring
    trackApiCall(
      microserviceUrl,
      'POST',
      startTime,
      response.status
    );

    if (!response.ok) {
      throw new Error(`SHAP service error: ${response.status}`);
    }

    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error || 'SHAP service returned an error');
    }
    
    return {
      type: query.analysis_type,
      data: data.results,
      summary: data.summary,
      featureImportance: data.feature_importance, // SHAP-specific data
      shapValues: data.shap_values // SHAP-specific data
    };
  } catch (error) {
    // Track the error
    trackApiCall(
      microserviceUrl,
      'POST',
      startTime,
      500,
      error instanceof Error ? error.message : 'Unknown error'
    );
    
    console.error('Error calling XGBoost/SHAP service:', error);
    throw new Error('Failed to process query with advanced analytics service');
  }
}
```

## Step 12: Troubleshooting

### Common Issues

1. **CSV Data Format Issues**:
   - Ensure your CSV has the correct column names expected by the code
   - Check for data type inconsistencies
   - Use data validation to catch issues early

2. **Model Training Failures**:
   - Check Render build logs for training errors
   - Ensure your CSV has enough data for meaningful training
   - Consider simplifying the model if you have limited data

3. **API Response Issues**:
   - Verify the request structure matches what the API expects
   - Check CORS settings if calling from a browser
   - Look for errors in console logs

4. **Deployment Failures**:
   - Inspect Render build logs for errors
   - Make sure your data handling strategy works in the cloud environment
   - Check memory usage if your model or data is large

### Monitoring Your Service

1. **Render Dashboard**:
   - Regularly check metrics in the Render dashboard
   - Set up alerts for service downtime

2. **Logging**:
   - Add detailed logging to your application
   - Consider log aggregation services for production

3. **Performance Tracking**:
   - Monitor response times
   - Track usage patterns to plan scaling needs

## Step 13: Ongoing Maintenance

For maintaining your microservice:

1. **Data Updates**:
   - Create a process for updating your dataset
   - Consider incremental updates to avoid full retraining

2. **Model Retraining**:
   - Establish a regular schedule for model retraining
   - Track model performance metrics over time

3. **API Evolution**:
   - Version your API if you make breaking changes
   - Document changes for consumers of your service

4. **Security**:
   - Implement API key authentication
   - Keep dependencies updated to address security vulnerabilities

## Future Enhancements

The following enhancements are planned for future releases:

### Incremental Model Updating

- Create a scheduled job that periodically checks for new data
- Implement incremental model training that incorporates new data without full retraining
- Add versioning for incremental updates
- Track model performance metrics across updates
- Implement automatic rollback if performance degrades

```python
# Sample code for incremental model updating
def update_model_incrementally(model, new_data, feature_names):
    """Update an existing XGBoost model with new data."""
    # Prepare new data
    X_new = new_data[feature_names]
    y_new = new_data['target_variable']
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_new, label=y_new)
    
    # Update model with new data (fewer iterations)
    update_params = {
        'eta': 0.01,  # Use smaller learning rate for updates
        'max_depth': 5,
        'objective': 'reg:squarederror'
    }
    
    # Only run a few boosting rounds for the update
    model.update(dtrain, 10)
    return model
```

### Monitoring Dashboard

- Create a dedicated monitoring UI for the microservice
- Track API usage statistics
- Monitor model performance metrics over time
- Visualize feature importance drift
- Set up alerts for performance degradation
- Create visual representations of SHAP values

```python
# Sample code for dashboard metrics collection
def collect_dashboard_metrics(api_request, response, execution_time):
    """Collect metrics about API usage for the monitoring dashboard."""
    metrics = {
        'timestamp': datetime.datetime.now().isoformat(),
        'endpoint': request.path,
        'execution_time_ms': execution_time * 1000,
        'status_code': response.status_code,
        'client_ip': request.remote_addr
    }
    
    # Log metrics for dashboard
    with open('logs/api_metrics.jsonl', 'a') as f:
        f.write(json.dumps(metrics) + '\n')
        
    return metrics
```

---
By following these steps, you'll have a fully functional SHAP/XGBoost microservice deployed to Render that integrates with your AI Analytics App, providing advanced data analysis capabilities through a clean API interface.
