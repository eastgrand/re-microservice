# Deploying SHAP/XGBoost Microservice to Render.com

This document provides a step-by-step guide for deploying the SHAP/XGBoost microservice to Render.com.

## Prerequisites

- GitHub account with the SHAP/XGBoost microservice code pushed to a repository
- Render.com account (create one at [https://render.com](https://render.com) if needed)

## Deployment Steps

### 1. Prepare Your Repository

Before deploying, ensure your GitHub repository has the following files:

- `requirements.txt` - Lists all Python dependencies
- `render.yaml` - Render configuration file
- `.env.example` - Template for environment variables
- All necessary code files (app.py, train_model.py, etc.)

### 2. Set Up GitHub Repository

1. Create a new GitHub repository (if you haven't already)
2. Push your code to the repository:

```bash
# Initialize Git repository
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit of SHAP/XGBoost microservice"

# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/shap-microservice.git

# Push to GitHub
git push -u origin main
```

### 3. Set Up Render.com Project

1. Log in to your Render account at [https://dashboard.render.com](https://dashboard.render.com)
2. Click "New" and select "Blueprint" from the dropdown menu
3. Connect your GitHub account if not already connected
4. Select the repository containing your SHAP/XGBoost microservice
5. Click "Apply Blueprint"

Render will automatically detect the `render.yaml` file and configure your service accordingly.

### 4. Configure Environment Variables

After the initial deployment, you need to set up environment variables:

1. Navigate to your service on the Render dashboard
2. Click on "Environment"
3. Add the following environment variables:

| Key | Value | Description |
|-----|-------|-------------|
| `PORT` | `10000` | Port for the application to listen on |
| `DEBUG` | `false` | Set to `true` for development |
| `LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `REQUIRE_AUTH` | `true` | Enable API key authentication |
| `API_KEY` | `your-secure-api-key` | Secret API key for authentication |

> **Important**: For `API_KEY`, use a secure, randomly generated value. This is a sensitive environment variable.

### 5. Deploy and Monitor

1. Navigate to the "Manual Deploy" section of your service
2. Click "Deploy latest commit" to trigger a deployment
3. Monitor the build logs to ensure everything is deploying correctly

The build process will:

- Install dependencies from requirements.txt
- Apply the SHAP compatibility patch
- Train the initial model (if sample data is being used)
- Start the service

### 6. Verify Deployment

After successful deployment:

1. Get your service URL from the Render dashboard (e.g., `https://shap-analytics.onrender.com`)
2. Test the API with curl:

```bash
# Test health endpoint
curl -H "X-API-KEY: your-api-key" https://shap-analytics.onrender.com/health

# Test metadata endpoint
curl -H "X-API-KEY: your-api-key" https://shap-analytics.onrender.com/metadata
```

### 7. Scaling and Monitoring

Render offers various scaling options:

1. **Vertical Scaling**: Increase resources for your service
   - Navigate to your service
   - Click "Settings"
   - Under "Instance Type", select a plan with more resources

2. **Monitoring**:
   - Use Render's built-in monitoring features to track performance
   - Set up Render alerts for service outages

## Troubleshooting

If you encounter issues during deployment:

### Build Failures

- Check the build logs for specific error messages
- Ensure all dependencies are correctly specified in `requirements.txt`
- Verify that the Python version specified in `render.yaml` is supported

### Runtime Errors

- Check the application logs in the Render dashboard
- Verify that all environment variables are correctly set
- Ensure the model training process completes successfully

### API Access Issues

- Confirm that the `X-API-KEY` header is included in all requests
- Verify that the API key used matches the one set in environment variables
- Check if `REQUIRE_AUTH` is set to `true`
