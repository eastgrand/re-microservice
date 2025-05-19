# Deploying Optimized SHAP/XGBoost Microservice to Render.com

This document provides a step-by-step guide for deploying the optimized SHAP/XGBoost microservice to Render.com. The optimizations include memory usage improvements, Redis connection stability fixes, and worker process enhancements to ensure reliable operation on Render's starter plan (512MB RAM, 0.5 CPU).

## Prerequisites

- GitHub account with the SHAP/XGBoost microservice code pushed to a repository
- Render.com account (create one at [https://render.com](https://render.com) if needed)
- Access to terminal/command line interface

## Optimization Summary

The following optimizations have been applied:

1. **Memory Usage Optimizations**
   - Increased memory threshold from 450MB to 475MB
   - Disabled aggressive memory management for the worker
   - Increased batch size from 300 to 500 rows

2. **Redis Connection Fixes**
   - Enhanced Redis connection handling with better error recovery
   - Added socket keepalive settings
   - Improved timeout and retry mechanisms

3. **Worker Process Enhancements**
   - Updated worker configuration to use `simple_worker.py`
   - Removed dependency on problematic `Connection` class
   - Improved error handling in worker processes

## Deployment Steps

### 1. Use the Optimized Deployment Script

We've created a comprehensive deployment script that handles all the necessary steps for deploying to Render.com with our memory and stability optimizations:

```bash
# Make the deployment script executable
chmod +x ./deploy_to_render_final.sh

# Run the deployment script
./deploy_to_render_final.sh
```

This script will:
- Verify all required files are present
- Create a `.skip_training` flag file to speed up deployment
- Ensure model files exist (or create minimal models)
- Set optimized environment variables
- Update `render.yaml` with the optimized settings
- Prepare the deployment package

### 2. Deploy to Render.com

You have two options for deploying to Render.com:

#### Option A: Git-based Deployment (Recommended)

1. Commit your optimized code to your Git repository:

```bash
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

We've created a verification script to help you confirm that your deployment is successful:

```bash
# Make the verification script executable
chmod +x ./verify_render_deployment.sh

# Run the verification script
./verify_render_deployment.sh
```

This script will:
- Check if your service is responding
- Verify Redis connection
- Check memory usage
- Submit a test SHAP job (optional)
- Monitor job status to ensure proper processing

Alternatively, you can manually verify the deployment:

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

## Monitoring and Maintenance

### Performance Monitoring

To ensure your SHAP microservice continues to operate efficiently on Render's starter plan:

1. **Regular Health Checks**
   - Use the `/health` endpoint to verify service status
   - Check the `/memory-stats` endpoint to monitor memory usage
   - Verify Redis connection via the `/redis-check` endpoint

2. **Job Processing Metrics**
   - Monitor job completion times and success rates
   - Watch for any increases in processing time
   - Check for jobs that get stuck in the "started" state

3. **Memory Optimization**
   - Keep memory usage below 475MB to avoid OOM issues
   - Consider implementing a cron job to restart the worker daily
   - Monitor memory leaks by checking if usage steadily increases over time

### Maintenance Tasks

1. **Regular Updates**
   - Periodically update dependencies in `requirements.txt`
   - Test updates in a staging environment before production
   - Keep an eye on SHAP library updates that might improve performance

2. **Log Rotation**
   - Render handles log rotation automatically
   - Regularly review logs for recurring errors or warnings
   - Set up log-based alerts for critical issues

3. **Backup Strategy**
   - Regularly backup your model files
   - Consider version control for your models
   - Document any manual configuration changes

### Scaling Considerations

As your usage grows, consider these scaling options:

1. **Vertical Scaling**
   - Upgrade to a larger Render plan for more memory and CPU
   - Update memory thresholds in `optimize_memory.py` accordingly

2. **Horizontal Scaling**
   - Add more worker instances for parallel processing
   - Implement a load balancer if needed

3. **Data Optimizations**
   - Implement feature selection to reduce dimensionality
   - Consider downsampling large datasets
   - Pre-process data to optimize for SHAP analysis

## Conclusion

With the optimizations implemented in this deployment, the SHAP microservice should now operate reliably on Render's starter plan. Regular monitoring and maintenance will ensure continued performance and stability.

Remember to check the Render dashboard and logs periodically, and adjust the optimization parameters as needed based on your specific usage patterns.

## Troubleshooting

If you encounter issues during deployment or operation:

### Build Failures

- Check the build logs for specific error messages
- Ensure all dependencies are correctly specified in `requirements.txt`
- Verify that the Python version specified in `render.yaml` is supported
- Check if the pre-deployment script (`render_pre_deploy.sh`) ran successfully

### Runtime Errors

- Check the application logs in the Render dashboard
- Verify that all environment variables are correctly set
- Ensure that model files exist in the deployed environment

### Memory Issues

- Monitor memory usage via the `/memory-stats` endpoint or Render dashboard
- If memory usage approaches 512MB:
  - Reduce `SHAP_MAX_BATCH_SIZE` in environment variables (try 400 or 300)
  - Enable `AGRESSIVE_MEMORY_MANAGEMENT` by setting it to "true"
  - Consider increasing the `MAX_MEMORY_MB` threshold down to 450MB

### Redis Connection Issues

- Verify Redis connection via the `/redis-check` endpoint
- Check Redis URL in environment variables
- Look for Redis timeout errors in the logs
- If you see connection errors:
  - Increase `REDIS_TIMEOUT` value (try 15 or 20 seconds)
  - Ensure `REDIS_SOCKET_KEEPALIVE` is set to "true"

### Jobs Stuck in "Started" State

- Check worker logs for any errors or exceptions
- Verify that the Redis queue is functioning properly
- Ensure that worker processes are running
- If jobs remain stuck:
  - Check memory usage (may be OOM issues)
  - Verify model loading is successful
  - Try restarting the worker service

### API Access Issues

- Confirm that the `X-API-KEY` header is included in all requests
- Verify that the API key used matches the one set in environment variables
- Check if `REQUIRE_AUTH` is set to `true`

## Troubleshooting Worker Name Collisions

If you encounter worker name collision errors during deployment, follow these steps:

### Symptoms
- Error messages like: `ValueError: There exists an active worker named 'simple-worker-65' already`
- Multiple worker services in Render dashboard
- Workers failing to start

### Solution

1. **Update Worker Naming**
   - The worker script has been updated to use unique identifiers
   - Includes hostname, PID, timestamp, and random UUID

2. **Clean Up Redis Worker Registrations**
   - Use the cleanup function in `simple_worker.py`
   - Removes stale worker registrations from Redis

3. **Update Render Configuration**
   - Ensure only one worker service is defined in render.yaml
   - Use a clean, simple name like `nesto-shap-worker`
   - Remove any duplicate start commands

4. **Execute the Fix Script**
   ```bash
   ./deploy_fix_to_render.sh
   ```
   This script:
   - Verifies and updates render.yaml
   - Commits and pushes changes
   - Provides instructions for Render dashboard cleanup

### Verification
After applying the fix:
1. Delete existing worker services in Render dashboard
2. Deploy the updated configuration
3. Verify only one worker is created
4. Check worker logs for successful startup
5. Submit a test job to verify it processes correctly

For more details, see [WORKER-NAME-COLLISION-FIX.md](WORKER-NAME-COLLISION-FIX.md)
