# SHAP 500 Error Fix - Deployment Instructions

**Date:** May 15, 2025  
**Issue:** HTTP 500 errors from /analyze endpoint with jobs getting stuck in "started" state

## Fix Summary

We've implemented a comprehensive solution for the 500 errors in the SHAP microservice:

1. **Memory-Optimized SHAP Analysis**
   - Created `shap_memory_fix.py` for chunked SHAP processing
   - Added enhanced error reporting and memory monitoring
   - Implemented garbage collection at critical points

2. **Worker Process Improvements**
   - Fixed worker configuration to include memory optimization
   - Added job recovery utilities to repair stuck jobs
   - Enhanced monitoring to detect issues earlier

3. **Render Configuration Updates**
   - Updated worker settings in `render.yaml`
   - Added memory optimization environment variables
   - Improved startup sequence with proper ordering

## Deployment Steps

1. **Prepare your local environment**

   All the necessary files have been created and configured. The changes have been applied to your codebase.

2. **Commit changes to Git**

   ```bash
   git add app.py shap_memory_fix.py render.yaml repair_stuck_jobs.py verify_shap_fix.py SHAP-500-ERROR-FIX.md SHAP-MEMORY-FIX.md
   git commit -m "Add SHAP memory optimization fix for 500 errors"
   ```

3. **Push to GitHub**

   ```bash
   git push origin main
   ```

4. **Deploy to Render**

   Option 1: Automatic deployment will trigger when you push to GitHub

   Option 2: Manual deployment from Render dashboard:
   - Go to your Render dashboard
   - Select your SHAP microservice
   - Click "Manual Deploy" > "Deploy latest commit"

5. **Verify the fix**

   After deployment completes:
   ```bash
   # Set your API key if needed
   export API_KEY="your_api_key" 
   
   # Set your service URL
   export SHAP_SERVICE_URL="https://your-service-url"
   
   # Run verification script
   python3 verify_shap_fix.py
   ```

## Troubleshooting

If you encounter any issues:

1. **Check Render Logs**
   - Look for memory usage reports
   - Check for any errors during SHAP analysis
   - Verify worker processes are starting correctly

2. **Monitor Memory Usage**
   ```bash
   curl https://your-service-url/admin/memory
   ```

3. **Clean Up Stuck Jobs**
   ```bash
   curl -X POST https://your-service-url/admin/cleanup_jobs
   ```

4. **Adjust Memory Settings**
   If you still experience issues, you can tune the batch size:
   - Go to Render dashboard > Environment Variables
   - Set `SHAP_BATCH_SIZE` to a smaller value (e.g., 250 or 100)

## Support

For any additional help or questions about this fix, please reach out to the DevOps team.
