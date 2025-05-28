# SHAP Microservice Deployment Instructions

This package contains the optimized SHAP microservice ready for deployment to Render.com.

## Optimizations Applied

1. **Memory Settings**
   - Standard memory threshold increased to 475MB
   - Aggressive memory management disabled for worker
   - Batch size increased to 500 rows

2. **Redis Connection Fixes**
   - Enhanced connection handling with failsafe methods
   - Better timeout and retry configuration
   - Fixed connection pool handling

3. **Worker Process Improvements**
   - Using simple_worker.py instead of the Context manager pattern
   - Fixed Connection import issue

## Deployment Steps

1. **Upload to Render**
   - Push these files to your Git repository connected to Render
   - Or use the Render dashboard to deploy manually

2. **Verify Deployment**
   - Check the worker logs to ensure it starts properly
   - Verify Redis connection with the /admin/redis_ping endpoint
   - Monitor memory usage to ensure it stays below 512MB

3. **Run Verification Scripts**
   - Use verify_config.py to check configuration
   - Use verify_redis_settings.py to check Redis connection
   - Use run_test_job.py to test end-to-end functionality

## Monitoring

- Monitor worker memory usage during job processing
- Watch for any timeouts or connection issues
- Check job processing times with different batch sizes
