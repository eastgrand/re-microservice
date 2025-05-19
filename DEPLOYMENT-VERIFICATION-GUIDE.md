# SHAP Microservice Deployment Verification Guide

## Overview

This document provides steps to verify that the SHAP microservice has been successfully deployed to Render.com, with specific focus on validating that the worker name collision issue has been fixed.

## Verification Steps

### 1. Check Deployment Status

1. Log into the [Render.com Dashboard](https://dashboard.render.com/)
2. Navigate to the project
3. Check that both the web service and worker service show as "Live" status
4. Note any warnings or errors displayed in the dashboard

### 2. Verify Worker Startup

Check the worker logs to ensure it starts successfully:

1. In the Render dashboard, click on the worker service (`nesto-mortgage-analytics-worker-05162025`)
2. Navigate to the "Logs" tab
3. Verify that you see the following log entries:
   
   ```
   Starting memory-optimized SHAP worker
   Garbage collection enabled
   Connecting to Redis...
   Redis connection successful.
   Checking for stale worker registrations...
   Starting simple worker...
   Worker created with ID: shap-worker-[unique-id]
   Applied memory optimization patches
   Using max batch size: 500 rows
   Patching Redis with: timeout=5s, keepalive=True, pool_size=10
   Redis connection handling has been patched
   Applied Redis connection patches
   Starting to process jobs...
   Worker rq:worker:shap-worker-[unique-id] started with PID [number]
   Listening on shap-jobs...
   ```

4. Confirm there are **NO** errors with worker name collision:

   ```
   ValueError: There exists an active worker named 'simple-worker-65' already
   ```

### 3. Check Redis Connection

Verify the Redis connection is stable:

1. In the web service logs, look for:
   ```
   Redis connection successful. Ping time: [XX] ms
   ```

2. Use the Redis health check endpoint:
   ```
   curl https://[your-service-url]/admin/redis_ping
   ```

   Expected response:
   ```json
   {
     "success": true,
     "ping": true,
     "response_time_ms": XX.XX
   }
   ```

### 4. Test Job Processing

Submit a test job to verify the worker processes it correctly:

1. Use the verification script:
   ```bash
   ./verify_render_deployment.sh
   ```

   Or manually submit a job:
   ```bash
   curl -X POST \
     -H "Content-Type: application/json" \
     -H "X-API-Key: YOUR_API_KEY" \
     -d '{
       "data": [
         {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}
       ],
       "options": {
         "batch_size": 1
       }
     }' \
     https://[your-service-url]/api/submit-shap-job
   ```

2. Note the job ID from the response and check its status:
   ```bash
   curl https://[your-service-url]/api/job-status/[job-id]
   ```

3. Verify that the job progresses from `"status": "started"` to `"status": "completed"` within a reasonable timeframe

### 5. Monitor Memory Usage

Check memory usage to ensure it stays within limits:

1. In the Render dashboard, view the metrics for the worker service
2. Verify memory usage stays below 512MB
3. Check the memory stats endpoint:
   ```bash
   curl https://[your-service-url]/memory-stats
   ```

   Example response:
   ```json
   {
     "memory_usage_mb": 325.45,
     "memory_limit_mb": 475,
     "percent_used": 68.52,
     "status": "ok"
   }
   ```

### 6. Verify Multiple Restarts

To fully confirm the worker name collision issue is fixed:

1. In the Render dashboard, restart the worker service
2. Check logs to confirm it starts with a new unique name
3. Repeat restart 2-3 times
4. Verify that each restart results in successful startup without worker name collision errors

## Troubleshooting Common Issues

### Worker Name Collision Persists

If worker name collision errors still appear:

1. Check that the latest code with the fix is deployed
2. Look for errors in the cleanup process log entries
3. Try manually clearing stale workers in Redis (if you have Redis access)

### Job Processing Issues

If jobs get stuck in the "started" state:

1. Check worker logs for exceptions during job processing
2. Verify memory usage is not hitting the limit
3. Look for timeouts in Redis operations

### Memory Usage Too High

If memory usage is approaching the 512MB limit:

1. Consider reducing the `SHAP_MAX_BATCH_SIZE` value
2. Enable `AGGRESSIVE_MEMORY_MANAGEMENT=true` in the environment variables
3. Decrease the `MAX_MEMORY_MB` threshold to start memory optimizations earlier

## Next Steps After Successful Verification

1. Document the deployed service URL and API endpoints
2. Set up monitoring alerts in Render dashboard
3. Schedule regular health checks
4. Plan for any needed performance optimizations based on observed memory and processing metrics
