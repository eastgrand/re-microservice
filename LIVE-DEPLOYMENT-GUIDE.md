# SHAP Microservice Live Deployment

## Current Status

The SHAP microservice is currently experiencing issues on the live Render.com deployment, identified as `https://nesto-mortgage-analytics.onrender.com`. 

### Issues Identified:

1. **Service is only partially responsive**:
   - The basic ping endpoint works
   - Administrative endpoints return 500 errors
   - Jobs are being queued but not processed

2. **Memory Optimization Required**:
   - The worker processes are likely running out of memory when processing SHAP calculations
   - The memory optimization fix we developed needs to be applied to the live service

## Solution

We've developed a comprehensive memory optimization solution that:

1. Processes SHAP calculations in batches to avoid memory issues
2. Adds garbage collection at strategic points
3. Monitors memory usage
4. Updates the worker configuration

## Deployment Instructions

Follow these steps to deploy the fix to the live service:

### 1. Deploy the Memory Optimization Fix

Run the deployment script:

```bash
./deploy_to_live_service.sh
```

This script will:
- Ensure the memory optimization code is in place
- Update app.py to use the optimization
- Configure Render.yaml with appropriate settings
- Commit the changes
- Push to the remote Git repository (triggering automatic deployment on Render)

### 2. Wait for Deployment

Render.com will automatically deploy the changes once they're pushed to the repository. This typically takes 5-10 minutes.

### 3. Verify the Deployment

Once the deployment is complete, verify that the fix is working:

```bash
./verify_live_endpoint.sh
```

This will run a verification test against the live service URL.

### 4. Run Comprehensive Health Check

For a more detailed health check:

```bash
./comprehensive_health_check.py
```

This will run a comprehensive health check on all aspects of the service.

## Monitoring and Maintenance

After deployment, you can monitor the service using:

```bash
./check_worker_status.py
```

## Fallback Plan

If the deployment doesn't resolve the issues, you can:

1. Check the Render.com logs for errors
2. Try increasing the SHAP_MAX_BATCH_SIZE environment variable to a smaller value (250 or 100)
3. Contact Render.com support if memory issues persist

## Documentation

The full implementation of the memory optimization strategy is documented in:
- `shap_memory_fix.py` - The core optimization code
- `MEMORY-OPTIMIZATIONS.md` - Detailed explanation of the memory optimizations
- `SHAP-MEMORY-FIX.md` - Implementation guide
