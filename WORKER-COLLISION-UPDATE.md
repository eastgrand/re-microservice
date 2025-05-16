# Deployment Update: Worker Name Collision Fix

## Issue Summary

During our deployment to Render.com, we encountered an issue where the worker service was failing to start with the following error:

```
ValueError: There exists an active worker named 'simple-worker-65' already
```

This prevented the SHAP microservice from processing jobs correctly, as the worker would fail to start properly after container restarts or redeployments.

## Root Cause Analysis

1. **Worker Naming Convention**: The worker was using a simple naming scheme based only on process ID (PID): `simple-worker-{os.getpid()}`
2. **Process ID Reuse**: When Render restarts containers, it often assigns the same PIDs
3. **Redis Key Persistence**: Redis retained the registration for the previous worker instance with the same name
4. **No Cleanup Mechanism**: Our code lacked a cleanup process to remove stale worker registrations

## Implemented Solution

1. **Unique Worker Names**: Updated the worker naming scheme to use a combination of:
   - Hostname
   - Process ID 
   - Current timestamp
   - Random UUID

2. **Stale Worker Cleanup**: Added a function that runs before worker startup to clean up any stale worker registrations in Redis

3. **Retry Mechanism**: Implemented a retry mechanism that detects name collisions and automatically retries with a new unique name

4. **Service Name Uniqueness**: Added a timestamp to the worker service name in `render.yaml` to prevent collisions at the Render.com service level

## Implementation Details

1. **Updated `simple_worker.py`**:
   - Added stale worker cleanup function
   - Implemented unique worker naming with timestamps and UUIDs
   - Added retry logic for worker startup failures

2. **Updated `render.yaml`**:
   - Changed worker service name to include a timestamp
   - Updated worker startup command for better error handling

3. **Created `fix_worker_name_collision.sh`**:
   - Script to apply these fixes in one step
   - Can be run as part of the deployment process

4. **Added Worker Status Monitoring**:
   - New `/worker-status` endpoint to check worker registration
   - Updated verification script to check worker health

## Verification

We can verify the fix is working by:

1. Checking worker logs in Render dashboard - no more name collision errors
2. Using the `/worker-status` endpoint to confirm worker registration
3. Monitoring job processing to ensure jobs transition from "started" to "completed" state

## Preventing Future Issues

1. **Regular Monitoring**: Use the `/worker-status` endpoint to monitor worker health
2. **Periodic Cleanup**: Redis keys should be periodically checked and cleaned up
3. **Health Checks**: Deploy proper health checks to detect and restart failed workers
4. **Worker Pool Management**: Consider using a worker pool manager for larger deployments

## Related Documentation

For more details, please refer to:
- [WORKER-NAME-COLLISION-FIX.md](/Users/voldeck/code/shap-microservice/WORKER-NAME-COLLISION-FIX.md) - Technical details of the fix
- [RENDER-DEPLOYMENT-GUIDE.md](/Users/voldeck/code/shap-microservice/RENDER-DEPLOYMENT-GUIDE.md) - Updated deployment guide
