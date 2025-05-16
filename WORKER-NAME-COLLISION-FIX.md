# Worker Name Collision Fix

## Issue Summary

During the deployment to Render.com, the worker service was failing to start with the following error:

```
ValueError: There exists an active worker named 'simple-worker-65' already
```

This occurred because:
1. The worker name was being generated using only the process ID (`os.getpid()`)
2. When Render restarts containers, it may reuse the same PID
3. Redis still had the registration for the previous worker with the same name

## Implemented Fix

The following changes were made to resolve the issue:

1. **Unique Worker Names**: Updated the worker naming scheme to use a combination of:
   - Hostname
   - Process ID (PID)
   - Current timestamp
   - Random UUID

2. **Stale Worker Cleanup**: Added a function that cleans up stale worker registrations in Redis before starting a new worker

3. **Retry Logic**: Implemented retry logic for worker startup that:
   - Detects worker name collisions
   - Waits a few seconds before retry
   - Generates a new unique worker name for each attempt
   - Limits retries to prevent infinite loops

4. **Service Name Uniqueness**: Updated the worker service name in `render.yaml` to include a timestamp, preventing Render.com from using the same Redis keys for different worker instances

## Technical Details

1. **Worker Naming**: Changed from:
   ```python
   worker_name = f"simple-worker-{os.getpid()}"
   ```
   
   To:
   ```python
   hostname = os.environ.get('HOSTNAME', 'unknown')
   unique_id = f"{hostname}-{os.getpid()}-{int(time.time())}-{str(uuid.uuid4())[:8]}"
   worker_name = f"shap-worker-{unique_id}"
   ```

2. **Added Stale Worker Cleanup**:
   ```python
   def cleanup_stale_workers(conn):
       """Clean up stale worker registrations from Redis"""
       # Implementation that checks for and removes stale worker registrations
   ```

3. **Service Name Update** in `render.yaml`:
   ```yaml
   # Changed from:
   name: nesto-mortgage-analytics-worker
   
   # To:
   name: nesto-mortgage-analytics-worker-05162025
   ```

## Verification

The worker now successfully starts without encountering the name collision error. The logs show:

```
2025-05-16 01:24:48,287 - INFO - Worker rq:worker:simple-worker-64 started with PID 64, version 2.3.3
...
2025-05-16 01:24:48,559 - INFO - *** Listening on shap-jobs...
...
2025-05-16 01:24:48,842 - INFO - Scheduler for shap-jobs started with PID 75
```

## Future Considerations

1. **Regular Worker Cleanup**: Consider implementing a regular cleanup job that removes stale worker registrations from Redis

2. **Health Monitoring**: Add health checks to detect and restart workers that might be registered but not actually running

3. **Worker Process Management**: Consider using a process manager like Supervisord to manage worker processes if the RQ worker built-in management continues to be problematic
