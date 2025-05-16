# Worker Name Collision Resolution

## Issue Summary

During the deployment to Render.com on May 15, 2025, the worker service failed to start with the error:

```
ValueError: There exists an active worker named 'simple-worker-65' already
```

This occurred because:
1. The worker name was being generated using only the process ID (`os.getpid()`)
2. When Render restarts containers, it often reuses the same PID
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
       try:
           logger.info("Checking for stale worker registrations...")
           existing_workers_key = 'rq:workers'
           worker_keys = conn.smembers(existing_workers_key)
           
           if not worker_keys:
               logger.info("No existing workers found.")
               return
               
           logger.info(f"Found {len(worker_keys)} worker registrations.")
           cleaned = 0
           
           for worker_key in worker_keys:
               try:
                   # Check if the worker is still alive using its heartbeat
                   heartbeat_key = f"{worker_key.decode()}:heartbeat"
                   last_heartbeat = conn.get(heartbeat_key)
                   
                   # If no heartbeat or heartbeat is old (>60 seconds), clean up
                   if not last_heartbeat or (time.time() - float(last_heartbeat.decode())) > 60:
                       logger.info(f"Cleaning up stale worker: {worker_key.decode()}")
                       
                       # Delete the worker key and remove from workers set
                       conn.delete(worker_key)
                       conn.srem(existing_workers_key, worker_key)
                       cleaned += 1
               except Exception as e:
                   logger.warning(f"Error checking worker {worker_key}: {str(e)}")
                   
           if cleaned > 0:
               logger.info(f"Cleaned up {cleaned} stale worker registrations")
           else:
               logger.info("No stale workers found")
       except Exception as e:
           logger.warning(f"Error during stale worker cleanup: {str(e)}")
   ```

3. **Service Name Update** in `render.yaml`:
   ```yaml
   # Changed from:
   name: nesto-mortgage-analytics-worker
   
   # To:
   name: nesto-mortgage-analytics-worker-05162025
   ```

## Implementation Steps

1. **Created Fix Script**: 
   - Developed `fix_worker_name_collision.sh`
   - This script backs up the original worker script
   - Deploys the fixed worker implementation
   - Updates the render.yaml configuration

2. **Applied the Fix**:
   ```bash
   ./fix_worker_name_collision.sh
   ```

3. **Deployed to Render**:
   ```bash
   ./deploy_to_render_final.sh
   ```

## Verification

The worker now successfully starts without the name collision error. The logs show:

```
2025-05-16 01:24:48,287 - INFO - Worker rq:worker:shap-worker-[unique-id] started with PID 64
2025-05-16 01:24:48,559 - INFO - *** Listening on shap-jobs...
2025-05-16 01:24:48,842 - INFO - Scheduler for shap-jobs started with PID 75
```

## Future Recommendations

1. **Periodic Cleanup**: Consider implementing a scheduled task to periodically clean up stale worker registrations in Redis.

2. **Heartbeat Monitoring**: Add a monitoring system to check worker heartbeats and restart workers that have stopped reporting.

3. **Worker Manager**: Consider using a process manager like Supervisor to manage worker processes more reliably.

4. **Render Service Updates**: When updating services in Render.com, consider using a new service name to avoid conflicts with previous deployments.
