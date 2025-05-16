# Worker Name Collision Resolution

## Issue Overview

During deployment to Render.com, the SHAP microservice worker was encountering this error:

```
ValueError: There exists an active worker named 'simple-worker-65' already
```

This error occurred because:

1. The worker script was using the process ID (PID) as part of the worker name
2. When Render restarts containers, PIDs can be reused
3. Redis still had old worker registrations that weren't properly cleaned up
4. Render.yaml had duplicate worker configurations that created multiple workers

## Solution Implemented

### 1. Fixed the Worker Naming Scheme

The worker naming scheme was updated to use multiple unique identifiers:
- Hostname from environment variables
- Process ID
- Current timestamp
- Random UUID segment

This ensures that even if the container is restarted with the same PID, the worker name will still be unique.

```python
hostname = os.environ.get('HOSTNAME', 'unknown')
unique_id = f"{hostname}-{os.getpid()}-{int(time.time())}-{str(uuid.uuid4())[:8]}"
worker_name = f"shap-worker-{unique_id}"
```

### 2. Added Stale Worker Cleanup

Implemented a cleanup function that runs before starting a new worker to remove any stale worker registrations from Redis:

```python
def cleanup_stale_workers(conn):
    """Clean up stale worker registrations from Redis"""
    try:
        existing_workers_key = 'rq:workers'
        worker_keys = conn.smembers(existing_workers_key)
        
        for worker_key in worker_keys:
            try:
                # Check if the worker is still alive using its heartbeat
                heartbeat_key = f"{worker_key.decode()}:heartbeat"
                last_heartbeat = conn.get(heartbeat_key)
                
                # If no heartbeat or heartbeat is old (>60 seconds), clean up
                if not last_heartbeat or (time.time() - float(last_heartbeat.decode())) > 60:
                    conn.delete(worker_key)
                    conn.srem(existing_workers_key, worker_key)
            except Exception as e:
                logger.warning(f"Error checking worker {worker_key}: {str(e)}")
    except Exception as e:
        logger.warning(f"Error during stale worker cleanup: {str(e)}")
```

### 3. Added Retry Logic

Added retry logic to handle worker name collisions:

```python
max_retries = 3
for attempt in range(1, max_retries + 1):
    try:
        if attempt > 1:
            # Generate a new unique name for retry
            unique_id = f"{hostname}-{os.getpid()}-{int(time.time())}-{str(uuid.uuid4())[:8]}"
            worker_name = f"shap-worker-retry-{unique_id}"
            worker = Worker(['shap-jobs'], connection=conn, name=worker_name)
        
        worker.work(with_scheduler=True)
        break
        
    except ValueError as ve:
        if "There exists an active worker" in str(ve) and attempt < max_retries:
            logger.warning(f"Worker name collision detected: {str(ve)}")
            logger.info(f"Waiting 5 seconds before retry...")
            time.sleep(5)
        else:
            logger.error(f"Worker error: {str(ve)}")
            if attempt >= max_retries:
                logger.error("Maximum retry attempts reached")
```

### 4. Simplified Render.yaml Worker Configuration

The render.yaml file was updated to have only one worker service with a clean name:

```yaml
# Worker service for processing SHAP jobs
- type: worker
  name: nesto-shap-worker
  env: python
  # ...other configuration...
  startCommand: >-
    echo "Starting memory-optimized SHAP worker" &&
    python3 -c "import gc; gc.enable(); print('Garbage collection enabled')" &&
    chmod +x ./simple_worker.py &&
    ./simple_worker.py
```

## Deployment Instructions

1. Delete any existing worker services in the Render dashboard
2. Push the updated render.yaml to trigger a new deployment
3. Verify that only one worker service is created
4. Monitor the worker logs to ensure it starts correctly without name collisions

## Verification

To verify the fix:
1. Check the worker logs for successful startup
2. Ensure there are no "Worker name collision" errors
3. Submit a test job and verify it completes successfully
4. Monitor Redis connections to ensure they remain stable

## Future Considerations

- Consider implementing a more robust worker management system
- Add health checks to automatically restart workers if they fail
- Implement Redis key expiration for worker registrations
- Consider using a dedicated Redis database for worker management