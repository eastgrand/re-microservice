# SHAP Microservice Redis Connection Fix Guide

## Issue
The SHAP microservice was experiencing issues with Redis connections:
- Worker processes failing to process jobs
- Jobs getting stuck in the "started" state
- Timeouts in the service's health check endpoints
- Occasional Redis connection errors

## Solution
We've implemented several improvements to the Redis connection handling in the SHAP microservice:

1. **Connection Pooling**: Added connection pooling to efficiently manage Redis connections
2. **Proper Timeouts**: Set appropriate timeouts to prevent hanging connections
3. **Automatic Reconnection**: Added code to reconnect automatically if the Redis connection is lost
4. **Health Check Endpoints**: Added a `/admin/redis_ping` endpoint to test the Redis connection
5. **Error Handling**: Enhanced error handling for Redis Queue operations

## Files Modified

1. **redis_connection_patch.py**: New file with Redis connection improvements
2. **app.py**: Modified to use the Redis connection patch
3. **.env**: Added Redis connection settings

## Implementation Steps

The Redis connection fix was implemented in several steps:

1. Created a Redis connection patch module that:
   - Wraps Redis connection creation with better defaults
   - Adds proper timeouts and connection pooling
   - Implements automatic reconnection for failed Redis operations
   - Adds a health check endpoint for Redis

2. Modified app.py to:
   - Import the patch module
   - Apply the patch before creating the Redis connection
   - Keep the same functionality while improving reliability

3. Added environment variables to control Redis behavior:
   - `REDIS_TIMEOUT`: Connection timeout in seconds
   - `REDIS_SOCKET_KEEPALIVE`: Enable TCP keepalive for long-lived connections
   - `REDIS_CONNECTION_POOL_SIZE`: Number of connections to maintain in the pool
   - `AGGRESSIVE_MEMORY_MANAGEMENT`: Enable memory optimization features

## Deployment

The fix has been deployed using the following steps:

1. Applied the Redis connection patch
2. Restarted the SHAP microservice
3. Verified the Redis connection is working properly

For Render.com deployment, make sure to add the following environment variables:
```
REDIS_TIMEOUT=5
REDIS_SOCKET_KEEPALIVE=true
REDIS_CONNECTION_POOL_SIZE=10
AGGRESSIVE_MEMORY_MANAGEMENT=true
```

## Verification

You can verify the Redis connection is working by running:
```bash
python3 verify_redis_connection.py
```

This script tests:
- Redis ping functionality
- Health endpoint with Redis connection status
- Job submission and processing

## Troubleshooting

If Redis connection issues persist:

1. Check the Redis URL is correct in the environment variables
2. Verify the Redis service is running and accessible
3. Check the logs for any connection errors
4. Try increasing the `REDIS_TIMEOUT` value

## Technical Details

### Redis Connection Patching

We patch the `redis.from_url` function to add better defaults:
```python
def patched_from_url(url, **kwargs):
    # Add better defaults if not specified
    if 'socket_timeout' not in kwargs:
        kwargs['socket_timeout'] = redis_timeout
    if 'socket_keepalive' not in kwargs:
        kwargs['socket_keepalive'] = redis_socket_keepalive
    # ...
    
    # Use connection pooling
    if 'connection_pool' not in kwargs:
        from redis import ConnectionPool
        pool = ConnectionPool.from_url(
            url,
            max_connections=redis_pool_size,
            # ...
        )
        kwargs['connection_pool'] = pool
    
    # Create the Redis client with the enhanced settings
    client = original_from_url(url, **kwargs)
    return client
```

### Redis Queue Enhancement

We also enhance the Redis Queue `enqueue` method with better error handling:
```python
def safe_enqueue(self, *args, **kwargs):
    try:
        return original_enqueue(self, *args, **kwargs)
    except Exception as e:
        # Try to reconnect Redis
        try:
            self.connection.ping()
        except:
            # This will force a new connection on next use
            if hasattr(self.connection, 'connection_pool'):
                self.connection.connection_pool.disconnect()
        
        # Retry once after reconnection attempt
        try:
            return original_enqueue(self, *args, **kwargs)
        except Exception as retry_e:
            raise
```

## Conclusion

These Redis connection improvements should resolve the issues with jobs getting stuck and workers failing. The service should now be more resilient to transient Redis connection issues and provide better error reporting.
