#!/usr/bin/env python3
"""
Redis Connection Patch Generator for SHAP Microservice

This script generates a patch file that can be applied to your SHAP microservice
to fix Redis connection issues. The patch improves Redis connection handling,
adds proper timeouts, and implements connection pooling.

Usage:
  python generate_redis_patch.py

Output:
  redis_connection_patch.py - A patch file that can be applied to your SHAP microservice
"""

import os
import datetime

# Configuration
PATCH_FILE = "redis_connection_patch.py"
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# The patch content with improved Redis connection handling
PATCH_CONTENT = f"""#!/usr/bin/env python3
'''
SHAP Microservice Redis Connection Patch
Generated: {TIMESTAMP}

This patch improves Redis connection handling in the SHAP microservice by:
1. Adding connection pooling
2. Setting proper timeouts
3. Implementing automatic reconnection
4. Adding health check endpoints for Redis

Apply this patch by copying it to your SHAP microservice directory and importing it
in your main app.py file: `from redis_connection_patch import patch_redis_connection`
Then call `patch_redis_connection()` before creating your Redis connection.
'''

import os
import time
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("redis-patch")

def patch_redis_connection():
    """Patch the Redis connection handling in the SHAP microservice"""
    import redis
    from redis.exceptions import ConnectionError, TimeoutError
    
    # Store the original from_url method
    original_from_url = redis.from_url
    
    # Get Redis configuration from environment or use defaults
    redis_timeout = int(os.environ.get('REDIS_TIMEOUT', '5'))
    redis_socket_keepalive = os.environ.get('REDIS_SOCKET_KEEPALIVE', 'true').lower() == 'true'
    redis_pool_size = int(os.environ.get('REDIS_CONNECTION_POOL_SIZE', '10'))
    
    logger.info(f"Patching Redis with: timeout={redis_timeout}s, keepalive={redis_socket_keepalive}, pool_size={redis_pool_size}")
    
    @wraps(original_from_url)
    def patched_from_url(url, **kwargs):
        """
        Enhanced version of redis.from_url with better defaults and error handling
        """
        # Add better defaults if not specified
        if 'socket_timeout' not in kwargs:
            kwargs['socket_timeout'] = redis_timeout
        if 'socket_keepalive' not in kwargs:
            kwargs['socket_keepalive'] = redis_socket_keepalive
        if 'socket_connect_timeout' not in kwargs:
            kwargs['socket_connect_timeout'] = redis_timeout
        if 'health_check_interval' not in kwargs:
            kwargs['health_check_interval'] = 30
        if 'retry_on_timeout' not in kwargs:
            kwargs['retry_on_timeout'] = True
            
        # Use connection pooling
        if 'connection_pool' not in kwargs:
            from redis import ConnectionPool
            pool = ConnectionPool.from_url(
                url,
                max_connections=redis_pool_size,
                socket_timeout=redis_timeout,
                socket_keepalive=redis_socket_keepalive,
                socket_connect_timeout=redis_timeout,
                health_check_interval=30,
                retry_on_timeout=True
            )
            kwargs['connection_pool'] = pool
        
        # Create the Redis client with the enhanced settings
        client = original_from_url(url, **kwargs)
        logger.info(f"Created enhanced Redis client with improved connection handling")
        return client
    
    # Replace the original from_url with our patched version
    redis.from_url = patched_from_url
    logger.info("Redis connection handling has been patched")
    
    # Add Redis health check endpoint to Flask app
    try:
        from flask import current_app, jsonify
        
        @current_app.route('/admin/redis_ping', methods=['GET'])
        def redis_ping():
            """Test Redis connection with PING command"""
            from flask import current_app
            try:
                # Get the Redis connection from the app
                redis_conn = current_app.config.get('redis_conn')
                if not redis_conn:
                    # Fall back to global redis_conn if available
                    global redis_conn
                    if 'redis_conn' not in globals():
                        return jsonify({
                            "success": False,
                            "error": "Redis connection not found in app config"
                        }), 500
                
                # Try to ping Redis
                start_time = time.time()
                result = redis_conn.ping()
                ping_time = time.time() - start_time
                
                # Return success
                return jsonify({
                    "success": True,
                    "ping": result,
                    "response_time_ms": round(ping_time * 1000, 2)
                })
            except Exception as e:
                logger.error(f"Redis ping failed: {str(e)}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
        
        logger.info("Added Redis health check endpoint: /admin/redis_ping")
    except Exception as e:
        logger.error(f"Could not add Redis health check endpoint: {str(e)}")
    
    return True

def wrap_redis_queue_functions():
    """
    Wrap Redis Queue functions with better error handling
    """
    try:
        import rq
        from rq.queue import Queue
        
        # Store original enqueue method
        original_enqueue = Queue.enqueue
        
        @wraps(original_enqueue)
        def safe_enqueue(self, *args, **kwargs):
            """
            Enhanced version of Queue.enqueue with better error handling
            """
            try:
                return original_enqueue(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Redis Queue enqueue error: {str(e)}")
                # Try to reconnect Redis
                try:
                    self.connection.ping()
                except:
                    logger.info("Attempting to reconnect to Redis...")
                    # This will force a new connection on next use
                    if hasattr(self.connection, 'connection_pool'):
                        self.connection.connection_pool.disconnect()
                
                # Retry once after reconnection attempt
                try:
                    return original_enqueue(self, *args, **kwargs)
                except Exception as retry_e:
                    logger.error(f"Redis Queue enqueue retry failed: {str(retry_e)}")
                    raise
        
        # Replace with safer version
        Queue.enqueue = safe_enqueue
        logger.info("RQ Queue functions wrapped with better error handling")
    except ImportError:
        logger.warning("Could not patch RQ functions - module not found")
    except Exception as e:
        logger.error(f"Error patching RQ functions: {str(e)}")

def apply_all_patches():
    """
    Apply all Redis-related patches
    """
    patch_redis_connection()
    wrap_redis_queue_functions()
    logger.info("All Redis patches applied successfully")
    
if __name__ == "__main__":
    print("This is a patch module for the SHAP microservice Redis connection handling.")
    print("To use, import this module in your main app.py file and call apply_all_patches()")
"""

# Write the patch file
with open(PATCH_FILE, "w") as f:
    f.write(PATCH_CONTENT)

print(f"Redis connection patch generated as: {PATCH_FILE}")
print("To apply this patch to your SHAP microservice:")
print(f"1. Copy {PATCH_FILE} to your SHAP microservice directory")
print("2. Add the following to the top of your app.py file:")
print("   from redis_connection_patch import apply_all_patches")
print("3. Call apply_all_patches() before creating your Redis connection:")
print("   apply_all_patches()")
print("4. Redeploy your SHAP microservice with these changes")
