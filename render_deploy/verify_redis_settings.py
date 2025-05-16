#!/usr/bin/env python3
'''
Redis Connection Verification Script
- Verifies the Redis connection settings and status
- Checks if patches have been applied correctly
'''

import os
import sys
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("redis-verify")

def verify_redis_connection():
    '''Verify Redis connection settings and status'''
    try:
        # Import redis and patching module
        import redis
        import redis_connection_patch
        
        # Get Redis URL from environment or use default
        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        
        # Print current environment configuration
        logger.info("=== Redis Environment Configuration ===")
        logger.info(f"REDIS_URL: {redis_url}")
        logger.info(f"REDIS_TIMEOUT: {os.environ.get('REDIS_TIMEOUT', '5')} seconds")
        logger.info(f"REDIS_SOCKET_KEEPALIVE: {os.environ.get('REDIS_SOCKET_KEEPALIVE', 'true')}")
        logger.info(f"REDIS_CONNECTION_POOL_SIZE: {os.environ.get('REDIS_CONNECTION_POOL_SIZE', '10')}")
        
        # Apply patches
        logger.info("Applying Redis connection patches...")
        redis_connection_patch.apply_all_patches()
        
        # Create Redis connection
        logger.info("Creating Redis connection...")
        start_time = time.time()
        redis_conn = redis.from_url(redis_url)
        conn_time = time.time() - start_time
        logger.info(f"Connection established in {conn_time:.2f} seconds")
        
        # Verify if connection is working with PING
        logger.info("Testing connection with PING...")
        start_time = time.time()
        ping_result = redis_conn.ping()
        ping_time = time.time() - start_time
        logger.info(f"PING successful: {ping_result} (took {ping_time:.2f}s)")
        
        # Check connection pool info
        if hasattr(redis_conn, 'connection_pool'):
            pool = redis_conn.connection_pool
            logger.info("=== Redis Connection Pool Info ===")
            logger.info(f"Pool size: {pool.max_connections}")
            logger.info(f"Current connections: {len(pool._connections)}")
            
            # Check if timeout settings were applied
            conn_kwargs = {}
            if hasattr(pool, 'connection_kwargs'):
                conn_kwargs = pool.connection_kwargs
            elif hasattr(pool, 'connection_class'):
                # Find connection class kwargs
                conn_kwargs = {key: getattr(pool, key) for key in dir(pool) if not key.startswith('_') and not callable(getattr(pool, key))}
            
            logger.info("=== Redis Connection Settings ===")
            for key, value in conn_kwargs.items():
                if key in ['socket_timeout', 'socket_connect_timeout', 'socket_keepalive', 
                          'health_check_interval', 'retry_on_timeout']:
                    logger.info(f"{key}: {value}")
            
        # Verify patch functions
        logger.info("=== Patch Verification ===")
        logger.info(f"Redis from_url patched: {redis.from_url != redis._original_from_url if hasattr(redis, '_original_from_url') else 'Unknown'}")
        logger.info(f"Failsafe ping method added: {redis.Redis.ping != redis.Redis._original_ping if hasattr(redis.Redis, '_original_ping') else 'Unknown'}")
        
        # Try to import RQ and verify if patched
        try:
            import rq
            from rq.queue import Queue
            logger.info(f"RQ Queue enqueue patched: {Queue.enqueue != Queue._original_enqueue if hasattr(Queue, '_original_enqueue') else 'Unknown'}")
        except ImportError:
            logger.info("RQ module not available for verification")
            
        logger.info("=== Verification Complete ===")
        logger.info("Redis connection patch appears to be working correctly")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    print("=== Redis Connection Patch Verification ===")
    success = verify_redis_connection()
    sys.exit(0 if success else 1)
