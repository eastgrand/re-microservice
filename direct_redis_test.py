#!/usr/bin/env python3
"""
Direct Redis Connection Test

This script tests the Redis connection directly using the redis-py client
without going through the SHAP microservice API.
"""

import os
import sys
import time

# Try to import Redis
try:
    import redis
    print("Successfully imported redis module")
except ImportError:
    print("Failed to import redis module. Please install it with: pip install redis")
    sys.exit(1)

def test_direct_redis_connection():
    """Test Redis connection directly"""
    # Get Redis URL from environment or use default
    redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    print(f"Testing direct Redis connection to: {redis_url}")
    
    try:
        # Create Redis client
        r = redis.from_url(
            redis_url,
            socket_timeout=5,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30,
            retry_on_timeout=True
        )
        
        # Test with PING command
        start_time = time.time()
        result = r.ping()
        ping_time = time.time() - start_time
        
        print(f"✅ Redis PING successful: {result}")
        print(f"Response time: {ping_time*1000:.2f} ms")
        
        # Test basic operations
        print("Testing basic Redis operations...")
        
        # Set a test key
        r.set('test_key', 'test_value')
        print("✅ Successfully set a test key")
        
        # Get the test key
        value = r.get('test_key')
        print(f"✅ Successfully retrieved the test key: {value}")
        
        # Delete the test key
        r.delete('test_key')
        print("✅ Successfully deleted the test key")
        
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {str(e)}")
        return False

def check_redis_info():
    """Get Redis server information"""
    # Get Redis URL from environment or use default
    redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    try:
        # Create Redis client
        r = redis.from_url(redis_url, socket_timeout=5)
        
        # Get server info
        info = r.info()
        
        print("\nRedis Server Information:")
        print(f"- Redis version: {info.get('redis_version', 'Unknown')}")
        print(f"- Uptime: {info.get('uptime_in_seconds', 0)} seconds")
        print(f"- Connected clients: {info.get('connected_clients', 0)}")
        print(f"- Used memory: {info.get('used_memory_human', 'Unknown')}")
        print(f"- Total connections received: {info.get('total_connections_received', 0)}")
        print(f"- Total commands processed: {info.get('total_commands_processed', 0)}")
        
        # Check for any warning flags
        used_memory = info.get('used_memory', 0)
        max_memory = info.get('maxmemory', 0)
        if max_memory > 0 and used_memory > max_memory * 0.8:
            print("⚠️ Warning: Redis memory usage is high!")
        
        return True
    except Exception as e:
        print(f"❌ Could not retrieve Redis information: {str(e)}")
        return False

if __name__ == "__main__":
    print("===== Direct Redis Connection Test =====")
    success = test_direct_redis_connection()
    
    if success:
        # If connection works, get additional info
        check_redis_info()
        print("\n✅ Direct Redis connection test PASSED")
        sys.exit(0)
    else:
        print("\n❌ Direct Redis connection test FAILED")
        sys.exit(1)
