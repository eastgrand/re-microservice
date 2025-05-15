#!/usr/bin/env python3
"""
Redis Connection Fix for SHAP Microservice

This script directly patches the Redis connection handling in your app.py file
to improve connection stability and error handling.
"""

import os
import shutil
import datetime

def backup_app_file():
    """Create a backup of the app.py file"""
    shutil.copy("app.py", f"app.py.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
    print("✅ Created backup of app.py")

def create_redis_patch_file():
    """Create the Redis connection patch file"""
    patch_content = """#!/usr/bin/env python3
'''
SHAP Microservice Redis Connection Patch
'''

import os
import time
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("redis-patch")

def patch_redis_connection():
    '''Patch the Redis connection handling in the SHAP microservice'''
    import redis
    import redis.exceptions
    
    # Store the original from_url method
    original_from_url = redis.from_url
    
    # Get Redis configuration from environment or use defaults
    redis_timeout = int(os.environ.get('REDIS_TIMEOUT', '5'))
    redis_socket_keepalive = os.environ.get('REDIS_SOCKET_KEEPALIVE', 'true').lower() == 'true'
    redis_pool_size = int(os.environ.get('REDIS_CONNECTION_POOL_SIZE', '10'))
    
    logger.info(f"Patching Redis with: timeout={redis_timeout}s, keepalive={redis_socket_keepalive}, pool_size={redis_pool_size}")
    
    @wraps(original_from_url)
    def patched_from_url(url, **kwargs):
        '''Enhanced version of redis.from_url with better defaults and error handling'''
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
            '''Test Redis connection with PING command'''
            try:
                # Get the Redis connection from the app
                redis_conn = current_app.config.get('redis_conn')
                if not redis_conn:
                    # Fall back to global redis_conn if available
                    import sys
                    this_module = sys.modules[__name__]
                    if hasattr(this_module, 'redis_conn'):
                        redis_conn = this_module.redis_conn
                    else:
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
    '''Wrap Redis Queue functions with better error handling'''
    try:
        import rq
        from rq.queue import Queue
        
        # Store original enqueue method
        original_enqueue = Queue.enqueue
        
        @wraps(original_enqueue)
        def safe_enqueue(self, *args, **kwargs):
            '''Enhanced version of Queue.enqueue with better error handling'''
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
    '''Apply all Redis-related patches'''
    patch_redis_connection()
    wrap_redis_queue_functions()
    logger.info("All Redis patches applied successfully")
    
if __name__ == "__main__":
    print("This is a patch module for the SHAP microservice Redis connection handling.")
    print("To use, import this module in your main app.py file and call apply_all_patches()")
"""
    
    with open("redis_connection_patch.py", "w") as f:
        f.write(patch_content)
    
    print("✅ Created redis_connection_patch.py with improved Redis handling")
    return True

def update_app_py():
    """Update app.py to use the Redis patch"""
    with open("app.py", "r") as f:
        app_content = f.read()
    
    # Add import at the top
    import_statement = "from redis_connection_patch import apply_all_patches\n"
    if import_statement not in app_content:
        # Find the end of the import section
        import_section_end = app_content.find("# --- FLASK APP SETUP")
        if import_section_end > 0:
            # Insert the import statement before the Flask app setup
            updated_content = (
                app_content[:import_section_end] + 
                "\n# Redis connection patch for better stability\n" + 
                import_statement + 
                app_content[import_section_end:]
            )
        else:
            # Fallback: just add it at the beginning
            updated_content = import_statement + app_content
            
        # Write the updated content back
        with open("app.py", "w") as f:
            f.write(updated_content)
        print("✅ Added import to app.py")
    else:
        print("✓ Import already exists in app.py")
    
    # Add the patch call before Redis connection initialization
    patch_call = "# Apply Redis connection patches for better stability\napply_all_patches()\n"
    
    with open("app.py", "r") as f:
        app_content = f.read()
    
    if "apply_all_patches()" not in app_content:
        # Find the Redis initialization line
        redis_init_index = app_content.find("redis_conn = redis.from_url(REDIS_URL)")
        if redis_init_index > 0:
            # Insert the patch call before the Redis initialization
            updated_content = (
                app_content[:redis_init_index] + 
                patch_call + 
                app_content[redis_init_index:]
            )
            
            # Write the updated content back
            with open("app.py", "w") as f:
                f.write(updated_content)
            print("✅ Added patch call to app.py before Redis initialization")
        else:
            print("⚠️ Could not find Redis initialization in app.py")
    else:
        print("✓ Patch call already exists in app.py")
    
    return True

def create_env_file():
    """Create or update .env file with Redis settings"""
    env_content = """
# Redis connection settings
REDIS_TIMEOUT=5
REDIS_SOCKET_KEEPALIVE=true
REDIS_CONNECTION_POOL_SIZE=10
# Memory management
AGGRESSIVE_MEMORY_MANAGEMENT=true
"""
    
    if os.path.exists(".env"):
        # Append to existing .env file
        with open(".env", "a") as f:
            f.write(env_content)
        print("✅ Added Redis settings to existing .env file")
    else:
        # Create new .env file
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created new .env file with Redis settings")
    
    return True

def main():
    """Main function"""
    print("===== SHAP Microservice Redis Connection Fix =====")
    print("This script will fix Redis connection issues in your SHAP microservice")
    print()
    
    # Check if we're in the SHAP microservice directory
    if not os.path.exists("app.py"):
        print("Error: app.py not found in current directory")
        print("Please run this script from the SHAP microservice directory")
        return False
    
    print("Detected SHAP microservice in current directory")
    
    # Create a backup
    backup_app_file()
    
    # Create the Redis patch file
    create_redis_patch_file()
    
    # Update app.py
    update_app_py()
    
    # Create .env file with Redis settings
    create_env_file()
    
    print()
    print("===== Redis Connection Fix Complete =====")
    print("The following changes have been made:")
    print("1. Added redis_connection_patch.py with improved Redis connection handling")
    print("2. Applied the patch to app.py")
    print("3. Added Redis connection settings to .env")
    print()
    print("Next steps:")
    print("1. Commit and push these changes to your repository")
    print("2. Redeploy the SHAP microservice in Render")
    print("3. Monitor for Redis connection issues")
    print()
    print("In the Render dashboard, add these environment variables:")
    print("REDIS_TIMEOUT=5")
    print("REDIS_SOCKET_KEEPALIVE=true") 
    print("REDIS_CONNECTION_POOL_SIZE=10")
    print("AGGRESSIVE_MEMORY_MANAGEMENT=true")
    print()
    
    return True

if __name__ == "__main__":
    main()
