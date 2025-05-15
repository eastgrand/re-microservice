#!/bin/bash

# Manual Redis Connection Fix for SHAP Microservice

echo "===== Manual Redis Connection Fix for SHAP Microservice ====="
echo "This script will apply the Redis connection improvements without requiring the service to be running"

# Check if we're in the SHAP microservice directory
if [ ! -f "app.py" ]; then
  echo "Error: app.py not found in current directory"
  echo "Please run this script from the SHAP microservice directory"
  exit 1
fi

echo "Detected SHAP microservice in current directory"

# Make sure the Redis connection patch file exists
if [ ! -f "redis_connection_patch.py" ]; then
  echo "Redis connection patch file not found!"
  echo "Creating redis_connection_patch.py..."
  
  # Create the patch file manually
  cat > redis_connection_patch.py << 'EOL'
#!/usr/bin/env python3
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
EOL

  echo "✅ Created redis_connection_patch.py"
else
  echo "✅ Redis connection patch file already exists"
fi

# Check if the patch import is in app.py
if ! grep -q "from redis_connection_patch import apply_all_patches" app.py; then
  echo "Adding Redis patch import to app.py..."
  
  # Create a temporary file with the import added
  awk '
  /^import/{
    if (!added) {
      print;
      if (done == 10) { # After roughly 10 import statements
        print "\n# Redis connection patch for better stability";
        print "from redis_connection_patch import apply_all_patches";
        added = 1;
      }
      done++;
    } else {
      print;
    }
  }
  !/^import/{
    if (!added && $0 ~ /^# --- FLASK APP SETUP/) {
      print "\n# Redis connection patch for better stability";
      print "from redis_connection_patch import apply_all_patches";
      added = 1;
      print;
    } else {
      print;
    }
  }
  ' app.py > app.py.new
  
  # Replace original file
  mv app.py.new app.py
  echo "✅ Added Redis patch import to app.py"
else
  echo "✅ Redis patch import already in app.py"
fi

# Check if the patch call is before Redis initialization
if ! grep -q "apply_all_patches()" app.py; then
  echo "Adding Redis patch call to app.py..."
  
  # Find the Redis initialization line
  if grep -q "redis_conn = redis.from_url" app.py; then
    # Add the patch call before Redis initialization
    sed -i.bak 's/redis_conn = redis.from_url/# Apply Redis connection patches for better stability\napply_all_patches()\n\nredis_conn = redis.from_url/' app.py
    echo "✅ Added patch call before Redis initialization"
  else
    # Add the patch call before Flask app definition as fallback
    sed -i.bak 's/app = Flask(__name__)/# Apply Redis connection patches for better stability\napply_all_patches()\n\napp = Flask(__name__)/' app.py
    echo "✅ Added patch call before Flask app definition (fallback)"
  fi
else
  echo "✅ Redis patch call already in app.py"
fi

# Create or update .env file with Redis settings
if [ ! -f ".env" ]; then
  echo "Creating new .env file with Redis settings..."
  cat > .env << 'EOL'
# Redis connection settings
REDIS_TIMEOUT=5
REDIS_SOCKET_KEEPALIVE=true
REDIS_CONNECTION_POOL_SIZE=10
# Memory management
AGGRESSIVE_MEMORY_MANAGEMENT=true
EOL
  echo "✅ Created new .env file with Redis settings"
else
  echo "Adding Redis settings to existing .env file..."
  
  # Check if the settings are already in .env
  if ! grep -q "REDIS_TIMEOUT" .env; then
    echo -e "\n# Redis connection settings" >> .env
    echo "REDIS_TIMEOUT=5" >> .env
    echo "REDIS_SOCKET_KEEPALIVE=true" >> .env
    echo "REDIS_CONNECTION_POOL_SIZE=10" >> .env
    echo "AGGRESSIVE_MEMORY_MANAGEMENT=true" >> .env
    echo "✅ Added Redis settings to .env file"
  else
    echo "✅ Redis settings already in .env file"
  fi
fi

# Create deployment instructions
cat > REDIS-DEPLOYMENT-INSTRUCTIONS.md << 'EOL'
# SHAP Microservice Redis Fix - Deployment Instructions

The Redis connection fixes have been successfully applied to your codebase. Follow these steps to deploy the changes:

## 1. Verify Local Changes

The following files have been modified:
- `redis_connection_patch.py`: Created with enhanced Redis connection handling
- `app.py`: Modified to use the Redis connection patch
- `.env`: Updated with Redis connection settings

## 2. Commit Changes

```bash
git add redis_connection_patch.py app.py .env
git commit -m "Add Redis connection improvements with timeouts and pooling"
git push
```

## 3. Deploy to Render.com

1. Wait for automatic deployment if you have CI/CD enabled
2. Or manually deploy from the Render dashboard

## 4. Add Environment Variables

In the Render dashboard for your service, add these environment variables:

```
REDIS_TIMEOUT=5
REDIS_SOCKET_KEEPALIVE=true
REDIS_CONNECTION_POOL_SIZE=10
AGGRESSIVE_MEMORY_MANAGEMENT=true
```

## 5. Verify Deployment

After deployment, you can test the service endpoints:
- `/ping` - Basic service availability
- `/health` - Check Redis connection status
- `/admin/redis_ping` - Test Redis ping functionality

## Troubleshooting

If you encounter Redis issues after deployment:

1. Check logs in the Render dashboard
2. Verify the Redis URL is correct in the environment variables
3. Try restarting the service
4. Confirm the Redis service is running and accessible

## Next Steps

Consider implementing additional improvements:
- Redis connection monitoring
- Automatic service recovery
- Queue monitoring and cleanup
EOL

echo "✅ Created deployment instructions in REDIS-DEPLOYMENT-INSTRUCTIONS.md"

echo
echo "===== Redis Connection Fix Complete ====="
echo "The following changes have been made:"
echo "1. Added redis_connection_patch.py with improved Redis connection handling"
echo "2. Applied the patch to app.py"
echo "3. Added Redis connection settings to .env"
echo "4. Created deployment instructions in REDIS-DEPLOYMENT-INSTRUCTIONS.md"
echo
echo "Next steps:"
echo "1. Commit and push these changes to your repository"
echo "2. Follow the deployment instructions in REDIS-DEPLOYMENT-INSTRUCTIONS.md"
echo
echo "In the Render dashboard, add these environment variables:"
echo "REDIS_TIMEOUT=5"
echo "REDIS_SOCKET_KEEPALIVE=true" 
echo "REDIS_CONNECTION_POOL_SIZE=10"
echo "AGGRESSIVE_MEMORY_MANAGEMENT=true"
