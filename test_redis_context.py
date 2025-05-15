#!/usr/bin/env python3
"""
Test Redis Context Fix

This is a simple Flask app to verify the fixes made to the Redis connection patch
to ensure it works properly with Flask application contexts.
"""

import os
import sys
import time
import logging
from flask import Flask, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("redis-test")

def test_redis_context_fix():
    """Test if the Redis patch works with Flask contexts"""
    
    # Create a test Flask app
    app = Flask(__name__)
    
    # Import our Redis connection patch
    from redis_connection_patch import apply_all_patches
    
    logger.info("Creating Flask app and applying Redis patches...")
    
    # Apply with app context
    with app.app_context():
        # Apply Redis patches
        apply_all_patches(app)
        
        # Setup Redis connection
        import redis
        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        redis_conn = redis.from_url(redis_url)
        app.config['redis_conn'] = redis_conn
    
    # Test if the Redis endpoint got registered
    endpoints = [rule.endpoint for rule in app.url_map.iter_rules()]
    
    # Check if our endpoint is registered
    if 'redis_ping' in endpoints:
        logger.info("✅ Success! Redis ping endpoint was registered properly")
        return True
    else:
        logger.error("❌ Redis ping endpoint was not registered")
        logger.info(f"Available endpoints: {', '.join(endpoints)}")
        return False

if __name__ == "__main__":
    result = test_redis_context_fix()
    sys.exit(0 if result else 1)
