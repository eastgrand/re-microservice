#!/usr/bin/env python3
# app_fix_wrapper.py - Apply NaN and worker fixes to the Flask app
# Created: May 16, 2025

import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app-fix-wrapper")

def apply_all_fixes_to_app(app):
    """Apply all fixes to the Flask app"""
    try:
        # First apply JSON serialization fixes
        from json_serialization_fix import apply_json_patches
        apply_json_patches()
        logger.info("Applied JSON serialization fixes to the Flask app")
        
        # Then apply worker fixes
        from analysis_worker_fix import apply_all_worker_fixes
        apply_all_worker_fixes(app)
        logger.info("Applied worker fixes to the Flask app")
        
        # Finally apply redis connection patches
        from redis_connection_patch import apply_all_patches
        apply_all_patches(app)
        logger.info("Applied Redis connection patches to the Flask app")
        
        logger.info("✅ All fixes applied to Flask app successfully")
        return True
    except Exception as e:
        logger.error(f"Error applying fixes to Flask app: {str(e)}")
        return False

# Auto-patch the app if this is imported
if __name__ != "__main__":
    logger.info("app_fix_wrapper loaded - will patch Flask app when imported")
else:
    print("This is a fix wrapper for the SHAP microservice Flask app.")
    print("To use, import this module in your main app.py file and call apply_all_fixes_to_app(app)")
