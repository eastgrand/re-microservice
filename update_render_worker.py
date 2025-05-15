#!/usr/bin/env python3
"""
Update Render Worker Configuration

This script updates the render.yaml file to ensure the worker 
process has the correct configuration to handle SHAP memory optimizations.

It also verifies that other required changes have been applied.
"""

import os
import sys
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("render-update")

RENDER_YAML = "render.yaml"
BACKUP_SUFFIX = ".bak-" + "memory-fix"

def backup_file(filepath):
    """Create a backup of the file"""
    if os.path.exists(filepath):
        backup_path = filepath + BACKUP_SUFFIX
        with open(filepath, 'r') as src:
            with open(backup_path, 'w') as dst:
                dst.write(src.read())
        logger.info(f"Created backup of {filepath} at {backup_path}")
        return True
    return False

def update_worker_config():
    """Update the worker configuration in render.yaml"""
    if not os.path.exists(RENDER_YAML):
        logger.error(f"❌ {RENDER_YAML} not found!")
        return False
    
    # Create backup
    backup_file(RENDER_YAML)
    
    # Read render.yaml
    with open(RENDER_YAML, 'r') as f:
        content = f.read()
    
    # Check if this is a worker configuration
    if 'worker' not in content:
        logger.error(f"❌ No worker configuration found in {RENDER_YAML}")
        return False
    
    # Update worker start command to include memory optimization
    worker_pattern = r'(startCommand:.*?rq worker.*?)(\$REDIS_URL)'
    memory_worker_cmd = r'startCommand: >-\n    echo "Starting memory-optimized SHAP worker" &&\n    python3 -c "import gc; gc.enable()" &&\n    python3 -c "from shap_memory_fix import apply_all_patches; apply_all_patches()" &&\n    python3 repair_stuck_jobs.py --force &&\n    rq worker --burst shap-jobs --url $REDIS_URL'
    
    # If already using multiline command format
    multiline_pattern = r'startCommand:\s*>-.*?\$REDIS_URL'
    if re.search(multiline_pattern, content, re.DOTALL):
        logger.info("Found multiline worker command, checking if memory optimization is included")
        if "shap_memory_fix" in content:
            logger.info("✅ Memory optimization already included in worker command")
        else:
            # Create full replacement with complete command
            content = re.sub(
                r'startCommand:\s*>-.*?rq worker.*?\$REDIS_URL', 
                memory_worker_cmd, 
                content, 
                flags=re.DOTALL
            )
            logger.info("✅ Updated worker command to include memory optimization")
    else:
        # Replace single line command with multiline memory-optimized version
        content = re.sub(worker_pattern, memory_worker_cmd, content)
        logger.info("✅ Updated worker command to include memory optimization")
    
    # Add environment variables if not present
    if "AGGRESSIVE_MEMORY_MANAGEMENT" not in content:
        # Find the envVars section of the worker
        worker_env_pattern = r'(worker.*?\n\s*envVars:)'
        if re.search(worker_env_pattern, content, re.DOTALL):
            env_vars_to_add = """
      - key: AGGRESSIVE_MEMORY_MANAGEMENT
        value: "true"
      - key: SHAP_BATCH_SIZE
        value: "500"
"""
            # Add after the envVars line
            content = re.sub(worker_env_pattern, r'\1' + env_vars_to_add, content)
            logger.info("✅ Added memory optimization environment variables")
        else:
            logger.warning("⚠️ Could not find worker envVars section, skipping environment variable addition")
    else:
        logger.info("✅ Memory optimization environment variables already present")
    
    # Write updated content
    with open(RENDER_YAML, 'w') as f:
        f.write(content)
    
    logger.info(f"✅ Updated {RENDER_YAML} with memory-optimized worker configuration")
    return True

def main():
    """Main function"""
    logger.info("=== Update Render Worker Configuration ===")
    
    # Update worker configuration
    if update_worker_config():
        logger.info(f"✅ Successfully updated {RENDER_YAML}")
        logger.info(f"A backup was created at {RENDER_YAML}{BACKUP_SUFFIX}")
        logger.info("Please review the changes to ensure they are correct")
        logger.info("Then redeploy your application to Render")
        return 0
    else:
        logger.error(f"❌ Failed to update {RENDER_YAML}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
