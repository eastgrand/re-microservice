#!/bin/bash

# SHAP Microservice Comprehensive Fix Deployment Script
# This script applies all fixes and deploys to Render
# Date: May 15, 2025
# Updated: May 15, 2025 - Fixed Connection import issue in worker

echo "====================================="
echo "SHAP Microservice Comprehensive Fix"
echo "====================================="

# Check required tools
echo "Checking required tools..."
if ! command -v git &> /dev/null; then
    echo "❌ git is not installed"
    exit 1
else
    echo "✅ git is installed"
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ python3 is not installed"
    exit 1
else
    echo "✅ python3 is installed"
fi

# Make scripts executable
echo "Making scripts executable..."
chmod +x deploy_shap_fix.sh
chmod +x update_render_worker.py
echo "✅ Scripts are now executable"

# Apply SHAP memory fix
echo "Applying SHAP memory fix..."
./deploy_shap_fix.sh
if [ $? -ne 0 ]; then
    echo "❌ Failed to apply SHAP memory fix"
    exit 1
fi
echo "✅ SHAP memory fix applied"

# Update render worker configuration
echo "Updating Render worker configuration..."
python3 update_render_worker.py
if [ $? -ne 0 ]; then
    echo "❌ Failed to update Render worker configuration"
    exit 1
fi
echo "✅ Render worker configuration updated"

# Check for repair_stuck_jobs.py
if [ ! -f "repair_stuck_jobs.py" ]; then
    echo "⚠️ repair_stuck_jobs.py not found, creating minimal version..."
    cat > repair_stuck_jobs.py << 'EOF'
#!/usr/bin/env python3
"""
Repair Stuck Jobs - Minimal Version

This script moves jobs from started state back to queue to be reprocessed.
"""

import os
import sys
import time
import logging
import argparse
import redis
from rq import Queue, Worker
from rq.job import Job
from rq.registry import StartedJobRegistry

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("job-repair")

def get_redis_connection(url=None):
    """Connect to Redis"""
    redis_url = url or os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    
    try:
        conn = redis.from_url(
            redis_url,
            socket_timeout=5,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30
        )
        conn.ping()
        logger.info("✅ Connected to Redis")
        return conn
    except Exception as e:
        logger.error(f"❌ Redis connection error: {str(e)}")
        return None

def repair_stuck_jobs(conn, force=False, timeout=300):
    """Repair jobs stuck in started state"""
    try:
        queue = Queue('shap-jobs', connection=conn)
        started_registry = StartedJobRegistry(queue.name, queue.connection)
        started_job_ids = started_registry.get_job_ids()
        
        if not started_job_ids:
            logger.info("No jobs in started registry")
            return 0
            
        logger.info(f"Found {len(started_job_ids)} jobs in started registry")
        
        # Check active workers
        workers = Worker.all(connection=conn)
        active_jobs = set()
        for worker in workers:
            job = worker.get_current_job()
            if job:
                active_jobs.add(job.id)
        
        # Process each started job
        repaired = 0
        for job_id in started_job_ids:
            try:
                job = Job.fetch(job_id, connection=conn)
                
                # Skip if job is actively being processed (unless forced)
                if job_id in active_jobs and not force:
                    continue
                
                # Check how long the job has been in started state
                if job.started_at:
                    time_in_started = time.time() - job.started_at
                    if force or time_in_started > timeout:
                        # Remove from started registry and requeue
                        started_registry.remove(job)
                        job.set_status('queued')
                        queue.enqueue_job(job)
                        logger.info(f"Repaired job {job_id}")
                        repaired += 1
            except Exception as e:
                logger.error(f"Error processing job {job_id}: {str(e)}")
                
        return repaired
    except Exception as e:
        logger.error(f"Error repairing jobs: {str(e)}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='Repair stuck jobs')
    parser.add_argument('--url', help='Redis URL')
    parser.add_argument('--force', action='store_true', help='Force repair all jobs')
    parser.add_argument('--timeout', type=int, default=300, help='Job timeout in seconds')
    args = parser.parse_args()
    
    conn = get_redis_connection(args.url)
    if not conn:
        return 1
        
    repaired = repair_stuck_jobs(conn, args.force, args.timeout)
    logger.info(f"Repaired {repaired} jobs")
    return 0

if __name__ == "__main__":
    sys.exit(main())
EOF
    chmod +x repair_stuck_jobs.py
    echo "✅ Created repair_stuck_jobs.py"
fi

# Deploy to Render
echo "Preparing for deployment to Render..."

# Check if we have any uncommitted changes
if git status --porcelain | grep -q "M"; then
    echo "⚠️ There are uncommitted changes. Committing them now..."
    git add shap_memory_fix.py deploy_shap_fix.sh update_render_worker.py repair_stuck_jobs.py app.py render.yaml
    git commit -m "Add SHAP memory optimization fix"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to commit changes"
        echo "Please commit the changes manually and then deploy"
        exit 1
    fi
    echo "✅ Changes committed"
fi

# Update deployment script if needed
if [ -f "deploy_to_render.sh" ]; then
    echo "Found deploy_to_render.sh, checking content..."
    if grep -q "shap_memory_fix" deploy_to_render.sh; then
        echo "✅ deploy_to_render.sh already includes memory fix"
    else
        echo "⚠️ Updating deploy_to_render.sh to include memory fix..."
        # Make a backup
        cp deploy_to_render.sh deploy_to_render.sh.memory-bak
        
        # Add our memory fix setup before any server start commands
        sed -i '' '/gunicorn\|python app.py/i \
# Apply SHAP memory optimization fix\
echo "Applying SHAP memory optimization fix..."\
python3 -c "from shap_memory_fix import apply_all_patches; apply_all_patches()"\
echo "✅ Applied SHAP memory optimization fix"\
\
' deploy_to_render.sh
        echo "✅ Updated deploy_to_render.sh"
    fi
else
    echo "⚠️ deploy_to_render.sh not found, creating a minimal version..."
    cat > deploy_to_render.sh << 'EOF'
#!/bin/bash

# Deploy to Render script with SHAP memory optimization
# Date: May 15, 2025

echo "====================================="
echo "Deploying to Render with SHAP memory optimization..."
echo "====================================="

# Apply memory optimization fix
echo "Applying SHAP memory optimization fix..."
python3 -c "from shap_memory_fix import apply_all_patches; apply_all_patches()"
echo "✅ Applied SHAP memory optimization fix"

# Start gunicorn server
echo "Starting server..."
gunicorn app:app --config=gunicorn_config.py
EOF
    chmod +x deploy_to_render.sh
    echo "✅ Created deploy_to_render.sh"
fi

echo "====================================="
echo "Fixing worker issues and deploying changes..."
echo "====================================="

# Ensure the simple_worker.py script is executable
chmod +x ./simple_worker.py

# Make a backup of setup_worker.py
cp -f setup_worker.py "backups/setup_worker.py.bak-$(date +%Y%m%d-%H%M%S)" 2>/dev/null || cp -f setup_worker.py setup_worker.py.bak

# Commit the changes
echo "Committing changes..."
git add render.yaml simple_worker.py setup_worker.py
git commit -m "Fix: Comprehensive worker fix - removed Connection dependency and added simple_worker.py"

# Push to GitHub
echo "Pushing changes to GitHub..."
git push origin main

echo "====================================="
echo "Deployment complete!"
echo "====================================="
echo ""
echo "The following fixes have been applied:"
echo ""
echo "1. Fixed setup_worker.py to remove Connection import dependency"
echo "2. Updated render.yaml to use simple_worker.py as an alternative"
echo ""
echo "Wait for Render.com to complete the deployment (~5 minutes)"
echo "Then verify the fix by checking the worker logs in the Render dashboard"
echo ""
echo "For more information, see SHAP-MEMORY-FIX.md"
