#!/bin/bash
# deploy_worker_fix.sh - Deploy the worker process fix to Render

set -e

echo "Preparing to deploy worker process fix to Render..."

# Make all scripts executable
chmod +x repair_stuck_jobs.py diagnose_worker.py worker_process_fix.py

# Make sure rq is installed
pip3 install rq --break-system-packages

# First, run diagnostics to understand the current state
echo ""
echo "🔍 Running worker diagnostics..."
python3 diagnose_worker.py

# Update app.py to use the worker fix
echo ""
echo "📝 Updating app.py to apply worker process fixes..."

if grep -q "worker_process_fix" app.py; then
  echo "Worker process fix is already imported in app.py"
else
  # Make a backup of app.py
  cp app.py app.py.bak
  
  # Update the import section
  sed -i '' '/from redis_connection_patch import/a\
from worker_process_fix import apply_all_worker_patches  # Added worker process fixes
' app.py
  
  # Update the application context section to use the worker fix
  sed -i '' 's/apply_all_patches(app)/apply_all_worker_patches(app)/' app.py
  
  echo "✅ Updated app.py to use worker process fixes"
fi

# Update render.yaml to use a better worker start command
echo ""
echo "📝 Updating render.yaml with worker improvements..."

if grep -q "cleanup_stale_jobs" render.yaml; then
  echo "Worker improvements already in render.yaml"
else
  # Make a backup of render.yaml
  cp render.yaml render.yaml.bak
  
  # Update the worker startCommand to run cleanup before starting
  sed -i '' 's/startCommand: rq worker shap-jobs --url $REDIS_URL/startCommand: >-\
    echo "Starting worker with automatic job monitoring" \&\&\
    python3 -c "from worker_process_fix import apply_all_worker_patches; apply_all_worker_patches()" \&\&\
    python3 repair_stuck_jobs.py --force \&\&\
    rq worker --burst shap-jobs --url $REDIS_URL/' render.yaml
  
  echo "✅ Updated render.yaml with worker improvements"
fi

# Create a script to deploy to Render
cat > deploy_worker_fixes.sh << 'EOL'
#!/bin/bash
# deploy_worker_fixes.sh - Deploy worker fixes to Render

echo "Deploying worker process fixes to Render..."
git add worker_process_fix.py repair_stuck_jobs.py diagnose_worker.py app.py render.yaml
git commit -m "Add worker process monitoring and stuck job handling"
git push

# Open the Render dashboard
echo ""
echo "✅ Changes pushed to repository"
echo ""
echo "Next steps:"
echo "1. Go to the Render dashboard and deploy the latest changes"
echo "2. Monitor the logs to ensure the worker is starting correctly"
echo "3. Test job processing with the test_deployment.py script"
echo ""
echo "If needed, manually trigger cleanup by calling POST /admin/cleanup_jobs"
EOL

chmod +x deploy_worker_fixes.sh

echo ""
echo "✅ Created deployment script: deploy_worker_fixes.sh"
echo ""
echo "Preview of changes:"
echo "1. Added worker process monitoring and job timeout handling"
echo "2. Created diagnostic tools for worker issues"
echo "3. Added endpoint to cleanup stuck jobs: /admin/cleanup_jobs"
echo "4. Created repair script for any jobs stuck in 'started' state"
echo ""
echo "To deploy the changes, run: ./deploy_worker_fixes.sh"
