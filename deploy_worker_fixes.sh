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
