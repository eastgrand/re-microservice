#!/bin/bash
# Script to deploy and verify the port binding fix

# Step 1: Commit and push changes
echo "Committing and pushing port binding fixes..."
git add .
git commit -m "Fix port binding issues for Render deployment"
git push

echo "Changes pushed to GitHub. Render should automatically start a new deployment."
echo "Monitor the deployment at: https://dashboard.render.com"

echo
echo "After deployment completes, verify with these commands:"
echo "---------------------------------------------------"
echo "1. Basic connectivity test:"
echo "   curl https://xgboost-qeb6.onrender.com/ping"
echo
echo "2. Health check with API key:"
echo "   curl -H 'X-API-KEY: \$HFqkccbN3LV5CaB' https://xgboost-qeb6.onrender.com/health"
echo
echo "If these commands fail, check the Render logs for port binding issues."
