#!/bin/bash
# Script to push changes to GitHub and trigger a Render deployment

echo "Pushing fixes to GitHub..."
git add .
git commit -m "Fix categorical data handling and API endpoints"
git push

echo "Changes pushed to GitHub. Render should automatically start a new deployment."
echo "Monitor the deployment at: https://dashboard.render.com"

echo "Once deployed, test the API with:"
echo "curl https://xgboost-qeb6.onrender.com/ping"
echo "curl -H 'X-API-KEY: \$HFqkccbN3LV5CaB' https://xgboost-qeb6.onrender.com/health"
