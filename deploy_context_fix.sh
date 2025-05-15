#!/bin/bash
# deploy_context_fix.sh - Deploy the Flask context fix to Render
# Created: May 15, 2025

echo "🔄 Deploying Flask Context Fix for Redis Health Check..."

# Ensure the verification script is executable
chmod +x verify_redis_endpoint.sh

# Add all modified files
git add redis_connection_patch.py app.py REDIS-FLASK-CONTEXT-FIX.md verify_redis_endpoint.sh test_redis_context.py

# Commit changes
git commit -m "Fix Redis health check Flask context issue"

# Push to deploy
git push origin main

# Wait for the deployment to complete (adjust time as needed)
echo "⏱️ Waiting for deployment to complete (60 seconds)..."
sleep 60

# Run verification after deployment
echo "🔍 Verifying Redis health check endpoint..."
./verify_redis_endpoint.sh "${SHAP_SERVICE_URL:-https://nesto-mortgage-analytics.onrender.com}"

echo "✅ Deployment and verification complete!"
