#!/bin/bash
# Memory Optimization Deployment Script for SHAP Microservice
# Created: May 15, 2025
# This script deploys the memory optimization changes to improve performance
# while staying within the constraints of the Render starter plan.

echo "========================================================"
echo "    SHAP Microservice Memory Optimization Deployment    "
echo "========================================================"
echo ""

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "❌ Error: Git is not installed or not in PATH"
    exit 1
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️ You have uncommitted changes. Committing them first..."
    git add render.yaml optimize_memory.py
    git commit -m "Optimize memory settings for better performance on Starter plan"
    echo "✅ Changes committed"
else
    echo "✅ No uncommitted changes"
fi

echo ""
echo "Summary of optimizations applied:"
echo "--------------------------------"
echo "1. Increased SHAP_MAX_BATCH_SIZE from 300 to 500 rows"
echo "2. Disabled aggressive memory management"
echo "3. Increased memory threshold from 450MB to 475MB"
echo ""
echo "These changes should allow processing of larger data batches"
echo "while still maintaining a safe buffer below the 512MB limit."
echo ""

# Push changes to remote repository
echo "Pushing changes to remote repository..."
git push origin main

echo ""
echo "✅ Changes pushed to GitHub. Render will automatically deploy the updates."
echo ""
echo "Next steps:"
echo "1. Wait for Render to complete the deployment (3-5 minutes)"
echo "2. Monitor the worker logs to verify the new settings"
echo "3. Check job processing times to confirm performance improvement"
echo ""
echo "To run a health check after deployment:"
echo "python3 comprehensive_health_check.py --api-key YOUR_API_KEY"
echo ""
echo "========================================================"
