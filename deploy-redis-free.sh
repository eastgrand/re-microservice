#!/bin/bash

# Deploy Redis-Free SHAP Microservice
echo "🚀 Deploying Redis-Free SHAP Microservice"
echo "=========================================="

# Create backup of current files
echo "📁 Creating backup of current files..."
cp app.py app_backup_$(date +%Y%m%d_%H%M%S).py
if [ -f requirements.txt ]; then
    cp requirements.txt requirements_backup_$(date +%Y%m%d_%H%M%S).txt
fi

# Replace with simplified versions
echo "🔄 Replacing with Redis-free versions..."
cp app_simplified.py app.py
cp requirements_simplified.txt requirements.txt

echo "✅ Files replaced:"
echo "  - app.py → Redis-free synchronous version"
echo "  - requirements.txt → Simplified dependencies"

# Remove Redis-related files
echo "🗑️  Removing Redis-related files..."
if [ -f redis_connection_patch.py ]; then
    mv redis_connection_patch.py redis_connection_patch.py.disabled
    echo "  - Disabled redis_connection_patch.py"
fi

if [ -f async_job_manager.py ]; then
    mv async_job_manager.py async_job_manager.py.disabled
    echo "  - Disabled async_job_manager.py"
fi

# Update render.yaml to remove Redis URL
if [ -f render.yaml ]; then
    echo "🔧 Updating render.yaml..."
    # Create backup
    cp render.yaml render_backup_$(date +%Y%m%d_%H%M%S).yaml
    
    # Remove Redis URL line (commented out approach for safety)
    sed 's/.*REDIS_URL.*/# &/' render.yaml > render_temp.yaml && mv render_temp.yaml render.yaml
    echo "  - Commented out REDIS_URL in render.yaml"
fi

# Git operations
echo "📦 Committing changes..."
git add app.py requirements.txt
git add -A  # Add any other changed files

# Commit with descriptive message
git commit -m "Deploy Redis-free SHAP microservice

- Remove all Redis dependencies and async processing
- Convert to synchronous dataset generation
- Eliminate Redis timeout issues
- Simplify architecture for dataset generation use case
- Keep memory optimization features

Breaking changes:
- /analyze now returns results directly (no job polling)
- Removed /job_status/<id> endpoint
- Removed Redis caching layer"

echo "🚢 Pushing to repository..."
git push

echo ""
echo "🎯 Deployment Summary:"
echo "======================"
echo "✅ Redis dependencies removed"
echo "✅ Converted to synchronous processing"
echo "✅ Simplified requirements.txt"
echo "✅ Updated configuration files"
echo "✅ Changes committed and pushed"
echo ""
echo "📋 Next Steps:"
echo "1. Wait for Render to auto-deploy (2-3 minutes)"
echo "2. Test endpoints with our test scripts"
echo "3. Verify no more Redis timeout errors"
echo ""
echo "🔗 The service will restart automatically on Render"
echo "   Monitor: https://dashboard.render.com" 