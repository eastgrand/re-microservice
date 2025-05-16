#!/bin/bash
# deploy_fixed_dependencies.sh
# Deploy to Render with fixed dependency installation
# Created: May 16, 2025

set -e  # Exit immediately if a command exits with a non-zero status

echo "============================================="
echo "    DEPLOYING WITH FIXED DEPENDENCIES        "
echo "============================================="

# Make all scripts executable
chmod +x *.sh

# Create .skip_training flag
echo "Skip model training during deployment - $(date)" > .skip_training
echo "✅ Created .skip_training flag file"

# Push changes to Render
if [ -z "$1" ]; then
  echo "Committing changes and pushing to repository..."
  git add render.yaml .skip_training install_dependencies.sh render_pre_deploy.sh deploy_fixed_dependencies.sh
  git commit -m "Fix dependency installation order for Render - May 16, 2025"
  git push origin main
else
  echo "Skipping git push as requested by parameter"
fi

echo ""
echo "============================================="
echo "  DEPLOYMENT WITH FIXED DEPENDENCIES COMPLETE"
echo "============================================="
echo ""
echo "The build on Render should now succeed because dependencies"
echo "will be installed before the model creation script runs."
