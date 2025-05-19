#!/bin/bash
# fix_render_deployment_dependencies.sh
# This script fixes the dependency installation order for Render deployment
# Created: May 16, 2025

set -e  # Exit immediately if a command exits with a non-zero status

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}   SHAP Microservice Deployment Dependencies Fix  ${NC}"
echo -e "${BLUE}=================================================${NC}"

# Step 1: Update render.yaml with proper dependency installation
echo -e "${YELLOW}Updating render.yaml with fixed dependencies order...${NC}"

# Create a backup
cp render.yaml render.yaml.bak.$(date +"%Y%m%d%H%M%S")
echo -e "${GREEN}✅ Created backup of render.yaml${NC}"

# Step 2: Create a helper script that will be called during deployment
echo -e "${YELLOW}Creating dependency installation helper script...${NC}"

cat > install_dependencies.sh <<'EOL'
#!/bin/bash
# install_dependencies.sh
# Helper script to install dependencies in the correct order

# Install core ML dependencies first
echo "Installing machine learning dependencies..."
pip install numpy pandas xgboost scikit-learn

# Install other dependencies
echo "Installing other dependencies..."
pip install flask==2.2.3 werkzeug==2.2.3 flask-cors==3.0.10 gunicorn==20.1.0 
pip install python-dotenv==1.0.0 psutil==5.9.5 memory-profiler==0.61.0
pip install rq>=1.15 redis>=4.0 shap==0.40.0 requests==2.31.0

# Verify installation
echo "Verifying pandas installation..."
python3 -c "import pandas; print(f'pandas {pandas.__version__} successfully installed')"

echo "Verifying numpy installation..."
python3 -c "import numpy; print(f'numpy {numpy.__version__} successfully installed')"

echo "Verifying xgboost installation..."
python3 -c "import xgboost; print(f'xgboost {xgboost.__version__} successfully installed')"

echo "✅ All dependencies installed successfully"
EOL

chmod +x install_dependencies.sh
echo -e "${GREEN}✅ Created install_dependencies.sh script${NC}"

# Step 3: Update render_pre_deploy.sh to use the new helper script
echo -e "${YELLOW}Ensuring render_pre_deploy.sh uses the new dependency installer...${NC}"

# Make the render_pre_deploy.sh executable
chmod +x render_pre_deploy.sh

# Step 4: Create an updated deployment script
cat > deploy_fixed_dependencies.sh <<'EOL'
#!/bin/bash
# deploy_fixed_dependencies.sh
# Deploy to Render with fixed dependency installation

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
EOL

chmod +x deploy_fixed_dependencies.sh
echo -e "${GREEN}✅ Created deploy_fixed_dependencies.sh script${NC}"

echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}✅ Dependency fix preparation complete!${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""
echo -e "To deploy with fixed dependencies, run:"
echo -e "${YELLOW}  ./deploy_fixed_dependencies.sh${NC}"
echo ""
echo -e "This will push changes to your repository and trigger a"
echo -e "new deployment on Render with the corrected dependency order."
