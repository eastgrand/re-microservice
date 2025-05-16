#!/bin/bash
# Deploy verification tools to Render.com server
# This script creates a small deployment package with just the verification tools

set -e  # Exit on error

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create temporary directory for deployment files
echo -e "${BLUE}Creating temporary directory for deployment...${NC}"
TEMP_DIR="verify_tools_deploy"
mkdir -p $TEMP_DIR

# Copy verification files to temp directory
echo -e "${BLUE}Copying verification tools...${NC}"
cp verify_redis_settings.py $TEMP_DIR/
cp verify_system_health.py $TEMP_DIR/
cp verify_config.py $TEMP_DIR/
cp run_test_job.py $TEMP_DIR/
cp VERIFICATION-RESULTS.md $TEMP_DIR/

# Make scripts executable
echo -e "${BLUE}Making scripts executable...${NC}"
chmod +x $TEMP_DIR/*.py

# Create a README file
echo -e "${BLUE}Creating README file...${NC}"
cat > $TEMP_DIR/README-VERIFICATION.md << EOF
# SHAP Microservice Verification Tools

These tools help verify the proper operation of the SHAP microservice optimizations.

## Available Tools

1. **verify_config.py** - Check configuration files (doesn't require external dependencies)
   \`\`\`
   ./verify_config.py
   \`\`\`

2. **verify_redis_settings.py** - Verify Redis connection settings
   \`\`\`
   ./verify_redis_settings.py
   \`\`\`

3. **verify_system_health.py** - Comprehensive system health check
   \`\`\`
   ./verify_system_health.py --all
   \`\`\`

4. **run_test_job.py** - Submit a test job and monitor completion
   \`\`\`
   ./run_test_job.py --sample-size 50
   \`\`\`

## Required Packages

These tools require the following Python packages:
\`\`\`
pip install redis rq flask pandas numpy psutil requests
\`\`\`

## Results

See VERIFICATION-RESULTS.md for a summary of the verification outcomes.
EOF

# Create a ZIP archive
echo -e "${BLUE}Creating ZIP archive...${NC}"
zip -r verification_tools.zip $TEMP_DIR

# Clean up temporary directory
echo -e "${BLUE}Cleaning up...${NC}"
rm -rf $TEMP_DIR

echo -e "${GREEN}Deployment package created: verification_tools.zip${NC}"
echo -e "${YELLOW}Upload this package to your Render.com server and extract to run verification tools.${NC}"
echo -e "${YELLOW}You can upload it using the Render.com web console or via scp/rsync if you have shell access.${NC}"

# Instructions for running tools
echo -e "${BLUE}=== Instructions ===${NC}"
echo -e "1. Upload verification_tools.zip to the server"
echo -e "2. Extract the archive: unzip verification_tools.zip"
echo -e "3. Run the verification tool: ./verify_tools_deploy/verify_config.py"
echo -e "4. Check the results and follow recommendations"
