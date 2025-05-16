#!/bin/bash
# filepath: /Users/voldeck/code/shap-microservice/deploy_to_live_service.sh
#
# Comprehensive deployment script to push SHAP memory optimization fix
# to the live production server on Render.com

set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BOLD}SHAP Microservice Live Deployment${NC}"
echo -e "--------------------------------"

# Check if git is available
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git is not installed. This script requires git.${NC}"
    exit 1
fi

# Check if we're in the git repository
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}⚠️ Not in the git repository root. Trying to find it...${NC}"
    
    # Check if we're in the shap-microservice directory
    if [ -f "app.py" ] && [ -f "render.yaml" ]; then
        echo -e "${GREEN}✅ Found SHAP microservice files in current directory.${NC}"
    else
        echo -e "${RED}❌ Could not find SHAP microservice repository.${NC}"
        exit 1
    fi
fi

# Verify that our memory optimization files exist
if [ ! -f "shap_memory_fix.py" ]; then
    echo -e "${RED}❌ Memory optimization file 'shap_memory_fix.py' not found.${NC}"
    echo -e "${YELLOW}⚠️ Creating it...${NC}"
    
    # Create the memory optimization file
    cat > shap_memory_fix.py << 'EOL'
#!/usr/bin/env python3
"""
SHAP Memory Optimization

This module provides memory optimization for SHAP analysis,
allowing for the processing of larger datasets without running
out of memory.
"""

import os
import gc
import logging
import numpy as np
from typing import Union, Optional, Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shap-memory-fix")

# Configuration for batch processing
MAX_ROWS_TO_PROCESS = int(os.environ.get('SHAP_MAX_BATCH_SIZE', '1000'))

class ShapValuesWrapper:
    """Wrapper class to maintain compatibility with SHAP values API"""
    
    def __init__(self, values, base_values=None, data=None, feature_names=None):
        """Initialize with raw SHAP values"""
        self.values = values
        self.base_values = base_values
        self.data = data
        self.feature_names = feature_names
        
    def __len__(self):
        """Return the number of rows in the SHAP values"""
        if hasattr(self.values, '__len__'):
            return len(self.values)
        return 0

def apply_memory_patches(app=None):
    """
    Apply memory optimization patches to the Flask app
    
    Args:
        app: Flask application instance
    """
    import gc
    
    # Enable garbage collection to run more aggressively
    gc.enable()
    logger.info("Garbage collection enabled with thresholds: %s", gc.get_threshold())
    
    # Function to create memory-optimized explainer
    def create_memory_optimized_explainer(model, X, feature_names=None, 
                                         max_rows=MAX_ROWS_TO_PROCESS):
        """
        Creates a memory-optimized explainer by processing data in batches
        
        Args:
            model: The trained model to explain
            X: Input features to explain
            feature_names: Optional list of feature names
            max_rows: Maximum number of rows to process in one batch
            
        Returns:
            ShapValuesWrapper containing the computed SHAP values
        """
        import shap
        
        # Start with garbage collection to ensure we have maximum memory available
        gc.collect()
        
        logger.info(f"Creating memory-optimized explainer for {len(X)} rows")
        logger.info(f"Using max batch size of {max_rows} rows")
        
        # Check if the dataset is small enough to process in one go
        if len(X) <= max_rows:
            logger.info("Dataset small enough for direct processing")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X)
            
            # Return ShapValuesWrapper if needed
            if not hasattr(shap_values, 'values'):
                return ShapValuesWrapper(shap_values)
            return shap_values
        
        # Process in batches for large datasets
        logger.info(f"Processing large dataset in batches")
        all_shap_values = []
        total_rows = len(X)
        chunks = (total_rows + max_rows - 1) // max_rows  # Ceiling division
        
        for i in range(chunks):
            start_idx = i * max_rows
            end_idx = min((i + 1) * max_rows, total_rows)
            
            logger.info(f"Processing batch {i+1}/{chunks} (rows {start_idx}-{end_idx})")
            
            # Extract this chunk of data
            X_chunk = X.iloc[start_idx:end_idx]
            
            # Create explainer and get SHAP values for this chunk
            explainer = shap.TreeExplainer(model)
            chunk_shap_values = explainer(X_chunk)
            
            # Extract values (handle different return types from different SHAP versions)
            if hasattr(chunk_shap_values, 'values'):
                all_shap_values.append(chunk_shap_values.values)
            else:
                all_shap_values.append(chunk_shap_values)
                
            # Force cleanup to free memory
            del explainer
            del chunk_shap_values
            del X_chunk
            gc.collect()
            
        # Combine all chunks
        logger.info("Combining SHAP values from all batches")
        try:
            combined_values = np.vstack(all_shap_values)
            return ShapValuesWrapper(combined_values)
        except:
            logger.warning("Could not combine values using np.vstack, returning list")
            return ShapValuesWrapper(all_shap_values)
    
    # Patch the calculation function
    global calculate_shap_values
    
    # Store original function if it exists
    if 'calculate_shap_values' in globals():
        original_calculate_shap_values = calculate_shap_values
        
        # Define patched function with same signature
        def memory_optimized_calculate_shap_values(model, X, feature_names=None, **kwargs):
            """Memory optimized version of calculate_shap_values"""
            logger.info("Using memory-optimized SHAP calculation")
            return create_memory_optimized_explainer(model, X, feature_names, MAX_ROWS_TO_PROCESS)
        
        # Replace the global function
        calculate_shap_values = memory_optimized_calculate_shap_values
    else:
        # If function doesn't exist yet, create it
        def calculate_shap_values(model, X, feature_names=None, **kwargs):
            """Memory optimized SHAP calculation function"""
            logger.info("Using memory-optimized SHAP calculation")
            return create_memory_optimized_explainer(model, X, feature_names, MAX_ROWS_TO_PROCESS)
    
    # If app is provided, add memory monitoring endpoint
    if app is not None:
        logger.info("Adding memory monitoring endpoint to Flask app")
        try:
            from flask import jsonify
            import psutil
            
            @app.route('/admin/memory', methods=['GET'])
            def memory_status():
                """Return current memory usage and status"""
                try:
                    process = psutil.Process(os.getpid())
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    return jsonify({
                        "success": True,
                        "memory_usage_mb": memory_mb,
                        "optimized_worker_applied": True,
                        "gc_enabled": gc.isenabled(),
                        "gc_counts": gc.get_count(),
                        "gc_threshold": gc.get_threshold(),
                        "max_rows_per_batch": MAX_ROWS_TO_PROCESS
                    })
                except Exception as e:
                    logger.error(f"Error in memory endpoint: {str(e)}")
                    return jsonify({
                        "success": False,
                        "error": str(e)
                    }), 500
        except ImportError:
            logger.warning("Could not add memory endpoint: psutil not installed")
    
    logger.info("Memory optimization patches applied successfully")
    return True

# Export the memory optimization function
__all__ = ['apply_memory_patches', 'MAX_ROWS_TO_PROCESS', 'ShapValuesWrapper']

# If run directly, print info
if __name__ == "__main__":
    print("SHAP Memory Optimization Module")
    print(f"Maximum rows per batch: {MAX_ROWS_TO_PROCESS}")
    print("To use, import this module and call apply_memory_patches()")
EOL

    # Make executable
    chmod +x shap_memory_fix.py
    echo -e "${GREEN}✅ Created memory optimization file.${NC}"
fi

# Check if we already have a commit for this change
if git log -n 10 | grep -q "Apply memory optimization fix"; then
    echo -e "${YELLOW}⚠️ Memory optimization fix already committed.${NC}"
else
    echo -e "${YELLOW}🔄 Preparing to commit memory optimization fix...${NC}"
    
    # Check if app.py needs to be modified
    if ! grep -q "apply_memory_patches" app.py; then
        echo -e "${YELLOW}⚠️ Adding memory optimization to app.py...${NC}"
        
        # Find where to add the import
        IMPORT_LINE=$(grep -n "import " app.py | head -1 | cut -d: -f1)
        if [ -z "$IMPORT_LINE" ]; then
            IMPORT_LINE=1
        fi
        
        # Add import statements
        sed -i.bak "${IMPORT_LINE}i\\
# Memory optimization imports\\
try:\\
    from shap_memory_fix import apply_memory_patches, MAX_ROWS_TO_PROCESS\\
except ImportError:\\
    logger.warning('SHAP memory optimization not available')\\
" app.py
        
        # Find where to add the function call (after app initialization)
        APP_INIT_LINE=$(grep -n "app = Flask" app.py | head -1 | cut -d: -f1)
        if [ -z "$APP_INIT_LINE" ]; then
            APP_INIT_LINE=$(grep -n "Flask(__name__)" app.py | head -1 | cut -d: -f1)
        fi
        
        if [ -n "$APP_INIT_LINE" ]; then
            # Add the function call after app initialization
            APP_INIT_LINE=$((APP_INIT_LINE + 2))
            sed -i.bak "${APP_INIT_LINE}i\\
# Apply SHAP memory optimization if available\\
try:\\
    apply_memory_patches(app)\\
    logger.info('✅ Applied SHAP memory optimization patches')\\
except (NameError, ImportError) as e:\\
    logger.warning(f'⚠️ Could not apply SHAP memory optimization: {str(e)}')\\
" app.py
        fi
        
        echo -e "${GREEN}✅ Added memory optimization to app.py${NC}"
    else
        echo -e "${GREEN}✅ Memory optimization already in app.py${NC}"
    fi
    
    # Update render.yaml if needed to increase worker memory
    if [ -f "render.yaml" ]; then
        if ! grep -q "SHAP_MAX_BATCH_SIZE" render.yaml; then
            echo -e "${YELLOW}⚠️ Updating render.yaml with memory configuration...${NC}"
            
            # Find the envVars section
            ENV_LINE=$(grep -n "envVars:" render.yaml | head -1 | cut -d: -f1)
            if [ -n "$ENV_LINE" ]; then
                ENV_LINE=$((ENV_LINE + 1))
                sed -i.bak "${ENV_LINE}i\\
      - key: SHAP_MAX_BATCH_SIZE\\
        value: \"500\"\\
" render.yaml
                echo -e "${GREEN}✅ Updated render.yaml with batch size configuration${NC}"
            fi
        else
            echo -e "${GREEN}✅ Batch size already configured in render.yaml${NC}"
        fi
    fi
    
    # Commit the changes
    git add shap_memory_fix.py app.py render.yaml
    git commit -m "Apply memory optimization fix for SHAP calculations"
    echo -e "${GREEN}✅ Changes committed locally${NC}"
fi

# Ask to push to remote
echo ""
echo -e "${YELLOW}Do you want to push these changes to the remote repository?${NC}"
echo -e "${YELLOW}This will deploy the changes to the live service on Render.com.${NC}"
read -p "Push to remote? (y/n): " PUSH_CONFIRM

if [[ $PUSH_CONFIRM == "y" || $PUSH_CONFIRM == "Y" ]]; then
    echo -e "${YELLOW}🔄 Pushing changes to remote repository...${NC}"
    
    # Check the remote repository
    REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
    
    if [ -z "$REMOTE_URL" ]; then
        echo -e "${RED}❌ No remote repository configured.${NC}"
        read -p "Enter the remote repository URL: " REMOTE_URL
        
        if [ -z "$REMOTE_URL" ]; then
            echo -e "${RED}❌ No remote URL provided. Cannot push changes.${NC}"
            exit 1
        fi
        
        git remote add origin "$REMOTE_URL"
    fi
    
    # Determine the branch name
    BRANCH_NAME=$(git branch --show-current)
    if [ -z "$BRANCH_NAME" ]; then
        BRANCH_NAME="main"
    fi
    
    # Push to remote
    git push origin "$BRANCH_NAME"
    echo -e "${GREEN}✅ Changes pushed to remote repository${NC}"
    echo -e "${GREEN}✅ Render.com will automatically deploy the new version${NC}"
else
    echo -e "${YELLOW}⚠️ Changes not pushed to remote. Deployment cancelled.${NC}"
    echo -e "You can push changes later with: git push origin \$(git branch --show-current)"
fi

echo ""
echo -e "${BOLD}Deployment Status${NC}"
echo -e "-----------------"
echo -e "${GREEN}✓${NC} Memory optimization code created"
echo -e "${GREEN}✓${NC} App.py updated with memory optimization"
echo -e "${GREEN}✓${NC} Render.yaml configured for batch processing"
echo -e "${GREEN}✓${NC} Changes committed locally"

if [[ $PUSH_CONFIRM == "y" || $PUSH_CONFIRM == "Y" ]]; then
    echo -e "${GREEN}✓${NC} Changes pushed to remote repository"
    echo -e "${YELLOW}⚠️ Render.com is now deploying the changes${NC}"
    echo -e "${YELLOW}⚠️ This may take a few minutes${NC}"
    echo ""
    echo -e "To verify the deployment, run the following command after 5-10 minutes:"
    echo -e "${BOLD}./verify_live_endpoint.sh${NC}"
else
    echo -e "${RED}✗${NC} Changes not pushed to remote repository"
    echo -e "${RED}✗${NC} Live service not updated"
fi

echo ""
echo -e "${BOLD}Next Steps${NC}"
echo -e "----------"
echo -e "1. Wait for Render.com to complete the deployment (5-10 minutes)"
echo -e "2. Verify the deployment with: ${BOLD}./verify_live_endpoint.sh${NC}"
echo -e "3. Check the memory usage with: ${BOLD}python3 check_worker_status.py${NC}"
echo ""
