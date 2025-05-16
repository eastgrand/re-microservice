#!/bin/bash

# This script prepares and deploys the SHAP/XGBoost microservice to Render.com
# It performs checks to ensure everything is ready for deployment

echo "====================================="
echo "SHAP/XGBoost Microservice Deployment"
echo "====================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git and try again."
    exit 1
fi

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
fi

# Check required files
required_files=("app.py" "train_model.py" "patch_shap.py" "requirements.txt" "render.yaml")
missing_files=0

echo "Checking required files..."
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing file: $file"
        missing_files=1
    else
        echo "✅ Found: $file"
    fi
done

if [ $missing_files -eq 1 ]; then
    echo "Error: Some required files are missing. Please ensure all required files are present."
    exit 1
fi

# Create .skip_training flag file to ensure we skip model training during deployment
echo "Creating .skip_training flag file..."
echo "Skip model training during deployment - $(date)" > .skip_training
echo "✅ Created .skip_training flag file"
export SKIP_MODEL_TRAINING=true

# Check if models directory exists and contains model files
if [ ! -d "models" ]; then
    echo "⚠️  Models directory not found. Creating it now..."
    mkdir -p models
else
    echo "✅ Models directory found."
fi

# Check for model files
if [ ! -f "models/xgboost_model.pkl" ] || [ ! -f "models/feature_names.txt" ]; then
    echo "⚠️  Model files are missing. Attempting to create minimal model..."
    
    if [ -f "create_minimal_model.py" ]; then
        python3 create_minimal_model.py
        if [ -f "models/xgboost_minimal.pkl" ]; then
            echo "Copying minimal model files to standard locations..."
            cp models/xgboost_minimal.pkl models/xgboost_model.pkl
            cp models/minimal_feature_names.txt models/feature_names.txt
            echo "✅ Created and installed minimal model files."
        else
            echo "❌ Failed to create minimal model files."
            exit 1
        fi
    else
        echo "❌ Cannot create model files (create_minimal_model.py not found)!"
        exit 1
    fi
else
    echo "✅ Model files found and ready for deployment."
    echo "   - models/xgboost_model.pkl"
    echo "   - models/feature_names.txt"
fi

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "⚠️  Data directory not found. Creating it now..."
    mkdir -p data
else
    echo "✅ Data directory found."
fi

# Prompt for GitHub repository URL
read -p "Enter your GitHub repository URL (or press Enter to skip): " github_url

if [ -n "$github_url" ]; then
    # Extract username and repo name from URL
    username=$(echo "$github_url" | sed -n 's/.*github.com\/\([^\/]*\).*/\1/p')
    reponame=$(echo "$github_url" | sed -n 's/.*github.com\/[^\/]*\/\([^\.]*\).*/\1/p')
    
    if [ -z "$username" ] || [ -z "$reponame" ]; then
        echo "Invalid GitHub URL. Using default remote name 'origin'."
        remote_name="origin"
    else
        echo "Repository: $username/$reponame"
        remote_name="origin"
    fi
    
    # Check if remote already exists
    if git remote | grep -q "$remote_name"; then
        echo "Remote $remote_name already exists."
    else
        echo "Adding remote $remote_name as $github_url"
        git remote add "$remote_name" "$github_url"
    fi
    
    # Stage all files
    git add .
    
    # Commit changes
    read -p "Enter commit message [Deployment preparation]: " commit_msg
    commit_msg=${commit_msg:-"Deployment preparation"}
    git commit -m "$commit_msg"
    
    # Push to remote
    echo "Pushing code to GitHub..."
    git push "$remote_name" main || git push "$remote_name" master
    
    echo "Code pushed to GitHub successfully."
    echo ""
    echo "====================================="
    echo "NEXT STEPS FOR RENDER DEPLOYMENT:"
    echo "====================================="
    echo "1. Go to https://dashboard.render.com/blueprints/new"
    echo "2. Connect your GitHub account"
    echo "3. Select the repository $username/$reponame"
    echo "4. Click 'Apply Blueprint'"
    echo "5. Set up environment variables in the Render dashboard"
    echo ""
    echo "For detailed instructions, see RENDER-DEPLOYMENT-GUIDE.md"
else
    echo "Skipping GitHub push."
    echo ""
    echo "To deploy to Render:"
    echo "1. Push your code to GitHub first"
    echo "2. Follow the instructions in RENDER-DEPLOYMENT-GUIDE.md"
fi

echo ""
echo "Deployment preparation complete!"
