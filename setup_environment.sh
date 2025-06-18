#!/bin/bash

# SHAP Microservice Environment Setup Script
# This script sets up the Python virtual environment and installs all dependencies

set -e  # Exit on any error

echo "🚀 Setting up SHAP Microservice Environment"
echo "=============================================="

# Check if we're in the right directory
if [[ ! -f "app.py" ]]; then
    echo "❌ Error: Please run this script from the shap-microservice directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected files: app.py, requirements.txt"
    exit 1
fi

# Detect Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "🐍 Detected Python version: $PYTHON_VERSION"

# Choose virtual environment name based on Python version
if [[ "$PYTHON_VERSION" == "3.11" ]]; then
    VENV_NAME="venv311"
elif [[ "$PYTHON_VERSION" == "3.13" ]]; then
    VENV_NAME="venv313"
else
    VENV_NAME="venv"
fi

echo "📁 Using virtual environment: $VENV_NAME"

# Create virtual environment if it doesn't exist
if [[ ! -d "$VENV_NAME" ]]; then
    echo "🔨 Creating virtual environment: $VENV_NAME"
    python3 -m venv "$VENV_NAME"
else
    echo "✅ Virtual environment already exists: $VENV_NAME"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Check if requirements.txt exists
if [[ ! -f "requirements.txt" ]]; then
    echo "📝 Creating requirements.txt with essential dependencies..."
    cat > requirements.txt << 'EOF'
# Core dependencies for SHAP microservice
flask==2.3.3
flask-cors==4.0.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
xgboost==1.7.6
shap==0.42.1
requests==2.31.0
redis==4.6.0
rq==1.15.1
gunicorn==21.2.0

# Data processing
pyyaml==6.0.1
python-dotenv==1.0.0

# Development and testing
pytest==7.4.0
pytest-cov==4.1.0
EOF
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create essential directories
echo "📁 Creating essential directories..."
mkdir -p data models precalculated config versions

# Create .env file if it doesn't exist
if [[ ! -f ".env" ]]; then
    echo "⚙️  Creating .env configuration file..."
    cat > .env << 'EOF'
# SHAP Microservice Configuration
FLASK_ENV=development
FLASK_DEBUG=True
REDIS_URL=redis://localhost:6379/0
MODEL_PATH=models/xgboost_model.pkl
DATA_PATH=data/cleaned_data.csv
PRECALCULATED_PATH=precalculated/
LOG_LEVEL=INFO
EOF
fi

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x extract_arcgis_data.py
chmod +x setup_arcgis_extraction.py
chmod +x update_data_pipeline.py

# Create activation script for easy use
echo "📝 Creating activation script..."
cat > activate_env.sh << EOF
#!/bin/bash
# Quick activation script for SHAP microservice environment
echo "🔌 Activating SHAP microservice environment..."
source "$VENV_NAME/bin/activate"
echo "✅ Environment activated!"
echo "📍 Current directory: \$(pwd)"
echo "🐍 Python version: \$(python --version)"
echo ""
echo "🚀 Available commands:"
echo "  python setup_arcgis_extraction.py     # Configure data extraction"
echo "  python extract_arcgis_data.py         # Extract ArcGIS data"
echo "  python update_data_pipeline.py        # Process data and train model"
echo "  python app.py                         # Start SHAP microservice"
echo ""
EOF

chmod +x activate_env.sh

# Test the installation
echo "🧪 Testing installation..."
python -c "
import flask, pandas, numpy, sklearn, xgboost, shap, requests
print('✅ All core dependencies imported successfully')
print(f'   - Flask: {flask.__version__}')
print(f'   - Pandas: {pandas.__version__}')
print(f'   - NumPy: {numpy.__version__}')
print(f'   - Scikit-learn: {sklearn.__version__}')
print(f'   - XGBoost: {xgboost.__version__}')
print(f'   - SHAP: {shap.__version__}')
"

echo ""
echo "🎉 SHAP Microservice Environment Setup Complete!"
echo "================================================="
echo ""
echo "📋 What was created:"
echo "  ✅ Virtual environment: $VENV_NAME"
echo "  ✅ Dependencies installed from requirements.txt"
echo "  ✅ Essential directories created"
echo "  ✅ Configuration files created"
echo "  ✅ Scripts made executable"
echo ""
echo "🚀 To get started:"
echo "  1. Activate environment: source activate_env.sh"
echo "  2. Configure data extraction: python setup_arcgis_extraction.py"
echo "  3. Extract your ArcGIS data: python extract_arcgis_data.py --config config/feature_services.json"
echo "  4. Process data and train model: python update_data_pipeline.py --use-existing"
echo ""
echo "💡 Quick activation for future sessions:"
echo "   source activate_env.sh"
echo ""
echo "🔧 Environment details:"
echo "   - Virtual environment: $VENV_NAME"
echo "   - Python version: $PYTHON_VERSION"
echo "   - Working directory: $(pwd)" 