#!/bin/bash
# install_dependencies.sh
# Helper script to install dependencies in the correct order
# Created: May 16, 2025

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
