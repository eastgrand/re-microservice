#!/bin/bash
# Activate the virtual environment
source venv/bin/activate

# Install all dependencies from requirements.txt
echo "Installing all dependencies from requirements.txt..."
pip install -r requirements.txt

# Confirm key installations
echo "Confirming key installations:"
python -c "import flask; print(f'Flask version: {flask.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
python -c "import shap; print(f'SHAP version: {shap.__version__}')"
python -c "import requests; print(f'Requests version: {requests.__version__}')"

echo "All dependencies have been installed."
