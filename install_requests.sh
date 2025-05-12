#!/bin/bash
# Activate the virtual environment
source venv/bin/activate

# Install the requests package
pip install requests

# Confirm installation
python -c "import requests; print(f'Requests package version: {requests.__version__}')"

echo "Requests package has been installed."
