#!/bin/bash

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Run the Flask application in the background
echo "Starting the SHAP-XGBoost Microservice..."
python app.py &

# Save the PID of the Flask app
FLASK_PID=$!

# Wait a moment for the app to start
echo "Waiting for the service to start..."
sleep 2

# Make the test script executable if it isn't already
chmod +x test_api.sh

# Run the test script
echo "Running API tests..."
./test_api.sh

# Ask the user if they want to stop the service
echo ""
read -p "Do you want to stop the service? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Stopping the service..."
    kill $FLASK_PID
    echo "Service stopped."
fi
