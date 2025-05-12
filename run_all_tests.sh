#!/bin/bash
# Complete testing script that activates the virtual environment and runs tests

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Update Python path to include current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Step 1: Test SHAP directly
echo "==================================="
echo "STEP 1: Testing SHAP functionality"
echo "==================================="
python quick_shap_test.py

# Capture the result
if [ $? -eq 0 ]; then
    echo -e "\n✅ SHAP test completed successfully"
else
    echo -e "\n❌ SHAP test failed"
    echo "Please check the errors above and fix them before proceeding."
    exit 1
fi

# Step 2: Test Flask API
echo -e "\n==================================="
echo "STEP 2: Testing Flask API"
echo "==================================="

# Update port in .env file
grep -q "PORT=8081" .env || sed -i '' 's/PORT=5000/PORT=8081/' .env

# Run the test script
python test_api_curl.py

echo -e "\n==================================="
echo "All tests complete!"
echo "==================================="
echo "If all tests passed, your service is ready for deployment to Render."
echo "If any tests failed, fix the issues before deploying."
