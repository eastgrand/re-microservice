#!/bin/bash
# Script to test the dynamic query processing system

echo "====== TESTING DYNAMIC QUERY PROCESSING SYSTEM ======"
echo

# Check if the Flask app is running
if ! curl -s http://localhost:5000/health > /dev/null; then
  echo "Error: Flask application is not running on port 5000."
  echo "Please start the application with:"
  echo "  python app.py"
  echo
  exit 1
fi

# Test classifier
echo "===== TESTING QUERY CLASSIFIER ====="
python test_query_classifier.py
echo

# Test module integration
echo "===== TESTING MODULE INTEGRATION ====="
python test_query_integration.py
echo

# Test API integration
echo "===== TESTING API INTEGRATION ====="
python test_enhanced_queries.py
echo

echo "====== ALL TESTS COMPLETED ======"
