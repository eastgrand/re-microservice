#!/bin/bash
# Script to test memory optimization for Render deployment

echo "========== TESTING MEMORY OPTIMIZATION =========="

# Install memory profiler if not already installed
pip install memory-profiler

# Set environment variable to enable memory optimization
export MEMORY_OPTIMIZATION=true

# Step 1: Test setup script with memory profile
echo ""
echo "1. Testing setup_for_render.py with memory profiling..."
python -m memory_profiler setup_for_render.py

# Step 2: Test model training with memory profile
echo ""
echo "2. Testing train_model.py with memory profiling..."
python -m memory_profiler train_model.py

# Step 3: Test the app with memory profile
echo ""
echo "3. Testing app.py with memory profiling..."
# Start the app in background
python -m memory_profiler app.py &
APP_PID=$!

# Wait for app to start
sleep 5

# Make a test request
curl -s -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"analysis_type":"correlation","target_variable":"Mortgage_Approvals"}' | head -n 20

# Kill the app
kill $APP_PID

echo ""
echo "========== MEMORY OPTIMIZATION TEST COMPLETE =========="
echo "Check the output above for memory usage at each step."
echo "If peak memory usage is below 512MB, the optimizations should work on Render."
