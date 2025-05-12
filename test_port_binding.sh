#!/bin/bash
# Script to test port binding for Render deployment

echo "===== PORT BINDING TEST ====="
echo "Testing port binding for Render deployment..."

# Set environment variables
export PORT=${PORT:-5000}
export MEMORY_OPTIMIZATION=true
export MAX_MEMORY_MB=400
export AGGRESSIVE_MEMORY_MANAGEMENT=true

# Print environment info
echo "PORT: $PORT"
echo "MEMORY_OPTIMIZATION: $MEMORY_OPTIMIZATION"
echo "MAX_MEMORY_MB: $MAX_MEMORY_MB"

# Start with gunicorn using the config file, same as production
echo "Starting Gunicorn on port $PORT..."
gunicorn app:app -c gunicorn_config.py --preload &
GUNICORN_PID=$!

# Give it a moment to start
echo "Waiting for service to start..."
sleep 5

# Check if the port is open
echo "Checking if port $PORT is open..."
if command -v lsof &> /dev/null; then
    lsof -i :$PORT
elif command -v netstat &> /dev/null; then
    netstat -tuln | grep $PORT
else
    echo "Cannot check port binding - neither lsof nor netstat found"
fi

# Test the endpoint
echo "Testing connection to the API..."
curl -s http://localhost:$PORT/ || echo "Failed to connect to API"

# Kill the gunicorn process
kill $GUNICORN_PID
wait $GUNICORN_PID 2>/dev/null

echo "Port binding test complete"
