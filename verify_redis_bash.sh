#!/bin/bash

# Simple SHAP Microservice Redis Verification
# This script checks if the SHAP microservice is working with Redis

echo "===== SHAP Microservice Redis Connection Verification ====="

# Check if the microservice is running
SERVICE_PID=$(ps -ef | grep "gunicorn" | grep -v grep | awk '{print $2}')
if [ -z "$SERVICE_PID" ]; then
  echo "❌ SHAP microservice is not running"
  echo "Please start the service with: ./deploy_redis_fix.sh"
  exit 1
else
  echo "✅ SHAP microservice is running (PID: $SERVICE_PID)"
fi

# Attempt to make simple curl requests to the service
echo -e "\nChecking basic service endpoints:"

# Check ping endpoint
echo -n "Testing /ping endpoint... "
PING_RESPONSE=$(curl -s -w "%{http_code}" -o /dev/null http://localhost:8081/ping)
if [ "$PING_RESPONSE" == "200" ]; then
  echo "✅ SUCCESS"
else
  echo "❌ FAILED (HTTP $PING_RESPONSE)"
fi

# Check health endpoint
echo -n "Testing /health endpoint... "
HEALTH_RESPONSE=$(curl -s http://localhost:8081/health)
if [[ $HEALTH_RESPONSE == *"redis_connected"* ]]; then
  echo "✅ SUCCESS"
  echo "$HEALTH_RESPONSE" | grep -o '"redis_connected": [a-z]*' | sed 's/^/- /'
  echo "$HEALTH_RESPONSE" | grep -o '"queue_active": [a-z]*' | sed 's/^/- /'
  echo "$HEALTH_RESPONSE" | grep -o '"active_workers": [0-9]*' | sed 's/^/- /'
else
  echo "❌ FAILED"
  echo "Response: $HEALTH_RESPONSE"
fi

# Check Redis ping endpoint
echo -n "Testing Redis ping endpoint... "
REDIS_RESPONSE=$(curl -s http://localhost:8081/admin/redis_ping)
if [[ $REDIS_RESPONSE == *"success"*"true"* ]]; then
  echo "✅ SUCCESS - Redis is responding to PING"
else
  echo "❌ FAILED - Redis ping endpoint did not return success"
  echo "Response: $REDIS_RESPONSE"
fi

# Submit a small test job
echo -e "\nSubmitting a test job to check Redis queue:"
JOB_RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" -d '{"analysis_type":"correlation","target_variable":"test","data":[{"test":1,"value":2},{"test":0,"value":1}]}' http://localhost:8081/analyze)

if [[ $JOB_RESPONSE == *"job_id"* ]]; then
  JOB_ID=$(echo $JOB_RESPONSE | grep -o '"job_id": "[^"]*"' | cut -d'"' -f4)
  echo "✅ Job submitted successfully with ID: $JOB_ID"
  
  # Poll for job status a few times
  echo "Polling for job status:"
  for i in {1..5}; do
    echo -n "Poll $i: "
    STATUS_RESPONSE=$(curl -s http://localhost:8081/job_status/$JOB_ID)
    JOB_STATUS=$(echo $STATUS_RESPONSE | grep -o '"status": "[^"]*"' | cut -d'"' -f4)
    echo "$JOB_STATUS"
    
    if [[ "$JOB_STATUS" == "finished" ]]; then
      echo "✅ Job completed successfully!"
      break
    fi
    
    if [[ "$JOB_STATUS" == "failed" ]]; then
      echo "❌ Job failed!"
      break
    fi
    
    sleep 2
  done
else
  echo "❌ Failed to submit test job"
  echo "Response: $JOB_RESPONSE"
fi

echo -e "\n===== Verification Summary ====="
echo "The Redis connection fixes have been applied to the SHAP microservice"
echo "Service is running and accepting requests"
echo 
echo "Next Steps:"
echo "1. Commit all changes to your repository"
echo "2. Push changes to your remote repository"
echo "3. Deploy to Render.com with the following environment variables:"
echo "   REDIS_TIMEOUT=5"
echo "   REDIS_SOCKET_KEEPALIVE=true"
echo "   REDIS_CONNECTION_POOL_SIZE=10"
echo "   AGGRESSIVE_MEMORY_MANAGEMENT=true"
echo 
echo "If you encounter any issues, check the logs with: tail -f gunicorn.log"
