#!/bin/bash
# verify_redis_endpoint.sh - Test the Redis health check endpoint after deployment

# Set the service URL (default to Render URL if not provided)
SERVICE_URL="${1:-https://nesto-mortgage-analytics.onrender.com}"
API_KEY="${API_KEY:-HFqkccbN3LV5CaB}"  # Default key if not specified

echo "🔍 Testing Redis health endpoint at $SERVICE_URL..."

# First try the ping endpoint to make sure the service is up
echo "Testing basic ping endpoint..."
curl -s "$SERVICE_URL/ping" | jq .

# Now test the Redis health check endpoint
echo "Testing Redis health check endpoint..."
if curl -s -H "X-API-KEY: $API_KEY" "$SERVICE_URL/admin/redis_ping" | grep -q "\"success\":true"; then
  echo "✅ Redis health check endpoint is working!"
  exit 0
else
  echo "❌ Redis health check endpoint failed or not available"
  curl -s -H "X-API-KEY: $API_KEY" "$SERVICE_URL/admin/redis_ping"
  exit 1
fi
