# Redis Connection Fix - Production Deployment Guide

This guide outlines the steps to deploy the Redis connection improvements to production (Render.com).

## Overview of Changes

The Redis connection fix addresses the following issues:
- Worker processes failing to process jobs
- Jobs getting stuck in the "started" state
- Timeouts in service health check endpoints
- Redis connection errors

## Deployment Steps

### 1. Commit and Push Code Changes

```bash
# Navigate to the SHAP microservice directory
cd /Users/voldeck/code/shap-microservice

# Add the Redis connection fixes
git add redis_connection_patch.py
git add app.py
git add .env
git add REDIS-CONNECTION-FIX.md
git add REDIS-DEPLOYMENT-INSTRUCTIONS.md

# Commit the changes
git commit -m "Fix Redis connection issues with timeouts, pooling, and automatic reconnection"

# Push to your repository
git push
```

### 2. Deploy to Render.com

1. Log into the [Render dashboard](https://dashboard.render.com/)
2. Navigate to your SHAP microservice
3. Either:
   - Wait for automatic deployment if you have CI/CD enabled, or
   - Click "Manual Deploy" > "Deploy latest commit"

### 3. Add Environment Variables

In the Render dashboard, add these environment variables:

| Variable | Value | Description |
|----------|-------|-------------|
| `REDIS_TIMEOUT` | `5` | Redis connection timeout in seconds |
| `REDIS_SOCKET_KEEPALIVE` | `true` | Enable TCP keepalive for connections |
| `REDIS_CONNECTION_POOL_SIZE` | `10` | Number of connections to maintain in pool |
| `AGGRESSIVE_MEMORY_MANAGEMENT` | `true` | Enable memory optimization features |

### 4. Verify Deployment

After deployment is complete, verify that the service is working properly:

```bash
# From the client application directory
cd /Users/voldeck/code/ai-analytics-app
node scripts/verify-shap-connection.mjs
```

You should see Redis connection success indicated in the output.

## Verifying Client Integration

After deploying the SHAP microservice with Redis improvements, verify that the client application can successfully interact with it:

1. Ensure the client application is using the latest `shapClientEnhanced.ts` with timeout protection
2. Run the verification script: `node scripts/verify-shap-connection.mjs`
3. Test submitting a job through the client application UI
4. Monitor the job status to ensure it progresses through the queue

## Monitoring and Maintenance

After deployment, monitor the service for any remaining issues:

1. Check Render logs for any Redis-related errors
2. Monitor job processing times and success rates
3. If issues persist, consider increasing the Redis timeouts or connection pool size

## Rollback Procedure

If problems occur after deployment:

1. In the Render dashboard, find the last known good deployment
2. Click "Manual Deploy" > "Deploy previous build"
3. Remove the Redis environment variables if they're causing issues

---

For any questions or issues, refer to the `REDIS-CONNECTION-FIX.md` file for detailed technical information about the implementation.
