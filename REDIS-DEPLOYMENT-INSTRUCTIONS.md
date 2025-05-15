# SHAP Microservice Redis Fix - Deployment Instructions

The Redis connection fixes have been successfully applied to your codebase. Follow these steps to deploy the changes:

## 1. Verify Local Changes

The following files have been modified:
- `redis_connection_patch.py`: Created with enhanced Redis connection handling
- `app.py`: Modified to use the Redis connection patch
- `.env`: Updated with Redis connection settings

## 2. Commit Changes

```bash
git add redis_connection_patch.py app.py .env
git commit -m "Add Redis connection improvements with timeouts and pooling"
git push
```

## 3. Deploy to Render.com

1. Wait for automatic deployment if you have CI/CD enabled
2. Or manually deploy from the Render dashboard

## 4. Add Environment Variables

In the Render dashboard for your service, add these environment variables:

```
REDIS_TIMEOUT=5
REDIS_SOCKET_KEEPALIVE=true
REDIS_CONNECTION_POOL_SIZE=10
AGGRESSIVE_MEMORY_MANAGEMENT=true
```

## 5. Verify Deployment

After deployment, you can test the service endpoints:
- `/ping` - Basic service availability
- `/health` - Check Redis connection status
- `/admin/redis_ping` - Test Redis ping functionality

## Troubleshooting

If you encounter Redis issues after deployment:

1. Check logs in the Render dashboard
2. Verify the Redis URL is correct in the environment variables
3. Try restarting the service
4. Confirm the Redis service is running and accessible

## Next Steps

Consider implementing additional improvements:
- Redis connection monitoring
- Automatic service recovery
- Queue monitoring and cleanup
