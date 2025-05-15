# Flask Application Context Fix for Redis Health Check

## Issue

When deploying the SHAP microservice to Render, the following error appeared in the logs:

```
ERROR:redis-patch:Could not add Redis health check endpoint: Working outside of application context.
This typically means that you attempted to use functionality that needed
the current application. To solve this, set up an application context
with app.app_context(). See the documentation for more information.
```

This error occurred because the `redis_connection_patch.py` file was trying to register a Flask route (the `/admin/redis_ping` endpoint) outside of the Flask application context.

## Solution

The solution involved several changes:

1. **Restructured route registration**: Instead of trying to register the route directly during the patch application, we created a function that will register the route when provided with a Flask app instance.

2. **Updated `apply_all_patches()` function**: Modified to accept an optional `app` parameter that can be used to register the Redis health check endpoint.

3. **Updated app.py**: Modified to pass the Flask app instance to the `apply_all_patches()` function and store the Redis connection in the app's config for the endpoint to access.

## Changes Made

### 1. In `redis_connection_patch.py`:

- Created a `register_redis_ping_endpoint()` function that takes a Flask app instance
- Modified `apply_all_patches()` to accept an optional `app` parameter
- Added proper handling for registering the endpoint when an app is provided

### 2. In `app.py`:

- Updated the call to `apply_all_patches()` to pass the `app` instance
- Added the Redis connection to the app's config for the endpoint to access

## Verification

To verify that the Redis health check endpoint is working properly:

```bash
python verify_redis_health.py [service_url]
```

This script will test the Redis health check endpoint and report whether it's working correctly.

## Lessons Learned

When patching functionality that interacts with Flask:

1. Always be aware of the application context requirements
2. Provide mechanisms to register routes properly with the Flask app
3. Don't assume that imports like `current_app` will work outside of request contexts
4. Use dependency injection (e.g., passing the app instance) when needed

## Related Documentation

- [Flask Application Context](https://flask.palletsprojects.com/en/2.0.x/appcontext/)
- [Flask Route Registration](https://flask.palletsprojects.com/en/2.0.x/api/#flask.Flask.route)
