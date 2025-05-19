# SHAP Job Processing Fixed: NaN JSON Serialization Issue

## Issue Overview
**Date:** May 16, 2025

The SHAP microservice was experiencing an issue where jobs were getting stuck in the "started" state and never completing. The AI Analytics app was receiving the following error when polling for job status:

```
SyntaxError: Unexpected token 'N', ..."ications":NaN,"mortg"... is not valid JSON
```

This error occurred because the worker was attempting to include `NaN` (Not-a-Number) values directly in JSON output without properly formatting them as strings. Standard JSON does not support `NaN` as a literal value - it must be represented as a string like `"NaN"`.

## Root Cause Analysis

1. **Invalid JSON Format**: NaN values were being serialized as literal `NaN` in the JSON output
2. **Missing JSON Encoder**: No custom JSON encoder was handling special floating-point values
3. **Client Parsing Error**: When the client attempted to parse the response, it failed due to invalid JSON syntax
4. **Stalled Jobs**: As a result, jobs got stuck in "started" state since the client couldn't process the results

The logs show this clearly:
```
[SHAP DEBUG] Job status: started { status: 'started', success: true }
[SHAP WARNING] Unknown job status: started
...
[SHAP METRICS] Query processing failed after 190.50s: SyntaxError: Unexpected token 'N', ..."ications":NaN,"mortg"... is not valid JSON
```

## Solution Implemented

We have implemented a comprehensive fix:

1. **Custom JSON Encoder** (`json_serialization_fix.py`):
   - Enhanced JSON encoder that properly handles numpy types and special values
   - Converts `NaN` to string `"NaN"` during serialization
   - Also handles `Infinity` and `-Infinity` values
   - Patches both `json.dumps` and Flask's `jsonify`

2. **Job Status/Result Fix** (`fix_nan_json.py`):
   - Specifically targets the API endpoints where malformed JSON was occurring
   - Wraps response functions to properly handle NaN values
   - Uses both object-level fixes and string-level regex fixes as backup

3. **Integration with App** (via updates to `app.py`):
   - Automatically applies JSON serialization patches at startup
   - Ensures all JSON responses are properly formatted

## Deployment

To deploy this fix:

1. Run the deployment script which will:
   - Update the app with the fixes
   - Create a backup of the original app
   - Add appropriate documentation
   - Test the fix before committing

```bash
./deploy_json_nan_fix.sh
```

The changes will be applied and a new deployment will be triggered on Render.

## Verification

You can verify the fix is working by:

1. Checking the worker logs for messages like:
   ```
   Applied JSON serialization patches to fix NaN handling
   Patched job_status function to handle NaN values correctly
   ```

2. Submitting a new analysis job and confirming it completes successfully

3. Running the test script to validate the serialization:
   ```bash
   python3 test_json_nan_fix.py
   ```

## Impact

This fix resolves:
1. Jobs stuck in "started" state
2. JSON parsing errors in the client application
3. User-facing timeout errors after long polling periods

The system should now reliably process jobs and return properly formatted results even when the data contains special floating-point values like NaN.

## Additional Notes

If jobs that are currently stuck in the "started" state need to be cleared, use the job cleanup endpoint:
```
GET /cleanup_jobs
```

This will clear any stale jobs and allow the system to process new requests properly.
