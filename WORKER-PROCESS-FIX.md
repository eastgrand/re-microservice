# SHAP WORKER PROCESS FIX

## Problem Diagnosed
After extensive testing, we've identified that the deployed SHAP microservice is successfully receiving and queueing jobs through Redis, but those jobs are getting stuck in the "started" state and never completing. This is causing the user interface to infinitely wait for analysis results that never arrive.

## Root Cause Analysis
1. **Job Processing:** Jobs are properly submitted and stored in Redis, as proven by successful responses from the `/analyze` endpoint.
2. **Worker Status:** The worker processes appear to be running, as they pick up jobs and mark them as "started".
3. **Processing Failure:** However, workers fail to complete the jobs, causing them to remain in the "started" state indefinitely.

Key indicators:
- The `/ping` endpoint responds correctly
- The `/admin/redis_ping` endpoint confirms Redis is connected
- Jobs are being submitted successfully with valid job IDs
- Jobs are transitioning to "started" state
- Jobs never transition to "finished" state

## Solution Components

### 1. Worker Process Monitoring (`worker_process_fix.py`)
This comprehensive fix enhances the RQ worker process handling in the following ways:

- **Improved job tracking**: Adds detailed timestamps for monitoring worker activity on each job
- **Timeouts detection**: Identifies jobs that have been in "started" state too long
- **Enhanced error handling**: Ensures proper job status updates, even in failure cases
- **Stale job cleanup**: Automatically detects and requeues or fails stuck jobs

### 2. Diagnostics Tool (`diagnose_worker.py`)
A diagnostic script that:
- Inspects the state of all job queues and registries
- Reports on active workers and their current tasks
- Identifies stuck jobs and their duration in that state
- Provides recommendations for fixing observed issues

### 3. Repair Utility (`repair_stuck_jobs.py`)
A script to fix stuck jobs that:
- Identifies jobs stuck in "started" state beyond a configurable timeout
- Safely moves them back to the queue for reprocessing
- Can force-requeue all started jobs when needed
- Respects actively running worker processes by default

### 4. Improved Deployment Configuration
Updates to `render.yaml` for better worker process handling:
- Worker starts with `--burst` flag for better cleanup on shutdown
- Performs stuck job cleanup during each deploy/restart
- Enhances environment variable handling
- Applies worker process monitoring automatically

## Implementation Plan

1. Apply `worker_process_fix.py` in the main application startup
2. Update the worker process configuration in `render.yaml`
3. Deploy the changes to Render
4. Run the diagnostic tool to verify the fix
5. Test end-to-end job processing

## Verification

After deploying these changes:
1. Jobs should complete normally
2. No jobs should remain stuck in "started" state
3. Any temporarily stuck jobs should be automatically recovered
4. The new `/admin/cleanup_jobs` endpoint can be used to manually trigger cleanup

## Long-term Monitoring

The fix includes logging enhancements that will make it easier to:
1. Track worker activity and job processing time
2. Detect abnormally long-running jobs
3. Identify memory or performance issues in the worker processes
4. Monitor the health of the Redis connection and job queues
