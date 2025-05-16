# SHAP Microservice Memory Optimization Guide

**Date: May 15, 2025**

## Overview

This guide documents the memory optimization changes made to improve the performance of the SHAP microservice while operating within the constraints of a Render starter plan (512MB memory, 0.5 CPU).

## Applied Optimizations

The following optimizations have been implemented to improve performance and processing speed:

1. **Increased batch size**:
   - Worker batch size increased from 300 to 500 rows
   - This allows processing more data in each batch, reducing overhead

2. **Adjusted memory management**:
   - Disabled aggressive memory management mode
   - Increased memory threshold from 450MB to 475MB for standard mode
   - Increased threshold from 400MB to 450MB for aggressive mode

3. **Preserved safety margins**:
   - Maintained approximately 37MB safety buffer (512MB - 475MB)
   - This ensures the service doesn't exceed Render's memory limits

## Expected Improvements

With these optimizations, you should see:

1. **Faster processing**: Jobs should complete in less time due to larger batch sizes
2. **Fewer batches**: For your 1668-row dataset, the number of batches will decrease from 6 to 4
3. **More efficient resource usage**: The service will utilize more of the available memory

## Monitoring Performance

A performance monitoring script has been created to help evaluate the impact of these optimizations:

```bash
python performance_monitor.py --api-key YOUR_API_KEY --rows 500 --columns 134
```

This script will:
- Submit test jobs with configurable data sizes
- Measure processing times
- Report memory usage statistics

## Current Configuration

Current memory usage with a 1668-row × 134-column dataset:

- **Batch size**: 500 rows
- **Peak memory usage**: ~242MB (observed)
- **Memory threshold**: 475MB
- **Available memory**: 512MB
- **Safety buffer**: ~37MB

## Future Considerations

If you find that the current optimizations don't meet your performance needs, consider:

1. **Upgrading to a larger plan**: The 2GB/1CPU plan would allow processing the entire dataset at once
2. **Further batch size increases**: You could potentially increase batch size to 600-700 with careful monitoring
3. **Custom SHAP implementation**: Develop a more memory-efficient SHAP calculation algorithm

## Troubleshooting

If you encounter any issues after applying these optimizations:

1. **Memory errors**: If you see memory-related crashes, revert the changes by setting:
   - `AGGRESSIVE_MEMORY_MANAGEMENT=true`
   - `SHAP_MAX_BATCH_SIZE=300`

2. **Slow processing**: If jobs are still processing slowly, check:
   - Render Dashboard for CPU usage (could be CPU-bound rather than memory-bound)
   - Network latency between your application and the SHAP microservice
   - Redis queue backlog (if many jobs are waiting in queue)

## Implementation Details

The changes were made in:

1. `render.yaml`: Updated environment variables for the worker service
2. `optimize_memory.py`: Adjusted memory thresholds for both normal and aggressive modes
3. Added `performance_monitor.py`: For evaluating the performance impact

## Deployment

To deploy these optimizations:

```bash
./deploy_memory_optimization.sh
```

This script will:
1. Commit the optimization changes
2. Push them to GitHub
3. Trigger automatic deployment on Render
