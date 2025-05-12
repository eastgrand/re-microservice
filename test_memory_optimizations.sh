#!/bin/bash
# Script to test memory optimization changes

echo "Testing memory optimizations..."

# Set environment variables to simulate Render environment
export RENDER=true
export MAX_MEMORY_MB=400
export AGGRESSIVE_MEMORY_MANAGEMENT=true
export DEFAULT_TARGET=Mortgage_Approvals

# Monitor memory usage
check_memory() {
  if command -v ps &> /dev/null; then
    ps -o rss= -p $$ | awk '{print "Current memory usage: " $1/1024 " MB"}'
  else
    echo "ps command not available, can't check memory"
  fi
}

echo "Current memory configuration:"
echo "MAX_MEMORY_MB: $MAX_MEMORY_MB"
echo "AGGRESSIVE_MEMORY_MANAGEMENT: $AGGRESSIVE_MEMORY_MANAGEMENT"

# Test the optimize_memory.py script directly
echo "Testing optimize_memory.py..."
check_memory
python -c "from optimize_memory import log_memory_usage, prune_dataframe_columns; import pandas as pd; log_memory_usage('Before test'); df = pd.read_csv('data/cleaned_data.csv', nrows=1000); df = prune_dataframe_columns(df); print(f'Columns after pruning: {list(df.columns)}'); log_memory_usage('After test')"

# Check if app can load with the optimizations
echo "Testing app.py loading..."
check_memory
python -c "from app import app; print('App loaded successfully')"

# Show the legacy fields that are removed
echo -e "\nLegacy fields being removed:"
echo "- Single_Status (SUM_ECYMARNMCL)"
echo "- Single_Family_Homes (SUM_ECYSTYSING)"
echo "- Married_Population (SUM_ECYMARM)" 
echo "- Aggregate_Income (SUM_HSHNIAGG)"
echo "- Market_Weight (Sum_Weight)"

echo -e "\nMemory optimization test completed"
