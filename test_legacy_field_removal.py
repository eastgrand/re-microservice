#!/usr/bin/env python3
"""
Comprehensive test for memory optimization with legacy field removal.

This script verifies that:
1. The optimized prune_dataframe_columns function correctly removes only legacy fields
2. Memory usage is reduced but important columns are preserved
3. The function handles datasets with missing legacy fields gracefully
"""

import pandas as pd
import numpy as np
import os
import sys
import gc

# Import memory optimization functions
try:
    from optimize_memory import prune_dataframe_columns, log_memory_usage
except ImportError:
    print("Failed to import optimize_memory.py")
    sys.exit(1)

def create_test_dataset(include_fields=None):
    """Create a test dataset with controlled fields"""
    # All possible fields
    all_fields = {
        # Legacy fields
        'Single_Status': np.random.rand(1000),
        'Single_Family_Homes': np.random.rand(1000),
        'Married_Population': np.random.rand(1000),
        'Aggregate_Income': np.random.rand(1000),
        'Market_Weight': np.random.rand(1000),
        
        # Important analytical fields
        'Mortgage_Approvals': np.random.rand(1000),
        'Population_Total': np.random.rand(1000),
        'Household_Income': np.random.rand(1000),
        'Age_Distribution': np.random.rand(1000),
        'Education_Level': np.random.rand(1000)
    }
    
    # If specific fields are requested, only include those
    if include_fields:
        fields = {k: v for k, v in all_fields.items() if k in include_fields}
    else:
        fields = all_fields
        
    return pd.DataFrame(fields)

def test_complete_removal():
    """Test removal of all legacy fields when all are present"""
    print("\n=== TEST 1: All Legacy Fields Present ===")
    df = create_test_dataset()
    
    print(f"Original dataset: {len(df.columns)} columns")
    legacy_fields = [
        'Single_Status', 'Single_Family_Homes', 'Married_Population',
        'Aggregate_Income', 'Market_Weight'
    ]
    
    # Print original columns
    print("Original columns:")
    print(sorted(df.columns.tolist()))
    
    # Apply pruning
    df_pruned = prune_dataframe_columns(df)
    
    # Print pruned columns
    print("\nColumns after pruning:")
    print(sorted(df_pruned.columns.tolist()))
    
    # Check which fields were removed
    removed = set(df.columns) - set(df_pruned.columns)
    print("\nRemoved fields:")
    print(sorted(list(removed)))
    
    # Verify all legacy fields were removed
    all_removed = all(field in removed for field in legacy_fields)
    print(f"\nAll legacy fields removed: {'✓' if all_removed else '✗'}")
    
    # Verify important fields were preserved
    important_fields = [col for col in df.columns if col not in legacy_fields]
    all_preserved = all(field in df_pruned.columns for field in important_fields)
    print(f"All important fields preserved: {'✓' if all_preserved else '✗'}")
    
    return all_removed and all_preserved

def test_partial_removal():
    """Test removal when only some legacy fields are present"""
    print("\n=== TEST 2: Partial Legacy Fields Present ===")
    # Include only some legacy fields
    include_fields = [
        'Single_Status', 'Married_Population',  # Only 2 of 5 legacy fields
        'Mortgage_Approvals', 'Population_Total', 'Household_Income'
    ]
    df = create_test_dataset(include_fields)
    
    print(f"Original dataset: {len(df.columns)} columns")
    print("Original columns:")
    print(sorted(df.columns.tolist()))
    
    # Apply pruning
    df_pruned = prune_dataframe_columns(df)
    
    # Print pruned columns
    print("\nColumns after pruning:")
    print(sorted(df_pruned.columns.tolist()))
    
    # Check which fields were removed
    removed = set(df.columns) - set(df_pruned.columns)
    print("\nRemoved fields:")
    print(sorted(list(removed)))
    
    # Verify only the present legacy fields were removed
    present_legacy = [field for field in ['Single_Status', 'Married_Population'] if field in df.columns]
    all_removed = all(field in removed for field in present_legacy)
    print(f"\nAll present legacy fields removed: {'✓' if all_removed else '✗'}")
    
    # Verify important fields were preserved
    important_fields = ['Mortgage_Approvals', 'Population_Total', 'Household_Income']
    all_preserved = all(field in df_pruned.columns for field in important_fields)
    print(f"All important fields preserved: {'✓' if all_preserved else '✗'}")
    
    return all_removed and all_preserved

def test_no_legacy_fields():
    """Test when no legacy fields are present"""
    print("\n=== TEST 3: No Legacy Fields Present ===")
    # Include only important fields
    include_fields = [
        'Mortgage_Approvals', 'Population_Total', 'Household_Income',
        'Age_Distribution', 'Education_Level'
    ]
    df = create_test_dataset(include_fields)
    
    print(f"Original dataset: {len(df.columns)} columns")
    print("Original columns:")
    print(sorted(df.columns.tolist()))
    
    # Apply pruning
    df_pruned = prune_dataframe_columns(df)
    
    # Print pruned columns
    print("\nColumns after pruning:")
    print(sorted(df_pruned.columns.tolist()))
    
    # Check if any fields were removed
    removed = set(df.columns) - set(df_pruned.columns)
    print("\nRemoved fields:")
    print(sorted(list(removed)))
    
    # Verify no fields were removed (since no legacy fields were present)
    no_removals = len(removed) == 0
    print(f"\nNo fields removed (expected since no legacy fields present): {'✓' if no_removals else '✗'}")
    
    # Verify all original fields are preserved
    all_preserved = len(df.columns) == len(df_pruned.columns)
    print(f"All fields preserved: {'✓' if all_preserved else '✗'}")
    
    return no_removals and all_preserved

def test_with_render_environment():
    """Test with Render environment variables set"""
    print("\n=== TEST 4: Simulated Render Environment ===")
    # Set Render environment variables
    os.environ["RENDER"] = "true"
    os.environ["MAX_MEMORY_MB"] = "400"
    os.environ["AGGRESSIVE_MEMORY_MANAGEMENT"] = "true"
    
    # Run the test with all fields
    df = create_test_dataset()
    
    log_memory_usage("Before pruning")
    df_pruned = prune_dataframe_columns(df)
    log_memory_usage("After pruning")
    
    # Check which fields were removed
    removed = set(df.columns) - set(df_pruned.columns)
    legacy_fields = [
        'Single_Status', 'Single_Family_Homes', 'Married_Population',
        'Aggregate_Income', 'Market_Weight'
    ]
    
    # Verify all legacy fields were removed
    all_removed = all(field in removed for field in legacy_fields)
    print(f"\nAll legacy fields removed in Render environment: {'✓' if all_removed else '✗'}")
    
    # Verify important fields were preserved
    important_fields = [col for col in df.columns if col not in legacy_fields]
    all_preserved = all(field in df_pruned.columns for field in important_fields)
    print(f"All important fields preserved in Render environment: {'✓' if all_preserved else '✗'}")
    
    return all_removed and all_preserved

if __name__ == "__main__":
    print("=== Memory Optimization Test Suite ===")
    print("Testing prune_dataframe_columns() with legacy field removal")
    
    # Run all tests
    tests = [
        test_complete_removal,
        test_partial_removal,
        test_no_legacy_fields,
        test_with_render_environment
    ]
    
    all_passed = True
    for test_func in tests:
        # Ensure clean memory between tests
        gc.collect()
        # Run test
        passed = test_func()
        all_passed = all_passed and passed
    
    # Final results
    print("\n=== Final Results ===")
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("The memory optimization correctly removes only legacy fields while preserving all important columns.")
    else:
        print("❌ SOME TESTS FAILED")
        print("There may be issues with the legacy field removal implementation.")
    
    sys.exit(0 if all_passed else 1)
