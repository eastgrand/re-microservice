#!/usr/bin/env python3
"""
Test script for data versioning functionality.
Validates that the DataVersionTracker correctly tracks dataset and model versions.
"""
import os
import pandas as pd
import json
import numpy as np
import pickle
from data_versioning import DataVersionTracker

def print_header(message):
    print("\n" + "="*70)
    print(f" {message}")
    print("="*70)

def print_success(message):
    print(f"✅ {message}")
    
def print_error(message):
    print(f"❌ {message}")

def run_tests():
    """Run tests for data versioning"""
    print_header("Testing Data Versioning Functionality")
    
    # Make sure we have a clean start
    if os.path.exists('data/versions_test.json'):
        os.remove('data/versions_test.json')
        
    tracker = DataVersionTracker('data/versions_test.json')
    
    # Test 1: Create and register a test dataset
    print("\nTest 1: Creating and registering a test dataset")
    
    # Create a small test dataset
    test_data = pd.DataFrame({
        'zip_code': ['10001', '20002', '30003', '40004', '50005'],
        'Income': [70000, 85000, 65000, 90000, 55000],
        'Nike_Sales': [12000, 15000, 11000, 16000, 9000]
    })
    
    # Save to CSV
    test_dataset_path = 'data/test_dataset.csv'
    test_data.to_csv(test_dataset_path, index=False)
    
    # Register dataset
    dataset_version_id = tracker.track_dataset(
        test_dataset_path,
        description="Test dataset for versioning",
        source="Test script"
    )
    
    if dataset_version_id and dataset_version_id.startswith('dataset_'):
        print_success(f"Dataset registered with ID: {dataset_version_id}")
    else:
        print_error("Failed to register dataset properly")
        
    # Test 2: Create and register a test model
    print("\nTest 2: Creating and registering a test model")
    
    # Create a dummy model (just a dictionary for testing)
    dummy_model = {
        'name': 'test_model',
        'parameters': {'max_depth': 5, 'learning_rate': 0.1},
        'accuracy': 0.85
    }
    
    # Save model to file
    test_model_path = 'models/test_model.pkl'
    with open(test_model_path, 'wb') as f:
        pickle.dump(dummy_model, f)
    
    # Create feature names file
    test_feature_names_path = 'models/test_features.txt'
    with open(test_feature_names_path, 'w') as f:
        f.write("Income\n")
        
    # Register model
    model_version_id = tracker.track_model(
        test_model_path,
        dataset_version_id,
        feature_names_path=test_feature_names_path,
        metrics={'rmse': 0.25, 'r2': 0.8}
    )
    
    if model_version_id and model_version_id.startswith('model_'):
        print_success(f"Model registered with ID: {model_version_id}")
    else:
        print_error("Failed to register model properly")

    # Test 3: Retrieve and validate version information
    print("\nTest 3: Retrieving version information")
    
    # Get latest dataset
    latest_dataset = tracker.get_latest_dataset()
    if latest_dataset and latest_dataset[0] == dataset_version_id:
        print_success("Retrieved correct latest dataset version")
    else:
        print_error("Failed to retrieve correct latest dataset version")
        
    # Get latest model
    latest_model = tracker.get_latest_model()
    if latest_model and latest_model[0] == model_version_id:
        print_success("Retrieved correct latest model version")
    else:
        print_error("Failed to retrieve correct latest model version")
    
    # Test 4: Verify relationships
    print("\nTest 4: Verifying relationships between model and dataset")
    
    # Get all versions
    all_versions = tracker.list_all_versions()
    
    # Check if model references correct dataset
    model_info = all_versions.get('models', {}).get(model_version_id, {})
    dataset_reference = model_info.get('dataset_version_id', '')
    
    if dataset_reference == dataset_version_id:
        print_success("Model correctly references the dataset it was trained on")
    else:
        print_error(f"Model references incorrect dataset: {dataset_reference}")
        
    # Clean up test files
    print("\nCleaning up test files...")
    os.remove('data/versions_test.json')
    os.remove('data/test_dataset.csv')
    os.remove('models/test_model.pkl')
    os.remove('models/test_features.txt')
    
    print_success("All test files cleaned up")
    
    print("\nTests completed!")

if __name__ == "__main__":
    run_tests()
