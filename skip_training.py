#!/usr/bin/env python3
"""
Skip Training Flag Manager

This script helps manage whether model training should be skipped during deployment.
It creates or removes a file that acts as a flag to indicate whether training should be skipped.

Usage:
  python skip_training.py enable   # Enable skipping of model training
  python skip_training.py disable  # Disable skipping of model training
  python skip_training.py status   # Check current status
"""

import os
import sys
import argparse

# Flag file that indicates training should be skipped
SKIP_TRAINING_FLAG = ".skip_training"

def enable_skip_training():
    """Create the flag file to skip training"""
    with open(SKIP_TRAINING_FLAG, 'w') as f:
        f.write("This file indicates that model training should be skipped during deployment.\n")
        f.write("Delete this file to re-enable training, or run 'python skip_training.py disable'\n")
    
    print(f"✅ Model training will be SKIPPED during deployment.")
    print(f"   Flag file created: {SKIP_TRAINING_FLAG}")
    check_model_exists()

def disable_skip_training():
    """Remove the flag file to enable training"""
    if os.path.exists(SKIP_TRAINING_FLAG):
        os.remove(SKIP_TRAINING_FLAG)
        print(f"✅ Model training will be PERFORMED during deployment.")
        print(f"   Flag file removed: {SKIP_TRAINING_FLAG}")
    else:
        print(f"ℹ️ Model training is already set to be performed during deployment.")
        print(f"   (Flag file {SKIP_TRAINING_FLAG} doesn't exist)")

def check_skip_training():
    """Check if training should be skipped"""
    if os.path.exists(SKIP_TRAINING_FLAG):
        print(f"✅ Model training will be SKIPPED during deployment.")
        print(f"   Flag file exists: {SKIP_TRAINING_FLAG}")
        check_model_exists()
    else:
        print(f"ℹ️ Model training will be PERFORMED during deployment.")
        print(f"   (Flag file {SKIP_TRAINING_FLAG} doesn't exist)")

def check_model_exists():
    """Check if model files exist when skipping training"""
    model_path = os.environ.get('MODEL_PATH', 'models/xgboost_model.pkl')
    feature_path = os.environ.get('FEATURE_NAMES_PATH', 'models/feature_names.txt')
    
    if not os.path.exists(model_path):
        print(f"⚠️  WARNING: Model file not found at {model_path}")
        print(f"   You need to train the model locally before deploying.")
        print(f"   Run: python train_model.py")
    else:
        print(f"✅ Model file found at {model_path}")

    if not os.path.exists(feature_path):
        print(f"⚠️  WARNING: Feature names file not found at {feature_path}")
        print(f"   You need to train the model locally before deploying.")
        print(f"   Run: python train_model.py")
    else:
        print(f"✅ Feature names file found at {feature_path}")

def main():
    parser = argparse.ArgumentParser(description='Manage model training behavior during deployment')
    parser.add_argument('action', choices=['enable', 'disable', 'status'], 
                        help='Action to perform: enable or disable skipping training, or check status')
    
    if len(sys.argv) < 2:
        parser.print_help()
        return
        
    args = parser.parse_args()
    
    if args.action == 'enable':
        enable_skip_training()
    elif args.action == 'disable':
        disable_skip_training()
    else:  # status
        check_skip_training()
        
if __name__ == "__main__":
    main()
