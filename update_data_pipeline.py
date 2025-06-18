#!/usr/bin/env python3
"""
Complete Data Pipeline Update Script for SHAP Microservice

This script automates the entire process of updating the data pipeline when new data arrives.
It handles data cleaning, model training, SHAP precalculation, and version tracking.

Usage:
    python update_data_pipeline.py --data-file path/to/new_data.csv
    python update_data_pipeline.py --use-existing  # Use existing nesto_merge_0.csv
"""

import os
import sys
import logging
import argparse
import shutil
import subprocess
from datetime import datetime
from data_versioning import DataVersionTracker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data-pipeline-update")

class DataPipelineUpdater:
    def __init__(self):
        self.version_tracker = DataVersionTracker()
        self.script_dir = os.path.dirname(__file__)
        
    def backup_existing_data(self):
        """Backup existing data and models"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = f"backups/backup_{timestamp}"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Backup key files
        files_to_backup = [
            'data/cleaned_data.csv',
            'models/xgboost_model.pkl',
            'models/feature_names.txt'
        ]
        
        for file_path in files_to_backup:
            if os.path.exists(file_path):
                shutil.copy2(file_path, backup_dir)
                logger.info(f"Backed up {file_path} to {backup_dir}")
        
        # Backup precalculated SHAP values
        if os.path.exists('precalculated'):
            shutil.copytree('precalculated', f"{backup_dir}/precalculated", dirs_exist_ok=True)
            logger.info(f"Backed up precalculated SHAP values")
        
        return backup_dir
    
    def install_new_data(self, data_file_path):
        """Install new data file as nesto_merge_0.csv"""
        target_path = 'data/nesto_merge_0.csv'
        
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"Data file not found: {data_file_path}")
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Copy new data file
        shutil.copy2(data_file_path, target_path)
        logger.info(f"Installed new data file: {data_file_path} -> {target_path}")
        
        # Track the new dataset
        dataset_info = {
            "source_file": data_file_path,
            "target_file": target_path,
            "file_size": os.path.getsize(target_path),
            "status": "installed"
        }
        version_id = self.version_tracker.track_dataset(dataset_info)
        logger.info(f"Tracked new dataset version: {version_id}")
        
        return version_id
    
    def clean_and_preprocess_data(self):
        """Run data cleaning and preprocessing"""
        logger.info("Starting data cleaning and preprocessing...")
        
        try:
            # Import and run the data mapping
            from map_nesto_data import load_and_preprocess_data
            df = load_and_preprocess_data()
            
            logger.info(f"Data preprocessing completed. Shape: {df.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            return False
    
    def train_model(self):
        """Train the XGBoost model on new data"""
        logger.info("Starting model training...")
        
        try:
            # Run the training script
            result = subprocess.run([
                sys.executable, 'train_model.py'
            ], capture_output=True, text=True, cwd=self.script_dir)
            
            if result.returncode == 0:
                logger.info("Model training completed successfully")
                
                # Track the new model
                model_info = {
                    "model_file": "models/xgboost_model.pkl",
                    "features_file": "models/feature_names.txt",
                    "training_output": result.stdout[-500:],  # Last 500 chars
                    "status": "trained"
                }
                version_id = self.version_tracker.track_model(model_info)
                logger.info(f"Tracked new model version: {version_id}")
                
                return True
            else:
                logger.error(f"Model training failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Model training error: {e}")
            return False
    
    def precalculate_shap_values(self):
        """Precalculate SHAP values for fast responses"""
        logger.info("Starting SHAP precalculation...")
        
        try:
            # Run the SHAP precalculation script
            result = subprocess.run([
                sys.executable, 'precalculate_shap.py'
            ], capture_output=True, text=True, cwd=self.script_dir)
            
            if result.returncode == 0:
                logger.info("SHAP precalculation completed successfully")
                return True
            else:
                logger.error(f"SHAP precalculation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"SHAP precalculation error: {e}")
            return False
    
    def verify_pipeline(self):
        """Verify that all pipeline components are working"""
        logger.info("Verifying pipeline integrity...")
        
        required_files = [
            'data/cleaned_data.csv',
            'models/xgboost_model.pkl',
            'models/feature_names.txt'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False
        
        # Check if precalculated SHAP values exist
        if not os.path.exists('precalculated') or not os.listdir('precalculated'):
            logger.warning("No precalculated SHAP values found")
        
        logger.info("Pipeline verification completed successfully")
        return True
    
    def run_full_pipeline(self, data_file_path=None):
        """Run the complete data pipeline update"""
        logger.info("🚀 Starting complete data pipeline update...")
        
        try:
            # Step 1: Backup existing data
            backup_dir = self.backup_existing_data()
            logger.info(f"✅ Backup created: {backup_dir}")
            
            # Step 2: Install new data (if provided)
            if data_file_path:
                self.install_new_data(data_file_path)
                logger.info("✅ New data installed")
            
            # Step 3: Clean and preprocess data
            if not self.clean_and_preprocess_data():
                raise Exception("Data preprocessing failed")
            logger.info("✅ Data cleaning completed")
            
            # Step 4: Train model
            if not self.train_model():
                raise Exception("Model training failed")
            logger.info("✅ Model training completed")
            
            # Step 5: Precalculate SHAP values
            if not self.precalculate_shap_values():
                logger.warning("⚠️ SHAP precalculation failed, but continuing...")
            else:
                logger.info("✅ SHAP precalculation completed")
            
            # Step 6: Verify pipeline
            if not self.verify_pipeline():
                raise Exception("Pipeline verification failed")
            logger.info("✅ Pipeline verification completed")
            
            logger.info("🎉 Data pipeline update completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline update failed: {e}")
            logger.error(f"💡 You can restore from backup: {backup_dir}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Update SHAP microservice data pipeline')
    parser.add_argument('--data-file', help='Path to new data file')
    parser.add_argument('--use-existing', action='store_true', 
                       help='Use existing nesto_merge_0.csv file')
    parser.add_argument('--backup-only', action='store_true',
                       help='Only create backup, do not update pipeline')
    
    args = parser.parse_args()
    
    if not args.use_existing and not args.data_file and not args.backup_only:
        parser.error('Must specify either --data-file, --use-existing, or --backup-only')
    
    updater = DataPipelineUpdater()
    
    if args.backup_only:
        backup_dir = updater.backup_existing_data()
        logger.info(f"Backup created: {backup_dir}")
        return
    
    data_file = args.data_file if args.data_file else None
    success = updater.run_full_pipeline(data_file)
    
    if success:
        logger.info("✅ Pipeline update completed successfully!")
        logger.info("🔄 Restart the microservice to use the new data and model")
    else:
        logger.error("❌ Pipeline update failed!")
        sys.exit(1)

if __name__ == '__main__':
    main() 