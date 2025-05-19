"""
Data versioning module for SHAP/XGBoost microservice.
Tracks dataset and model versions to ensure reproducibility of analysis results.
"""
import os
import json
import hashlib
import datetime
from pathlib import Path
import pandas as pd


class DataVersionTracker:
    """
    Tracks versions of datasets and models used in the SHAP/XGBoost microservice.
    """
    
    def __init__(self, version_file="data/versions.json"):
        """Initialize the version tracker."""
        self.version_file = version_file
        self._ensure_version_file_exists()
        
    def _ensure_version_file_exists(self):
        """Create version file if it doesn't exist."""
        os.makedirs(os.path.dirname(self.version_file), exist_ok=True)
        if not os.path.exists(self.version_file):
            with open(self.version_file, 'w') as f:
                json.dump({
                    "datasets": {},
                    "models": {}
                }, f, indent=2)
    
    def _calculate_hash(self, filepath):
        """Calculate MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _calculate_dataframe_hash(self, df):
        """Calculate hash of a pandas dataframe."""
        return hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
    
    def register_dataset(self, dataset_path, description=None, source=None):
        """
        Register a new dataset version.
        
        Args:
            dataset_path: Path to the dataset file
            description: Optional description of this dataset version
            source: Optional source information
            
        Returns:
            version_id: Unique identifier for this dataset version
        """
        # Load dataframe to calculate stats
        df = pd.read_csv(dataset_path)
        
        # Calculate hash and stats
        file_hash = self._calculate_hash(dataset_path)
        data_hash = self._calculate_dataframe_hash(df)
        
        # Create version info
        version_id = f"dataset_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        timestamp = datetime.datetime.now().isoformat()
        
        version_info = {
            "path": dataset_path,
            "file_hash": file_hash,
            "data_hash": data_hash,
            "timestamp": timestamp,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "description": description or "No description provided",
            "source": source or "Unknown"
        }
        
        # Update version file
        with open(self.version_file, 'r') as f:
            versions = json.load(f)
        
        versions["datasets"][version_id] = version_info
        
        with open(self.version_file, 'w') as f:
            json.dump(versions, f, indent=2)
        
        return version_id
    
    def register_model(self, model_path, dataset_version_id, feature_names_path=None, metrics=None):
        """
        Register a new model version.
        
        Args:
            model_path: Path to the model file
            dataset_version_id: ID of the dataset used to train this model
            feature_names_path: Optional path to feature names file
            metrics: Optional dictionary of model performance metrics
            
        Returns:
            version_id: Unique identifier for this model version
        """
        # Calculate hash
        model_hash = self._calculate_hash(model_path)
        
        # Create version info
        version_id = f"model_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        timestamp = datetime.datetime.now().isoformat()
        
        version_info = {
            "path": model_path,
            "hash": model_hash,
            "timestamp": timestamp,
            "dataset_version_id": dataset_version_id,
            "metrics": metrics or {}
        }
        
        # Add feature names info if provided
        if feature_names_path and os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            version_info["feature_names"] = feature_names
            version_info["feature_names_path"] = feature_names_path
        
        # Update version file
        with open(self.version_file, 'r') as f:
            versions = json.load(f)
        
        versions["models"][version_id] = version_info
        
        with open(self.version_file, 'w') as f:
            json.dump(versions, f, indent=2)
        
        return version_id
    
    def get_latest_dataset(self):
        """Get the latest registered dataset version."""
        with open(self.version_file, 'r') as f:
            versions = json.load(f)
        
        if not versions["datasets"]:
            return None
        
        # Find latest version by timestamp
        latest_version = max(
            versions["datasets"].items(),
            key=lambda x: x[1]["timestamp"]
        )
        
        return latest_version[0], latest_version[1]
    
    def get_latest_model(self):
        """Get the latest registered model version."""
        with open(self.version_file, 'r') as f:
            versions = json.load(f)
        
        if not versions["models"]:
            return None
        
        # Find latest version by timestamp
        latest_version = max(
            versions["models"].items(),
            key=lambda x: x[1]["timestamp"]
        )
        
        return latest_version[0], latest_version[1]
    
    def list_all_versions(self):
        """List all registered versions."""
        with open(self.version_file, 'r') as f:
            versions = json.load(f)
        
        return versions


# When run directly, show all registered versions
if __name__ == "__main__":
    tracker = DataVersionTracker()
    versions = tracker.list_all_versions()
    
    print("\nRegistered Dataset Versions:")
    for version_id, info in versions["datasets"].items():
        print(f"  - {version_id}: {info['path']} ({info['timestamp']})")
        print(f"    {info['row_count']} rows, {info['column_count']} columns")
        print(f"    Description: {info['description']}")
    
    print("\nRegistered Model Versions:")
    for version_id, info in versions["models"].items():
        print(f"  - {version_id}: {info['path']} ({info['timestamp']})")
        print(f"    Trained on dataset: {info['dataset_version_id']}")
        if "metrics" in info and info["metrics"]:
            metrics_str = ", ".join([f"{k}: {v}" for k, v in info["metrics"].items()])
            print(f"    Metrics: {metrics_str}")
