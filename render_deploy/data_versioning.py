import os
import json
import logging
from datetime import datetime

logger = logging.getLogger("shap-microservice")

class DataVersionTracker:
    def __init__(self, version_dir="versions"):
        self.version_dir = version_dir
        os.makedirs(version_dir, exist_ok=True)
        self.dataset_versions = {}
        self.model_versions = {}
        self._load_versions()

    def _load_versions(self):
        """Load existing versions from disk"""
        try:
            if os.path.exists(os.path.join(self.version_dir, "dataset_versions.json")):
                with open(os.path.join(self.version_dir, "dataset_versions.json"), "r") as f:
                    self.dataset_versions = json.load(f)
            
            if os.path.exists(os.path.join(self.version_dir, "model_versions.json")):
                with open(os.path.join(self.version_dir, "model_versions.json"), "r") as f:
                    self.model_versions = json.load(f)
        except Exception as e:
            logger.error(f"Error loading versions: {str(e)}")
            self.dataset_versions = {}
            self.model_versions = {}

    def _save_versions(self):
        """Save versions to disk"""
        try:
            with open(os.path.join(self.version_dir, "dataset_versions.json"), "w") as f:
                json.dump(self.dataset_versions, f)
            
            with open(os.path.join(self.version_dir, "model_versions.json"), "w") as f:
                json.dump(self.model_versions, f)
        except Exception as e:
            logger.error(f"Error saving versions: {str(e)}")

    def track_dataset(self, dataset_info):
        """Track a new dataset version"""
        version_id = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_info["timestamp"] = datetime.now().isoformat()
        self.dataset_versions[version_id] = dataset_info
        self._save_versions()
        return version_id

    def track_model(self, model_info):
        """Track a new model version"""
        version_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_info["timestamp"] = datetime.now().isoformat()
        self.model_versions[version_id] = model_info
        self._save_versions()
        return version_id

    def get_latest_dataset(self):
        """Get the latest dataset version"""
        if not self.dataset_versions:
            return None
        latest_id = max(self.dataset_versions.keys())
        return latest_id, self.dataset_versions[latest_id]

    def get_latest_model(self):
        """Get the latest model version"""
        if not self.model_versions:
            return None
        latest_id = max(self.model_versions.keys())
        return latest_id, self.model_versions[latest_id]

    def list_all_versions(self):
        """List all tracked versions"""
        return {
            "datasets": self.dataset_versions,
            "models": self.model_versions
        } 