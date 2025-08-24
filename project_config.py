# Project Configuration Module
# Handles target variable, schema, and data loading for the SHAP microservice

import os
import logging
from typing import Dict, Any

import pandas as pd

# ---------------------------------------------------------------------------
# PROJECT CONFIGURATION
# This module provides project-specific configuration including target variables,
# schema mappings, and data loading functionality for the SHAP microservice.
# 
# Current Project: Red Bull Energy Drinks Market Analysis
# Domain: Energy Drinks / Functional Beverages
# Data Source: ArcGIS FeatureServer with percentage-based usage data
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Public constants expected by app.py -----------------------------------------------------------

# Target variable for Red Bull energy drinks analysis
# Updated from H&R Block tax services to Red Bull energy drink usage percentage
TARGET_VARIABLE: str = "MP12207A_B_P"  # Red Bull usage percentage (2025 data)

# Canonical <field_code -> metadata> mapping.  The micro-service only needs
# this for alias resolution; an *empty* dict is acceptable because the fallback
# logic in app.py resolves fields directly.  Populate as needed.
MASTER_SCHEMA: Dict[str, Dict[str, Any]] = {}

# Public helpers expected by app.py -------------------------------------------------------------

def load_and_preprocess_data() -> pd.DataFrame:
    """Load (or regenerate) `data/cleaned_data.csv` and return it as a DataFrame.

    The original implementation rebuilt the file from raw sources.  For the
    current deployment we adopt a best-effort strategy:
        1. If `cleaned_data.csv` exists, simply load and return it.
        2. Otherwise create an **empty** CSV so that downstream code does not
           crash on missing files and return an empty DataFrame.

    This is sufficient for the service to start; analytical endpoints may
    still require proper data so the full pipeline should be restored when
    convenient.
    """
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, "data")
    cleaned_path = os.path.join(data_dir, "cleaned_data.csv")

    try:
        if os.path.exists(cleaned_path):
            logger.info("Loading existing cleaned_data.csv (%s)", cleaned_path)
            return pd.read_csv(cleaned_path)
        else:
            logger.warning("cleaned_data.csv not found – creating empty placeholder at %s", cleaned_path)
            os.makedirs(data_dir, exist_ok=True)
            # Create an empty CSV with no rows/columns
            pd.DataFrame().to_csv(cleaned_path, index=False)
            return pd.DataFrame()
    except Exception as exc:
        logger.error("Failed to load or create cleaned_data.csv: %s", exc)
        # Return an empty DataFrame to prevent startup failure
        return pd.DataFrame()


def initialize_schema(df: pd.DataFrame | None = None) -> None:
    """Placeholder for dynamic schema initialisation.

    The real implementation would analyse *df* and extend `MASTER_SCHEMA` with
    inferred fields.  For now we simply log the call so that downstream code
    continues to execute.
    """
    field_count = len(df.columns) if df is not None else 0
    logger.info("initialize_schema() placeholder called – received DataFrame with %d columns", field_count)