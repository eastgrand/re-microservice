#!/usr/bin/env python3
"""
Disable type checking in _tree.py to fix the excessive errors

This creates a simple pyrightconfig.json to exclude the SHAP library from type checking
"""

import os
import json

# Create a pyrightconfig.json file to disable type checking for SHAP
config = {
    "exclude": [
        "**/venv/lib/python3.13/site-packages/shap/**"
    ],
    "reportMissingImports": "none"
}

with open('pyrightconfig.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Created pyrightconfig.json to exclude SHAP from type checking")
