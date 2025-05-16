#!/bin/bash
# Wrapper script to apply JSON NaN serialization fix before starting the app
# This will be used in render.yaml

# Apply the JSON NaN serialization fix
echo "Applying JSON NaN serialization fix..."
python3 patch_app_with_fixes.py

# Start the app with the original command
echo "Starting app with original command..."
exec "$@"
