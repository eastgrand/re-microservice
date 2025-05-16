#!/bin/bash
# Wrapper script to apply JSON NaN serialization fix before starting the app
# This will be used in render.yaml

# Apply the JSON NaN serialization fix
echo "Applying JSON NaN serialization fix..."
python3 patch_app_with_fixes.py

# Apply enhanced diagnostics (if available)
if [ -f "enhanced_diagnostics.py" ]; then
    echo "Applying enhanced diagnostics..."
    python3 enhanced_diagnostics.py
    echo "Enhanced diagnostics applied"
fi

# Start the app with the original command
echo "Starting app with original command..."
exec "$@"
