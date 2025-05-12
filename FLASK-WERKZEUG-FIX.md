# Flask-Werkzeug Compatibility Fix

This document explains the Flask/Werkzeug compatibility issue encountered during deployment and how it was fixed.

## Issue

During deployment to Render, we encountered the following error:

```
ImportError: cannot import name 'url_quote' from 'werkzeug.urls'
```

This occurred because Flask 2.2.3 was being installed with Werkzeug 3.1.3, which was incompatible. Flask 2.2.x requires Werkzeug < 3.0.0 because the `url_quote` function was removed in Werkzeug 3.0.0.

## Fix

We implemented the following fixes to resolve the compatibility issue:

1. **Pin Werkzeug version in requirements.txt**:
   ```
   flask==2.2.3
   werkzeug==2.2.3  # Must be compatible with Flask 2.2.3
   ```

2. **Created a compatibility check script**: 
   - `fix_flask_werkzeug.py` checks the Flask and Werkzeug versions at build time
   - If an incompatible combination is detected, it installs a compatible version

3. **Updated render.yaml**:
   ```yaml
   buildCommand: pip install -r requirements.txt && python fix_flask_werkzeug.py && python patch_shap.py && python train_model.py
   ```

## Prevention

To prevent similar issues in the future:

1. Always pin all package versions in `requirements.txt`
2. When updating Flask or other major dependencies, check their dependency requirements
3. Include compatibility checks in your build process

## Related Documentation

- [Flask Documentation](https://flask.palletsprojects.com/en/2.2.x/changes/)
- [Werkzeug Documentation](https://werkzeug.palletsprojects.com/en/2.2.x/changes/)
