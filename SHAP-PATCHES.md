# SHAP Library Patches Documentation

This document provides a comprehensive overview of patches applied to the SHAP library in this microservice to address compatibility issues with Python 3.13 and newer versions of NumPy.

## Issues Addressed

### 1. Import Error Handling

#### Pyspark Import Issue
**Location**: `shap/explainers/_tree.py`, line ~30
**Problem**: When pyspark is not available, the code records the import error but doesn't set `pyspark = None`, leading to potential NameErrors later.
**Fix**: Added `pyspark = None` fallback after the import error.

```python
try:
    import pyspark
except ImportError as e:
    record_import_error("pyspark", "PySpark could not be imported!", e)
    pyspark = None  # Added to prevent NameError
```

#### Catboost Import Issue
**Location**: `shap/explainers/_tree.py`, line ~365
**Problem**: Direct import of catboost without proper error handling, potentially causing errors when catboost is not installed.
**Fix**: Implemented try/except with a flag to safely check for catboost availability.

```python
try:
    import catboost
    catboost_imported = True
except ImportError:
    catboost_imported = False
    
if catboost_imported and type(X) != catboost.Pool:
    X = catboost.Pool(X, cat_features=self.model.cat_feature_indices)
```

### 2. NumPy API Compatibility

#### bool8 Deprecation
**Problem**: NumPy's `bool8` type is deprecated in favor of `bool_`.
**Fix**: Replaced all occurrences of `np.bool8` with `np.bool_`.

#### obj2sctype Deprecation
**Problem**: NumPy's `obj2sctype` function is deprecated.
**Fix**: Replaced calls like `np.obj2sctype(dtype)` with `np.dtype(dtype).type`.

## Tools Created

### 1. Verification Scripts
- `verify_shap.py`: Tests basic SHAP functionality with XGBoost models.
- `test_numpy_api.py`: Checks for NumPy API compatibility issues.
- `find_deprecated_apis.py`: Searches for deprecated API usage in the SHAP library.

### 2. Patching Scripts
- `patch_shap_library.py`: Comprehensive script that applies all necessary patches.
- `direct_shap_patch.py`: Directly modifies specific issues in the SHAP library.
- `patch_shap_imports.py`: Focuses on fixing import-related issues.

### 3. IDE Configuration
- `disable_shap_typecheck.py`: Generates `pyrightconfig.json` to exclude SHAP from type checking.
- `pyrightconfig.json`: Configuration that excludes the SHAP library from type checking to prevent false positives.

## Future Maintenance

When upgrading SHAP or related dependencies:

1. Run the verification scripts to check if the patches are still needed.
2. Check for any new deprecation warnings related to NumPy APIs.
3. Update the patching scripts if necessary.

## Best Practices

- Always test the SHAP library with actual models after applying patches.
- Be aware that these are workarounds and may need to be updated as SHAP evolves.
- Consider contributing these fixes upstream to the SHAP project.

## Conclusion

These patches enable the SHAP library to work with Python 3.13 and newer NumPy versions by addressing import error handling and deprecated API usage. The verification tools ensure the patches are working correctly and can be used for future maintenance.
