#!/usr/bin/env python3
"""
Debug script to focus specifically on model and feature loading issues
"""

import os
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_model_detailed():
    """Debug model loading in detail"""
    logger.info("=== Detailed Model Debugging ===")
    
    model_path = "models/xgboost_model.pkl"
    
    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"📊 Model type: {type(model)}")
            
            # Try different approaches to inspect the model
            try:
                logger.info(f"🔧 Model class: {model.__class__}")
                logger.info(f"🔧 Model dict keys: {list(model.__dict__.keys()) if hasattr(model, '__dict__') else 'No __dict__'}")
            except Exception as e:
                logger.warning(f"⚠️ Could not access model dict: {e}")
            
            # Try to access model features without triggering the sklearn issue
            try:
                if hasattr(model, 'n_features_in_'):
                    n_features = model.n_features_in_
                    logger.info(f"🎯 Model expects {n_features} features")
                else:
                    logger.warning("⚠️ Model doesn't have n_features_in_ attribute")
            except Exception as e:
                logger.warning(f"⚠️ Could not get feature count: {e}")
            
            # Try to access booster
            try:
                if hasattr(model, 'get_booster'):
                    booster = model.get_booster()
                    logger.info(f"🔧 Booster accessed: {type(booster)}")
                    
                    # Get feature names from booster
                    if hasattr(booster, 'feature_names'):
                        feature_names = booster.feature_names
                        logger.info(f"📋 Booster feature names: {len(feature_names) if feature_names else 0}")
                        if feature_names:
                            logger.info(f"📋 First 5 features: {feature_names[:5]}")
                    else:
                        logger.warning("⚠️ Booster doesn't have feature_names")
                        
            except Exception as e:
                logger.warning(f"⚠️ Could not access booster: {e}")
            
            return model
            
        else:
            logger.error(f"❌ Model file not found: {model_path}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def debug_feature_names_detailed():
    """Debug feature names loading in detail"""
    logger.info("=== Detailed Feature Names Debugging ===")
    
    feature_names_path = "models/feature_names.txt"
    
    try:
        if os.path.exists(feature_names_path):
            logger.info(f"📁 Feature names file exists")
            
            # Check file size
            file_size = os.path.getsize(feature_names_path)
            logger.info(f"📏 File size: {file_size} bytes")
            
            # Read first few lines
            with open(feature_names_path, 'r') as f:
                first_lines = []
                for i, line in enumerate(f):
                    if i < 10:
                        first_lines.append(line.strip())
                    else:
                        break
                        
            logger.info(f"📋 First 10 lines: {first_lines}")
            
            # Read all lines
            with open(feature_names_path, 'r') as f:
                all_lines = [line.strip() for line in f.readlines()]
                
            logger.info(f"✅ Total features: {len(all_lines)}")
            logger.info(f"📋 Last 5 features: {all_lines[-5:]}")
            
            # Check for empty lines
            empty_lines = [i for i, line in enumerate(all_lines) if not line.strip()]
            if empty_lines:
                logger.warning(f"⚠️ Empty lines found at positions: {empty_lines}")
            
            return all_lines
            
        else:
            logger.error(f"❌ Feature names file not found: {feature_names_path}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error loading feature names: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def check_compatibility():
    """Check version compatibility"""
    logger.info("=== Version Compatibility Check ===")
    
    try:
        import xgboost as xgb
        import sklearn
        
        logger.info(f"📦 XGBoost version: {xgb.__version__}")
        logger.info(f"📦 Scikit-learn version: {sklearn.__version__}")
        
        # Check if XGBoost has the problematic attribute
        from xgboost.sklearn import XGBRegressor
        dummy_model = XGBRegressor()
        
        logger.info(f"🔧 XGBRegressor attributes: {dir(dummy_model)}")
        
        if hasattr(dummy_model, 'gpu_id'):
            logger.info("✅ gpu_id attribute exists")
        else:
            logger.warning("⚠️ gpu_id attribute missing - this might be the compatibility issue")
            
    except Exception as e:
        logger.error(f"❌ Version compatibility check failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Run focused debugging"""
    logger.info("🚀 Starting Focused Model Debug")
    logger.info("=" * 50)
    
    # Check versions first
    check_compatibility()
    logger.info("")
    
    # Debug model loading
    model = debug_model_detailed()
    logger.info("")
    
    # Debug feature names loading
    feature_names = debug_feature_names_detailed()
    logger.info("")
    
    # Summary
    if model is not None and feature_names is not None:
        logger.info("🎉 Both model and feature names loaded successfully!")
        logger.info("🔍 The issue is likely in the SHAP initialization or feature alignment")
    elif model is not None:
        logger.info("⚠️ Model loaded but feature names failed")
        logger.info("🔍 The issue is in feature names loading")
    elif feature_names is not None:
        logger.info("⚠️ Feature names loaded but model failed")
        logger.info("🔍 The issue is in model loading - likely XGBoost version incompatibility")
    else:
        logger.error("❌ Both model and feature names failed to load")
    
    logger.info("=" * 50)

if __name__ == "__main__":
    main()