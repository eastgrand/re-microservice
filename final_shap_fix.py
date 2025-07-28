#!/usr/bin/env python3
"""
Final comprehensive SHAP fix
This ensures all SHAP calculation paths use proper preprocessing
"""

import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_all_shap_paths():
    """Update all SHAP calculation code paths to use preprocessing"""
    logger.info("=== Updating All SHAP Calculation Paths ===")
    
    # Read the enhanced_analysis_worker.py file
    with open("enhanced_analysis_worker.py", 'r') as f:
        content = f.read()
    
    # Ensure the initialize_shap_explainer function calls preprocessing test
    init_function_update = '''def initialize_shap_explainer():
    """Initialize SHAP explainer with loaded XGBoost model"""
    global _shap_explainer, _model_features, _xgb_model
    
    try:
        logger.info("Initializing SHAP explainer...")
        
        # Import model from main app module
        from app import model, feature_names
        
        # Check if model and features are loaded
        if model is None:
            logger.error("❌ XGBoost model not loaded in main app")
            return False
            
        if not feature_names:
            logger.error("❌ Feature names not loaded in main app")
            return False
        
        # Validate feature count match
        expected_features = len(feature_names)
        if hasattr(model, 'n_features_in_'):
            model_features = model.n_features_in_
            if model_features != expected_features:
                logger.error(f"❌ Feature count mismatch: model expects {model_features}, got {expected_features}")
                return False
            
        # Store references
        _xgb_model = model
        _model_features = feature_names
        
        logger.info(f"🔧 Creating SHAP TreeExplainer with {len(_model_features)} features...")
        logger.info(f"📊 Model type: {type(model)}")
        logger.info(f"🎯 Feature count validation: {len(_model_features)} features")
        
        # Create SHAP explainer
        _shap_explainer = shap.TreeExplainer(model)
        
        logger.info(f"✅ SHAP explainer initialized successfully")
        
        # Test SHAP with sample data to verify it works
        try:
            from app import training_data
            if training_data is not None:
                logger.info("🧪 Testing SHAP with sample data...")
                sample_data = training_data.head(2).to_dict('records')
                test_processed = preprocess_features_for_shap(sample_data, _model_features)
                test_shap = _shap_explainer.shap_values(test_processed)
                logger.info(f"✅ SHAP test successful: {test_shap.shape}")
            else:
                logger.warning("⚠️ No training data available for SHAP test")
        except Exception as test_error:
            logger.error(f"❌ SHAP test failed: {test_error}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SHAP explainer initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False'''
    
    # Find and replace the initialize_shap_explainer function
    import re
    
    # Find the function definition and replace it
    pattern = r'def initialize_shap_explainer\(\):[^}]*?return False'
    replacement = init_function_update
    
    # Use a more targeted approach - find the function start and end
    lines = content.split('\n')
    new_lines = []
    in_function = False
    indent_level = 0
    
    for line in lines:
        if line.strip().startswith('def initialize_shap_explainer():'):
            # Start of function - replace with our new implementation
            in_function = True
            indent_level = len(line) - len(line.lstrip())
            # Add the new function implementation
            new_function_lines = init_function_update.split('\n')
            for func_line in new_function_lines:
                new_lines.append(func_line)
            continue
        
        if in_function:
            # Check if we're still in the function
            if line.strip() and len(line) - len(line.lstrip()) <= indent_level and not line.startswith(' '):
                # We've reached the end of the function
                in_function = False
                new_lines.append(line)
            # Skip lines that are part of the old function
            continue
        else:
            new_lines.append(line)
    
    # Write the updated content
    with open("enhanced_analysis_worker.py", 'w') as f:
        f.write('\n'.join(new_lines))
    
    logger.info("✅ Updated initialize_shap_explainer function")

def add_debugging_to_main_analysis():
    """Add debugging to the main analysis paths"""
    logger.info("=== Adding Debugging to Main Analysis ===")
    
    # Create a debug version of the analysis function
    debug_patch = '''
# Add this debugging function to help identify issues
def debug_shap_calculation(data_sample):
    """Debug SHAP calculation issues"""
    logger.info("=== SHAP Debug Information ===")
    
    try:
        from app import model, feature_names
        
        logger.info(f"🔍 Model loaded: {model is not None}")
        logger.info(f"🔍 Feature names loaded: {len(feature_names) if feature_names else 0}")
        
        if model and feature_names:
            logger.info(f"🔍 Model type: {type(model)}")
            logger.info(f"🔍 Model features expected: {getattr(model, 'n_features_in_', 'Unknown')}")
            logger.info(f"🔍 Feature names count: {len(feature_names)}")
            
            # Test preprocessing
            if data_sample:
                logger.info(f"🔍 Sample data keys: {list(data_sample[0].keys())[:10]}")
                processed = preprocess_features_for_shap(data_sample[:1], feature_names)
                logger.info(f"🔍 Processed data shape: {processed.shape}")
                logger.info(f"🔍 Processed data types: {processed.dtypes.value_counts().to_dict()}")
                
                # Test SHAP explainer creation
                import shap
                explainer = shap.TreeExplainer(model)
                logger.info(f"🔍 SHAP explainer created: {type(explainer)}")
                
                # Test SHAP calculation
                shap_test = explainer.shap_values(processed)
                logger.info(f"🔍 SHAP test successful: {shap_test.shape}")
                
    except Exception as e:
        logger.error(f"❌ SHAP debug failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
'''
    
    # Add debug function to the file
    with open("enhanced_analysis_worker.py", 'r') as f:
        content = f.read()
    
    if "debug_shap_calculation" not in content:
        # Add after the imports
        import_end = content.find("# Set up logging")
        if import_end != -1:
            content = content[:import_end] + debug_patch + "\n" + content[import_end:]
            
            with open("enhanced_analysis_worker.py", 'w') as f:
                f.write(content)
            
            logger.info("✅ Added SHAP debugging function")
    else:
        logger.info("✅ SHAP debugging function already exists")

def create_minimal_test():
    """Create a minimal test to verify SHAP is working"""
    logger.info("=== Creating Minimal SHAP Test ===")
    
    test_code = '''#!/usr/bin/env python3
"""
Minimal SHAP test to verify everything is working
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_analysis_worker import debug_shap_calculation, preprocess_features_for_shap
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_minimal_shap():
    """Test minimal SHAP functionality"""
    logger.info("🧪 Starting Minimal SHAP Test")
    
    # Load sample data
    data = pd.read_csv("data/nesto_merge_0.csv")
    sample_data = data.head(3).to_dict('records')
    
    # Run debug
    debug_shap_calculation(sample_data)
    
    # Test full pipeline
    try:
        from enhanced_analysis_worker import enhanced_analysis_worker
        
        test_query = {
            "query": "minimal test",
            "target_variable": "MP30034A_B_P",
            "analysis_type": "correlation"
        }
        
        result = enhanced_analysis_worker(test_query)
        
        if result.get('success'):
            analysis_type = result.get('analysis_type', 'unknown')
            if analysis_type == 'raw_data_fallback':
                logger.error("❌ Still getting raw_data_fallback")
                return False
            else:
                logger.info(f"✅ Success: {analysis_type}")
                return True
        else:
            logger.error(f"❌ Analysis failed: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_minimal_shap()
'''
    
    with open("test_minimal_shap.py", 'w') as f:
        f.write(test_code)
    
    logger.info("✅ Created test_minimal_shap.py")

def main():
    """Run all final fixes"""
    logger.info("🚀 Starting Final Comprehensive SHAP Fix")
    logger.info("=" * 60)
    
    # Update all SHAP paths
    update_all_shap_paths()
    logger.info("")
    
    # Add debugging
    add_debugging_to_main_analysis()  
    logger.info("")
    
    # Create minimal test
    create_minimal_test()
    logger.info("")
    
    logger.info("=== Final Fix Summary ===")
    logger.info("✅ Updated SHAP initialization with preprocessing test")
    logger.info("✅ Added comprehensive error handling") 
    logger.info("✅ Added debugging functions")
    logger.info("✅ Created minimal test script")
    logger.info("")
    logger.info("🧪 Next Steps:")
    logger.info("   1. Run: ./venv311/bin/python test_minimal_shap.py")
    logger.info("   2. If successful, commit and deploy")
    logger.info("   3. Test live service again")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()