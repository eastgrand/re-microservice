#!/usr/bin/env python3
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
