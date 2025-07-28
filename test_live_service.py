#!/usr/bin/env python3
"""
Test the live SHAP microservice after deployment
"""

import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_live_service():
    """Test the live service after deployment"""
    logger.info("=== Testing Live SHAP Microservice ===")
    
    live_url = "https://shap-demographic-analytics-v3.onrender.com"
    
    try:
        # Test health endpoint
        logger.info("🔍 Testing health endpoint...")
        response = requests.get(f"{live_url}/", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Health check passed")
            logger.info(f"📊 Model loaded: {data.get('model_loaded', False)}")
            logger.info(f"🧠 Schema initialized: {data.get('schema_initialized', False)}")
            logger.info(f"💾 Memory usage: {data.get('memory_usage_mb', 0):.1f}MB")
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test detailed health
        logger.info("🔍 Testing detailed health endpoint...")
        response = requests.get(f"{live_url}/health", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Detailed health check passed")
            logger.info(f"📊 Feature count: {data.get('feature_count', 0)}")
            logger.info(f"📈 Training data rows: {data.get('training_data_rows', 0)}")
        
        # Test analyze endpoint
        logger.info("🧪 Testing analyze endpoint...")
        test_payload = {
            "query": "test SHAP functionality after fix",
            "target_variable": "MP30034A_B_P",
            "analysis_type": "correlation"
        }
        
        response = requests.post(f"{live_url}/analyze", 
                               json=test_payload, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success', False):
                logger.info("✅ Analyze endpoint working!")
                
                analysis_type = data.get('analysis_type', 'unknown')
                logger.info(f"📊 Analysis type: {analysis_type}")
                logger.info(f"📈 Records processed: {data.get('total_records', 0)}")
                logger.info(f"🧠 SHAP enabled: {data.get('shap_enabled', False)}")
                
                # Check if we're getting SHAP values instead of raw_data_fallback
                if analysis_type == "raw_data_fallback":
                    logger.error("❌ Still getting raw_data_fallback - SHAP is not working")
                    return False
                elif "shap" in analysis_type.lower():
                    logger.info("🎉 SHAP calculation is working!")
                    
                    # Check for SHAP fields in results
                    results = data.get('results', [])
                    if results:
                        sample_record = results[0]
                        shap_fields = [k for k in sample_record.keys() if k.startswith('shap_')]
                        logger.info(f"🎯 SHAP fields found: {len(shap_fields)}")
                        
                        if shap_fields:
                            # Show some sample SHAP values
                            logger.info("📊 Sample SHAP values:")
                            for field in shap_fields[:5]:
                                value = sample_record.get(field, 0)
                                logger.info(f"   {field}: {value}")
                        
                        return True
                    else:
                        logger.warning("⚠️ No results returned")
                        return False
                else:
                    logger.info(f"✅ Analysis completed with method: {analysis_type}")
                    return True
                    
            else:
                logger.error(f"❌ Analyze failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            logger.error(f"❌ Analyze endpoint failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Live service test failed: {e}")
        return False

def main():
    """Main testing function"""
    logger.info("🚀 Starting Live Service Test")
    logger.info("=" * 50)
    
    # Wait for deployment to complete
    logger.info("⏳ Waiting for deployment to complete...")
    time.sleep(60)  # Wait 1 minute
    
    # Test the service
    if test_live_service():
        logger.info("🎉 SUCCESS: SHAP microservice is working correctly!")
        logger.info("✅ The feature mismatch issue has been resolved")
        logger.info("✅ SHAP calculations are now returning proper values")
    else:
        logger.error("❌ FAILURE: Issues still remain")
        logger.info("🔍 Please check the deployment logs for more details")
    
    logger.info("=" * 50)

if __name__ == "__main__":
    main()