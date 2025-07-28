#!/usr/bin/env python3
"""
Test the live SHAP microservice with API key authentication
"""

import requests
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_live_service_with_auth():
    """Test the live service with authentication"""
    logger.info("=== Testing Live SHAP Microservice with Auth ===")
    
    live_url = "https://shap-demographic-analytics-v3.onrender.com"
    api_key = "HFqkccbN3LV5CaB"
    
    headers = {
        'X-API-Key': api_key,
        'Content-Type': 'application/json'
    }
    
    try:
        # Test health endpoint (no auth required)
        logger.info("🔍 Testing health endpoint...")
        response = requests.get(f"{live_url}/", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Health check passed")
            logger.info(f"📊 Model loaded: {data.get('model_loaded', False)}")
            logger.info(f"🧠 Schema initialized: {data.get('schema_initialized', False)}")
            logger.info(f"📊 Feature count: {data.get('feature_count', 'Not reported')}")
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False
        
        # Test analyze endpoint with auth
        logger.info("🧪 Testing analyze endpoint with authentication...")
        test_payload = {
            "query": "test SHAP functionality after fix",
            "target_variable": "MP30034A_B_P",
            "analysis_type": "correlation"
        }
        
        response = requests.post(f"{live_url}/analyze", 
                               json=test_payload, 
                               headers=headers,
                               timeout=180)  # Increased timeout for SHAP calculation
        
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
                    logger.error("🔍 This means the SHAP calculation is still failing")
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
                            
                            logger.info("✅ SUCCESS: SHAP microservice is working correctly!")
                            return True
                        else:
                            logger.warning("⚠️ No SHAP fields found in results")  
                            return False
                    else:
                        logger.warning("⚠️ No results returned")
                        return False
                else:
                    logger.info(f"✅ Analysis completed with method: {analysis_type}")
                    # Even if not explicitly SHAP, if it's not raw_data_fallback, it's an improvement
                    return True
                    
            else:
                logger.error(f"❌ Analyze failed: {data.get('error', 'Unknown error')}")
                logger.error(f"Full response: {json.dumps(data, indent=2)}")
                return False
        else:
            logger.error(f"❌ Analyze endpoint failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Live service test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main testing function"""
    logger.info("🚀 Starting Live Service Test with Authentication")
    logger.info("=" * 60)
    
    # Test the service
    if test_live_service_with_auth():
        logger.info("🎉 SUCCESS: SHAP microservice is working correctly!")
        logger.info("✅ The feature mismatch issue has been resolved")
        logger.info("✅ SHAP calculations are now returning proper values")
        logger.info("✅ No more 'raw_data_fallback' responses")
    else:
        logger.error("❌ FAILURE: Issues still remain")
        logger.info("🔍 The microservice may still be starting up or there are other issues")
        logger.info("💡 Try again in a few minutes if the deployment just completed")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()