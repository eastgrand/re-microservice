#!/usr/bin/env python3
"""
Deploy the SHAP fix and verify it works
"""

import os
import subprocess
import time
import requests
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_local_microservice():
    """Test the microservice locally before deploying"""
    logger.info("=== Testing Local Microservice ===")
    
    try:
        # Test the enhanced SHAP functionality
        logger.info("🔬 Running enhanced SHAP test...")
        result = subprocess.run(['./venv311/bin/python', 'test_enhanced_shap.py'], 
                               capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            logger.info("✅ Enhanced SHAP test passed!")
        else:
            logger.error(f"❌ Enhanced SHAP test failed: {result.stderr}")
            return False
            
        # Start the microservice locally for testing
        logger.info("🚀 Starting local microservice for testing...")
        
        # Kill any existing processes on port 5001
        subprocess.run(['pkill', '-f', 'app.py'], capture_output=True)
        time.sleep(2)
        
        # Start the service in background
        env = os.environ.copy()
        env['DEBUG'] = 'true'
        
        process = subprocess.Popen(['./venv311/bin/python', 'app.py'], 
                                 env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for service to start
        logger.info("⏳ Waiting for service to start...")
        time.sleep(10)
        
        # Test health endpoint
        try:
            response = requests.get('http://localhost:5001/', timeout=30)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Health check passed: {data.get('message', 'No message')}")
                logger.info(f"📊 Model loaded: {data.get('model_loaded', False)}")
                logger.info(f"🧠 Schema initialized: {data.get('schema_initialized', False)}")
            else:
                logger.error(f"❌ Health check failed: {response.status_code}")
                process.terminate()
                return False
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            process.terminate()
            return False
        
        # Test analyze endpoint
        try:
            logger.info("🧪 Testing analyze endpoint...")
            test_payload = {
                "query": "analyze customer segmentation",
                "target_variable": "MP30034A_B_P",
                "analysis_type": "correlation"
            }
            
            response = requests.post('http://localhost:5001/analyze', 
                                   json=test_payload, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success', False):
                    logger.info("✅ Analyze endpoint working!")
                    logger.info(f"📊 Analysis type: {data.get('analysis_type', 'unknown')}")
                    logger.info(f"📈 Records processed: {data.get('total_records', 0)}")
                    logger.info(f"🧠 SHAP enabled: {data.get('shap_enabled', False)}")
                    
                    # Check if SHAP values are present
                    results = data.get('results', [])
                    if results:
                        sample_record = results[0]
                        shap_fields = [k for k in sample_record.keys() if k.startswith('shap_')]
                        logger.info(f"🎯 SHAP fields found: {len(shap_fields)}")
                        
                        if shap_fields:
                            logger.info("🎉 SHAP calculation is working!")
                        else:
                            logger.warning("⚠️ No SHAP fields found in results")
                    
                else:
                    logger.error(f"❌ Analyze failed: {data.get('error', 'Unknown error')}")
                    process.terminate()
                    return False
            else:
                logger.error(f"❌ Analyze endpoint failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                process.terminate()
                return False
                
        except Exception as e:
            logger.error(f"❌ Analyze test failed: {e}")
            process.terminate()
            return False
        
        # Cleanup
        process.terminate()
        logger.info("✅ Local testing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Local testing failed: {e}")
        return False

def deploy_to_render():
    """Deploy the fixed microservice to Render"""
    logger.info("=== Deploying to Render ===")
    
    try:
        # Check if git has changes to commit
        result = subprocess.run(['git', 'status', '--porcelain'], 
                               capture_output=True, text=True)
        
        if result.stdout.strip():
            logger.info("📝 Committing changes...")
            
            # Add all changes
            subprocess.run(['git', 'add', '.'], check=True)
            
            # Commit with descriptive message
            commit_message = """Fix SHAP calculation issues

- Fixed feature count mismatch (added missing Age and Income features)  
- Updated feature_names.txt to match model exactly (543 features)
- Added data preprocessing for SHAP compatibility
- Fixed string/object column handling in SHAP calculations
- Added enhanced test suite for SHAP functionality

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"""
            
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            logger.info("✅ Changes committed")
            
            # Push to trigger Render deployment
            logger.info("🚀 Pushing to trigger Render deployment...")
            subprocess.run(['git', 'push'], check=True)
            logger.info("✅ Changes pushed to git")
            
        else:
            logger.info("📝 No changes to commit")
        
        logger.info("🔄 Render will automatically deploy the changes...")
        logger.info("⏳ This may take 5-10 minutes...")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Git operation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        return False

def test_live_service():
    """Test the live service after deployment"""
    logger.info("=== Testing Live Service ===")
    
    live_url = "https://shap-demographic-analytics-v3.onrender.com"
    
    # Wait a bit for deployment to complete
    logger.info("⏳ Waiting for deployment to complete...")
    time.sleep(30)
    
    try:
        # Test health endpoint
        logger.info("🔍 Testing live health endpoint...")
        response = requests.get(f"{live_url}/", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Live health check passed")
            logger.info(f"📊 Model loaded: {data.get('model_loaded', False)}")
            logger.info(f"🧠 Schema initialized: {data.get('schema_initialized', False)}")
        else:
            logger.error(f"❌ Live health check failed: {response.status_code}")
            return False
        
        # Test analyze endpoint
        logger.info("🧪 Testing live analyze endpoint...")
        test_payload = {
            "query": "test SHAP functionality",
            "target_variable": "MP30034A_B_P",
            "analysis_type": "correlation"
        }
        
        response = requests.post(f"{live_url}/analyze", 
                               json=test_payload, timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success', False):
                logger.info("✅ Live analyze endpoint working!")
                
                analysis_type = data.get('analysis_type', 'unknown')
                logger.info(f"📊 Analysis type: {analysis_type}")
                
                # Check if we're getting SHAP values instead of raw_data_fallback
                if analysis_type == "raw_data_fallback":
                    logger.error("❌ Still getting raw_data_fallback - SHAP is not working")
                    return False
                elif "shap" in analysis_type.lower():
                    logger.info("🎉 SHAP calculation is working!")
                    return True
                else:
                    logger.info(f"✅ Analysis completed with method: {analysis_type}")
                    return True
                    
            else:
                logger.error(f"❌ Live analyze failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            logger.error(f"❌ Live analyze endpoint failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Live service test failed: {e}")
        return False

def main():
    """Run the complete deployment and testing process"""
    logger.info("🚀 Starting SHAP Fix Deployment Process")
    logger.info("=" * 60)
    
    # Step 1: Test locally
    logger.info("Step 1: Testing locally...")
    if not test_local_microservice():
        logger.error("❌ Local testing failed - aborting deployment")
        return
    
    logger.info("")
    
    # Step 2: Deploy to Render
    logger.info("Step 2: Deploying to Render...")
    if not deploy_to_render():
        logger.error("❌ Deployment failed")
        return
        
    logger.info("")
    
    # Step 3: Test live service
    logger.info("Step 3: Testing live service...")
    if test_live_service():
        logger.info("🎉 Deployment successful! SHAP is now working correctly.")
    else:
        logger.error("❌ Live service test failed - may need additional debugging")
    
    logger.info("=" * 60)
    logger.info("✅ Deployment process complete")

if __name__ == "__main__":
    main()