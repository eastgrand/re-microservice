#!/usr/bin/env python3
"""
Test script to verify comprehensive field mapping is working correctly.
This tests the field_aliases dictionary and mapping logic.
"""

import requests
import json
import time

# Test different field types that should now be supported
TEST_CASES = [
    {
        "name": "Nike Athletic Shoes",
        "target_variable": "MP30034A_B",
        "expected_mapping": "value_MP30034A_B"
    },
    {
        "name": "Adidas Athletic Shoes", 
        "target_variable": "MP30029A_B",
        "expected_mapping": "value_MP30029A_B"
    },
    {
        "name": "Total Population",
        "target_variable": "TOTPOP_CY",
        "expected_mapping": "value_TOTPOP_CY"
    },
    {
        "name": "Median Income",
        "target_variable": "MEDDI_CY", 
        "expected_mapping": "value_MEDDI_CY"
    },
    {
        "name": "Diversity Index",
        "target_variable": "DIVINDX_CY",
        "expected_mapping": "value_DIVINDX_CY"
    },
    {
        "name": "White Population",
        "target_variable": "WHITE_CY",
        "expected_mapping": "value_WHITE_CY"
    },
    {
        "name": "Generation Z",
        "target_variable": "GENZ_CY",
        "expected_mapping": "value_GENZ_CY"
    },
    {
        "name": "NFL Super Fan",
        "target_variable": "MP33107A_B",
        "expected_mapping": "value_MP33107A_B"
    },
    {
        "name": "Weight Lifting Participation",
        "target_variable": "MP33031A_B", 
        "expected_mapping": "value_MP33031A_B"
    },
    {
        "name": "Dick's Sporting Goods Shopping",
        "target_variable": "MP31035A_B",
        "expected_mapping": "value_MP31035A_B"
    }
]

def test_field_mapping():
    """Test that various field types are properly mapped."""
    
    base_url = "https://shap-demographic-analytics-v2.onrender.com"
    # base_url = "http://localhost:8000"  # Uncomment for local testing
    
    print("🧪 Testing Comprehensive Field Mapping")
    print("=" * 50)
    
    success_count = 0
    total_count = len(TEST_CASES)
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n{i}. Testing {test_case['name']} ({test_case['target_variable']})")
        
        # Create test request
        test_request = {
            "query": f"Analyze {test_case['name'].lower()} distribution",
            "analysis_type": "ranking",
            "target_variable": test_case['target_variable'],
            "matched_fields": [test_case['target_variable']],
            "relevant_layers": ["demographic_layer"],
            "top_n": 10
        }
        
        try:
            # Send request to microservice
            response = requests.post(
                f"{base_url}/analyze",
                json=test_request,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success"):
                    print(f"   ✅ SUCCESS: Field mapping worked")
                    print(f"   📊 Found {len(result.get('results', []))} results")
                    
                    # Check if the target variable was properly mapped
                    model_info = result.get("model_info", {})
                    actual_target = model_info.get("target_variable", "")
                    
                    if test_case['expected_mapping'] in actual_target or test_case['target_variable'] in actual_target:
                        print(f"   🎯 Target mapping: {actual_target}")
                        success_count += 1
                    else:
                        print(f"   ⚠️  Unexpected target mapping: {actual_target}")
                        print(f"   Expected: {test_case['expected_mapping']}")
                        
                else:
                    print(f"   ❌ FAILED: {result.get('error', 'Unknown error')}")
                    
            else:
                print(f"   ❌ HTTP ERROR: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ TIMEOUT: Request took longer than 30 seconds")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ REQUEST ERROR: {e}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            
        # Small delay between requests
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print(f"📈 SUMMARY: {success_count}/{total_count} field mappings successful")
    
    if success_count == total_count:
        print("🎉 All field mappings are working correctly!")
        return True
    else:
        print(f"⚠️  {total_count - success_count} field mappings need attention")
        return False

if __name__ == "__main__":
    success = test_field_mapping()
    exit(0 if success else 1) 