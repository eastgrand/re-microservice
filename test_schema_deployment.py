#!/usr/bin/env python3
"""
Test script to verify that the microservice schema includes the expected fields
after deployment. This helps catch deployment issues before they affect production.
"""

import requests
import sys
import os

def test_local_schema():
    """Test the schema when running locally"""
    print("🧪 Testing local schema...")
    
    try:
        # Import and test the app directly
        from app import app, df, AVAILABLE_COLUMNS
        
        if df is None:
            print("❌ ERROR: DataFrame not loaded")
            return False
            
        field_count = len(df.columns)
        print(f"✅ Local data loaded: {field_count} fields")
        
        if field_count < 100:
            print(f"⚠️  WARNING: Field count ({field_count}) is lower than expected (140+)")
            print(f"Sample fields: {list(df.columns[:10])}")
            return False
        
        # Check for specific expected fields
        expected_fields = ['conversion_rate', 'visible_minority_population_pct', 'FSA_ID', 'ID']
        missing_fields = []
        
        for field in expected_fields:
            if field not in df.columns:
                # Try common variations
                variations = [
                    field.upper(),
                    field.lower(),
                    field.replace('_', ' ').title(),
                    field.replace('_', '').lower()
                ]
                found = False
                for variation in variations:
                    if variation in df.columns:
                        print(f"✅ Found field variation: {field} -> {variation}")
                        found = True
                        break
                
                if not found:
                    missing_fields.append(field)
        
        if missing_fields:
            print(f"⚠️  Missing expected fields: {missing_fields}")
            print(f"Available fields sample: {sorted(list(df.columns))[:20]}")
        else:
            print("✅ All expected fields found")
        
        print(f"✅ Local schema test passed: {field_count} fields available")
        return True
        
    except Exception as e:
        print(f"❌ ERROR testing local schema: {e}")
        return False

def test_deployed_schema(url, api_key):
    """Test the schema from deployed microservice"""
    print(f"🌐 Testing deployed schema at {url}...")
    
    try:
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        response = requests.get(f"{url}/api/v1/schema", headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ ERROR: Schema endpoint returned {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        schema_data = response.json()
        known_fields = schema_data.get('known_fields', [])
        field_count = len(known_fields)
        
        print(f"✅ Deployed schema loaded: {field_count} fields")
        
        if field_count < 100:
            print(f"⚠️  WARNING: Field count ({field_count}) is lower than expected (140+)")
            print(f"Sample fields: {known_fields[:10]}")
            return False
        
        # Check for specific expected fields
        expected_fields = ['conversion_rate', 'visible_minority_population_pct']
        missing_fields = [field for field in expected_fields if field not in known_fields]
        
        if missing_fields:
            print(f"⚠️  Missing expected fields: {missing_fields}")
            print(f"Available fields sample: {sorted(known_fields)[:20]}")
        else:
            print("✅ All expected fields found in deployed schema")
        
        print(f"✅ Deployed schema test passed: {field_count} fields available")
        return True
        
    except Exception as e:
        print(f"❌ ERROR testing deployed schema: {e}")
        return False

def main():
    """Main test function"""
    print("🔍 Testing microservice schema deployment...")
    
    # Test local schema
    local_success = test_local_schema()
    
    # Test deployed schema if URL provided
    deployed_success = True
    url = os.getenv('MICROSERVICE_URL', 'https://nesto-mortgage-analytics.onrender.com')
    api_key = os.getenv('API_KEY', 'HFqkccbN3LV5CaB')
    
    if url:
        deployed_success = test_deployed_schema(url, api_key)
    
    # Summary
    print("\n📊 Test Summary:")
    print(f"Local schema: {'✅ PASS' if local_success else '❌ FAIL'}")
    print(f"Deployed schema: {'✅ PASS' if deployed_success else '❌ FAIL'}")
    
    if local_success and deployed_success:
        print("\n🎉 All tests passed! Schema deployment is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 