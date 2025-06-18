#!/usr/bin/env python3
"""
Quick test script to check FeatureServer accessibility and layers
"""
import requests
import json

def test_featureserver():
    url = 'https://services8.arcgis.com/VhrZdFGa39zmfR47/arcgis/rest/services/Synapse54_Vetements_layers/FeatureServer'
    
    print(f"🔍 Testing FeatureServer: {url}")
    
    try:
        response = requests.get(f'{url}?f=json', timeout=30)
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ Service accessible!")
            print(f"📋 Service Name: {data.get('serviceDescription', 'N/A')}")
            print(f"🗂️ Layers found: {len(data.get('layers', []))}")
            
            if 'layers' in data:
                for layer in data['layers']:
                    print(f"  📄 Layer {layer['id']}: {layer['name']} ({layer.get('type', 'Feature Layer')})")
            
            if 'tables' in data:
                for table in data['tables']:
                    print(f"  📊 Table {table['id']}: {table['name']}")
            
            # Test first layer access
            if data.get('layers'):
                first_layer = data['layers'][0]
                layer_url = f"{url}/{first_layer['id']}"
                print(f"\n🔍 Testing first layer access: {layer_url}")
                
                layer_response = requests.get(f'{layer_url}?f=json', timeout=30)
                if layer_response.status_code == 200:
                    layer_data = layer_response.json()
                    fields = layer_data.get('fields', [])
                    print(f"✅ Layer accessible with {len(fields)} fields")
                    
                    # Show first few fields
                    for field in fields[:5]:
                        print(f"    🔑 {field['name']} ({field['type']})")
                    
                    if len(fields) > 5:
                        print(f"    ... and {len(fields) - 5} more fields")
                else:
                    print(f"❌ Layer not accessible: {layer_response.status_code}")
            
        else:
            print(f"❌ Service not accessible")
            print(f"Response: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_featureserver() 