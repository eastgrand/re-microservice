#!/usr/bin/env python3
"""
Check fields in layers that have data to find the correct ID field
"""
import requests
import json

def check_layer_fields():
    base_url = 'https://services8.arcgis.com/VhrZdFGa39zmfR47/arcgis/rest/services/Synapse54_Vetements_layers/FeatureServer'
    
    # Check layers that have data (skip the empty ones)
    layers_with_data = [3, 7, 8, 9, 10]  # Based on the extraction log
    
    for layer_id in layers_with_data:
        print(f"\n🔍 Checking Layer {layer_id}:")
        
        try:
            # Get layer info
            response = requests.get(f'{base_url}/{layer_id}?f=json', timeout=30)
            if response.status_code == 200:
                layer_data = response.json()
                
                print(f"  📋 Name: {layer_data.get('name', 'N/A')}")
                print(f"  📊 Fields:")
                
                fields = layer_data.get('fields', [])
                for field in fields:
                    field_type = field['type']
                    field_name = field['name']
                    alias = field.get('alias', field_name)
                    
                    # Highlight potential ID fields
                    if any(id_term in field_name.upper() for id_term in ['ID', 'ZIP', 'FIPS', 'GEOID', 'CODE']):
                        print(f"    🔑 {field_name} ({field_type}) - ALIAS: {alias} ⭐ POTENTIAL ID")
                    else:
                        print(f"    📄 {field_name} ({field_type}) - ALIAS: {alias}")
                
                # Test a sample query to see what data looks like
                print(f"  🧪 Sample data:")
                query_response = requests.get(f'{base_url}/{layer_id}/query?where=1=1&outFields=*&resultRecordCount=3&f=json', timeout=30)
                if query_response.status_code == 200:
                    query_data = query_response.json()
                    features = query_data.get('features', [])
                    if features:
                        sample_record = features[0]['attributes']
                        for key, value in list(sample_record.items())[:5]:
                            print(f"    📝 {key}: {value}")
                        if len(sample_record) > 5:
                            print(f"    ... and {len(sample_record) - 5} more fields")
                    else:
                        print("    ❌ No sample data available")
                
            else:
                print(f"  ❌ Failed to access layer {layer_id}: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error checking layer {layer_id}: {e}")

if __name__ == "__main__":
    check_layer_fields() 