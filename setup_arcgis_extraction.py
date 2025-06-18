#!/usr/bin/env python3
"""
Setup script for ArcGIS data extraction

This script helps you configure the data extraction from your ArcGIS feature services.
It can discover available layers and create the configuration file automatically.
"""

import os
import json
import requests
import logging
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("arcgis-setup")

def discover_service_layers(base_url: str) -> List[Dict[str, Any]]:
    """Discover all available layers in an ArcGIS service"""
    logger.info(f"🔍 Discovering layers in service: {base_url}")
    
    try:
        # Get service metadata
        response = requests.get(f"{base_url}?f=json", timeout=30)
        response.raise_for_status()
        service_info = response.json()
        
        layers = []
        
        # Extract layer information
        if 'layers' in service_info:
            for layer in service_info['layers']:
                layer_info = {
                    'id': str(layer['id']),
                    'name': layer['name'],
                    'description': layer.get('description', ''),
                    'type': layer.get('type', 'Feature Layer')
                }
                layers.append(layer_info)
                logger.info(f"📋 Found layer {layer['id']}: {layer['name']}")
        
        # Also check tables if available
        if 'tables' in service_info:
            for table in service_info['tables']:
                layer_info = {
                    'id': str(table['id']),
                    'name': table['name'],
                    'description': table.get('description', ''),
                    'type': 'Table'
                }
                layers.append(layer_info)
                logger.info(f"📊 Found table {table['id']}: {table['name']}")
        
        logger.info(f"✅ Discovered {len(layers)} layers/tables")
        return layers
        
    except Exception as e:
        logger.error(f"Failed to discover layers: {e}")
        raise

def check_layer_fields(base_url: str, layer_id: str) -> List[Dict[str, Any]]:
    """Check the fields available in a specific layer"""
    try:
        response = requests.get(f"{base_url}/{layer_id}?f=json", timeout=30)
        response.raise_for_status()
        layer_info = response.json()
        
        fields = []
        if 'fields' in layer_info:
            for field in layer_info['fields']:
                field_info = {
                    'name': field['name'],
                    'type': field['type'],
                    'alias': field.get('alias', field['name']),
                    'length': field.get('length', 0)
                }
                fields.append(field_info)
        
        return fields
        
    except Exception as e:
        logger.error(f"Failed to get fields for layer {layer_id}: {e}")
        return []

def find_id_field(fields: List[Dict[str, Any]]) -> str:
    """Find the most likely ID field for joining"""
    # Common ID field names to look for
    id_candidates = ['ID', 'OBJECTID', 'FID', 'ZIP', 'ZIPCODE', 'GEOID', 'FIPS']
    
    field_names = [f['name'].upper() for f in fields]
    
    for candidate in id_candidates:
        if candidate in field_names:
            # Find the original case
            for field in fields:
                if field['name'].upper() == candidate:
                    return field['name']
    
    # If no standard ID field found, return the first field
    return fields[0]['name'] if fields else 'OBJECTID'

def create_interactive_config():
    """Create configuration file interactively"""
    print("🚀 ArcGIS Data Extraction Setup")
    print("=" * 50)
    
    # Get service URL
    base_url = input("\n📍 Enter your ArcGIS service URL: ").strip()
    if not base_url:
        print("❌ Service URL is required")
        return
    
    # Remove trailing slash and detect service type
    base_url = base_url.rstrip('/')
    
    # Detect if it's a FeatureServer or MapServer
    if base_url.endswith('/FeatureServer'):
        # Keep as FeatureServer
        service_type = "FeatureServer"
    elif base_url.endswith('/MapServer'):
        # Keep as MapServer
        service_type = "MapServer"
    else:
        # Try to detect from URL or default to FeatureServer for modern services
        if 'FeatureServer' in base_url:
            base_url += '/FeatureServer' if not base_url.endswith('/FeatureServer') else ''
            service_type = "FeatureServer"
        else:
            base_url += '/MapServer' if not base_url.endswith('/MapServer') else ''
            service_type = "MapServer"
    
    print(f"🔍 Detected service type: {service_type}")
    print(f"🌐 Service URL: {base_url}")
    
    try:
        # Discover layers
        available_layers = discover_service_layers(base_url)
        
        if not available_layers:
            print("❌ No layers found in the service")
            return
        
        print(f"\n📋 Available layers:")
        for i, layer in enumerate(available_layers):
            print(f"  {i+1}. Layer {layer['id']}: {layer['name']} ({layer['type']})")
        
        # Select layers to extract
        print(f"\n🎯 Which layers would you like to extract?")
        print("Enter layer numbers separated by commas (e.g., 1,2,3) or 'all' for all layers:")
        
        selection = input("Selection: ").strip().lower()
        
        if selection == 'all':
            selected_layers = available_layers
        else:
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                selected_layers = [available_layers[i] for i in indices if 0 <= i < len(available_layers)]
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                return
        
        if not selected_layers:
            print("❌ No layers selected")
            return
        
        print(f"\n✅ Selected {len(selected_layers)} layers")
        
        # Check ID field in first layer
        first_layer_fields = check_layer_fields(base_url, selected_layers[0]['id'])
        suggested_id_field = find_id_field(first_layer_fields)
        
        print(f"\n🔑 Available fields in first layer:")
        for field in first_layer_fields[:10]:  # Show first 10 fields
            print(f"  - {field['name']} ({field['type']})")
        if len(first_layer_fields) > 10:
            print(f"  ... and {len(first_layer_fields) - 10} more fields")
        
        id_field = input(f"\n🔗 Enter the field to join on [{suggested_id_field}]: ").strip()
        if not id_field:
            id_field = suggested_id_field
        
        # Output file
        output_file = input(f"\n💾 Enter output CSV file path [data/nesto_merge_0.csv]: ").strip()
        if not output_file:
            output_file = "data/nesto_merge_0.csv"
        
        # Create configuration
        config = {
            "base_url": base_url,
            "layers": selected_layers,
            "id_field": id_field,
            "where_clause": "1=1",
            "output_file": output_file,
            "max_records_per_request": 2000
        }
        
        # Save configuration
        config_path = "config/feature_services.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Configuration saved to {config_path}")
        print(f"\n🚀 To extract data, run:")
        print(f"   python extract_arcgis_data.py --config {config_path}")
        
        return config_path
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return None

def main():
    """Main setup function"""
    try:
        config_path = create_interactive_config()
        
        if config_path:
            print(f"\n📋 Next steps:")
            print(f"1. Review the configuration in {config_path}")
            print(f"2. Run: python extract_arcgis_data.py --config {config_path}")
            print(f"3. The extracted data will be ready for the SHAP microservice!")
        
    except KeyboardInterrupt:
        print(f"\n👋 Setup cancelled by user")
    except Exception as e:
        print(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    main() 