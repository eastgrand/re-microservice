import pandas as pd
import numpy as np
import json
import os

def select_model_for_analysis(query):
    """Select the best pre-calculated model based on the analysis request"""
    
    analysis_type = query.get('analysis_type', 'correlation')
    target_variable = query.get('target_variable', 'CONVERSION_RATE')
    focus_area = query.get('focus_area', None)  # New parameter: 'demographic', 'financial', 'geographic'
    
    # Load available models metadata
    metadata_path = 'precalculated/models/metadata.json'
    if not os.path.exists(metadata_path):
        return 'conversion'  # Fallback to default
    
    with open(metadata_path, 'r') as f:
        available_models = json.load(f)
    
    # Model selection logic
    if target_variable == 'CONVERSION_RATE':
        if focus_area == 'demographic':
            return 'demographic_analysis'
        elif focus_area == 'geographic': 
            return 'geographic_analysis'
        else:
            return 'conversion'  # General conversion model
    elif target_variable == 'SUM_FUNDED':
        return 'volume'
    elif target_variable == 'FREQUENCY':
        return 'frequency'
    else:
        # Default to conversion model
        return 'conversion'

def load_precalculated_model_data(model_name):
    """Load pre-calculated SHAP data for specified model"""
    
    # Load metadata
    metadata_path = 'precalculated/models/metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if model_name not in metadata:
        raise ValueError(f"Model {model_name} not found in pre-calculated data")
    
    model_info = metadata[model_name]
    
    # Load pre-calculated SHAP data
    shap_file = model_info['shap_file']
    precalc_df = pd.read_pickle(shap_file, compression='gzip')
    
    return precalc_df, model_info

def enhanced_analysis_worker(query):
    """Enhanced analysis worker with multiple model support"""
    
    try:
        # Select appropriate model
        selected_model = select_model_for_analysis(query)
        logger.info(f"Selected model: {selected_model} for analysis")
        
        # Load pre-calculated data for selected model
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_variable = model_info['target']
        features = model_info['features']
        
        logger.info(f"Using model: {selected_model}")
        logger.info(f"Target: {target_variable}")
        logger.info(f"Features: {len(features)}")
        logger.info(f"Model R² score: {model_info['r2_score']:.4f}")
        
        # Apply filters to data (same as before)
        filters = query.get('demographic_filters', [])
        filtered_ids = apply_filters_to_get_ids(precalc_df, filters, target_variable)
        
        # Get top areas based on target variable
        top_data = precalc_df[precalc_df['ID'].isin(filtered_ids)]
        top_data = top_data.nlargest(25, target_variable)
        
        # Extract SHAP values for this model's features
        shap_columns = [f'shap_{feature}' for feature in features if f'shap_{feature}' in precalc_df.columns]
        shap_values_data = top_data[shap_columns].values
        
        # Calculate feature importance
        feature_importance = []
        for i, feature in enumerate(features):
            if f'shap_{feature}' in precalc_df.columns:
                shap_col_idx = [j for j, col in enumerate(shap_columns) if col == f'shap_{feature}'][0]
                importance = abs(shap_values_data[:, shap_col_idx]).mean()
                feature_importance.append({'feature': feature, 'importance': float(importance)})
        
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        # Build results
        results = []
        for _, row in top_data.iterrows():
            result = {
                'zip_code': str(row['ID']),
                target_variable.lower(): float(row[target_variable])
            }
            
            # Add key demographic/geographic fields based on model type
            if 'demographic' in selected_model:
                demo_fields = ['2024 Total Population', '2024 Visible Minority Total Population (%)']
                for field in demo_fields:
                    if field in row:
                        result[field.lower().replace(' ', '_').replace('(%)', '_pct')] = float(row[field])
            elif 'geographic' in selected_model:
                geo_fields = ['2024 Structure Type Single-Detached House (%)', '2024 Condominium Status - In Condo (%)']
                for field in geo_fields:
                    if field in row:
                        result[field.lower().replace(' ', '_').replace('(%)', '_pct')] = float(row[field])
            
            results.append(result)
        
        # Generate model-specific summary
        if len(feature_importance) > 0:
            top_factor = feature_importance[0]['feature']
            model_desc = model_info['description']
            summary = f"{model_desc}. The top factor influencing {target_variable} is {top_factor}."
        else:
            summary = f"Analysis complete using {selected_model} model."
        
        # Add top 3 factors if available
        if len(feature_importance) >= 3:
            summary += f" The top 3 factors are {feature_importance[0]['feature']}, "
            summary += f"{feature_importance[1]['feature']}, and {feature_importance[2]['feature']}."
        
        # Build SHAP values dict
        shap_values_dict = {}
        for i, feature in enumerate(features):
            if f'shap_{feature}' in precalc_df.columns:
                shap_col_idx = [j for j, col in enumerate(shap_columns) if col == f'shap_{feature}'][0]
                shap_values_dict[feature] = shap_values_data[:, shap_col_idx].tolist()[:10]
        
        return {
            "success": True,
            "results": results,
            "summary": summary,
            "feature_importance": feature_importance,
            "shap_values": shap_values_dict,
            "model_info": {
                "model_name": selected_model,
                "model_description": model_info['description'],
                "target_variable": target_variable,
                "feature_count": len(features),
                "r2_score": model_info['r2_score']
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced analysis error: {str(e)}")
        return {"success": False, "error": str(e)}

def apply_filters_to_get_ids(df, filters, target_col):
    """Apply filters and return matching IDs"""
    filtered_data = df.copy()
    
    for filter_item in filters:
        if isinstance(filter_item, dict):
            field = filter_item.get('field')
            operator = filter_item.get('operator')
            value = filter_item.get('value')
            
            if field in filtered_data.columns:
                if operator == 'isNotNull':
                    filtered_data = filtered_data[filtered_data[field].notna()]
                elif operator == 'greaterThan' and value is not None:
                    filtered_data = filtered_data[filtered_data[field] > value]
                elif operator == 'lessThan' and value is not None:
                    filtered_data = filtered_data[filtered_data[field] < value]
        elif isinstance(filter_item, str):
            # Handle legacy string filters
            if '>' in filter_item:
                feature, value = filter_item.split('>')
                feature = feature.strip()
                value = float(value.strip())
                if feature in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data[feature] > value]
    
    return filtered_data['ID'].tolist()

# Example usage patterns for different analysis types:

ANALYSIS_EXAMPLES = {
    "demographic_focus": {
        "target_variable": "CONVERSION_RATE",
        "focus_area": "demographic",
        "demographic_filters": [
            {"field": "2024 Visible Minority Total Population (%)", "operator": "greaterThan", "value": 10}
        ],
        "analysis_type": "correlation",
        "query": "How do demographic factors affect conversion rates in diverse areas?"
    },
    
    "financial_analysis": {
        "target_variable": "SUM_FUNDED", 
        "focus_area": "financial",
        "demographic_filters": [
            {"field": "2024 Household Average Income (Current Year $)", "operator": "greaterThan", "value": 80000}
        ],
        "analysis_type": "ranking",
        "query": "Which high-income areas generate the most loan volume?"
    },
    
    "geographic_patterns": {
        "target_variable": "CONVERSION_RATE",
        "focus_area": "geographic", 
        "demographic_filters": [
            {"field": "2024 Structure Type Single-Detached House (%)", "operator": "greaterThan", "value": 50}
        ],
        "analysis_type": "correlation",
        "query": "How do housing types affect conversion rates?"
    },
    
    "frequency_analysis": {
        "target_variable": "FREQUENCY",
        "demographic_filters": [
            {"field": "2024 Total Population", "operator": "greaterThan", "value": 5000}
        ],
        "analysis_type": "ranking", 
        "query": "Which populated areas have the highest application frequency?"
    }
} 