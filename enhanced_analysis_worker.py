import pandas as pd
import numpy as np
import json
import os
import logging
from typing import List, Dict

# Set up logging
logger = logging.getLogger(__name__)

def select_model_for_analysis(query):
    """Select the best pre-calculated model based on the analysis request"""
    
    analysis_type = query.get('analysis_type', 'correlation')
    target_variable = query.get('target_variable', 'CONVERSION_RATE')
    focus_area = query.get('focus_area', None)
    user_query = query.get('query', '').lower()
    conversation_context = query.get('conversationContext', '').lower()
    
    # Load available models metadata
    metadata_path = 'precalculated/models/metadata.json'
    if not os.path.exists(metadata_path):
        return 'conversion'  # Fallback to default
    
    with open(metadata_path, 'r') as f:
        available_models = json.load(f)
    
    # Check for specific concepts in query and context
    specific_concepts = set()
    
    # Check query for specific concepts
    if 'diversity' in user_query or 'population' in user_query:
        specific_concepts.add('demographic')
    if 'income' in user_query or 'financial' in user_query:
        specific_concepts.add('financial')
    if 'housing' in user_query or 'structure' in user_query:
        specific_concepts.add('geographic')
    
    # Check conversation context for additional concepts
    if conversation_context:
        if 'diversity' in conversation_context or 'population' in conversation_context:
            specific_concepts.add('demographic')
        if 'income' in conversation_context or 'financial' in conversation_context:
            specific_concepts.add('financial')
        if 'housing' in conversation_context or 'structure' in conversation_context:
            specific_concepts.add('geographic')
    
    # Model selection logic based on specific concepts
    if target_variable == 'CONVERSION_RATE':
        if 'demographic' in specific_concepts and 'financial' not in specific_concepts:
            return 'demographic_analysis'
        elif 'geographic' in specific_concepts and 'financial' not in specific_concepts:
            return 'geographic_analysis'
        elif 'financial' in specific_concepts:
            return 'financial_analysis'
        else:
            return 'conversion'  # General conversion model
    elif target_variable == 'SUM_FUNDED':
        if 'demographic' in specific_concepts:
            return 'demographic_volume'
        elif 'geographic' in specific_concepts:
            return 'geographic_volume'
        else:
            return 'volume'
    elif target_variable == 'FREQUENCY':
        if 'demographic' in specific_concepts:
            return 'demographic_frequency'
        elif 'geographic' in specific_concepts:
            return 'geographic_frequency'
        else:
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
                'geo_id': str(row['ID']),
                'FSA_ID': str(row['ID']),  # Add FSA_ID for geographic joining
                'ID': str(row['ID']),      # Add ID for geographic joining
                target_variable.lower(): float(row[target_variable])
            }
            
            # Add key demographic/geographic fields with expected field names
            # Always include conversion rate if available
            if 'CONVERSION_RATE' in row:
                result['conversion_rate'] = float(row['CONVERSION_RATE'])
            
            # Add demographic fields with expected names (using correct field names with 'value_' prefix)
            if 'value_2024 Visible Minority Total Population (%)' in row:
                result['visible_minority_population_pct'] = float(row['value_2024 Visible Minority Total Population (%)'])
            
            # Add other common fields that might be expected
            if 'value_2024 Total Population' in row:
                result['total_population'] = float(row['value_2024 Total Population'])
            
            if 'value_2024 Household Average Income (Current Year $)' in row:
                result['household_average_income'] = float(row['value_2024 Household Average Income (Current Year $)'])
            
            # Add geographic fields
            if 'value_2024 Structure Type Single-Detached House (%)' in row:
                result['single_detached_house_pct'] = float(row['value_2024 Structure Type Single-Detached House (%)'])
            
            if 'value_2024 Condominium Status - In Condo (%)' in row:
                result['condominium_pct'] = float(row['value_2024 Condominium Status - In Condo (%)'])
            
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

def analyze_query_intent(query: str, target_field: str, results: List[Dict], feature_importance: List[Dict], conversation_context: str = '') -> Dict[str, any]:
    """Analyze the user's query to understand their analytical intent"""
    
    query_lower = query.lower()
    
    # Special handling for application count queries
    if target_field.lower() == 'frequency' and ('application' in query_lower or 'applications' in query_lower):
        if not results or len(results) == 0:
            return {
                'summary': 'No areas found with mortgage applications matching your criteria.',
                'feature_importance': [],
                'query_type': 'topN'
            }
        
        # Get top areas by application count
        top_areas = [result.get('FSA_ID', result.get('ID', 'Unknown Area')) for result in results[:5]]
        top_counts = [int(result.get('FREQUENCY', 0)) for result in results[:5]]
        
        # Generate summary focusing only on application counts
        summary = f"The areas with the most mortgage applications are {top_areas[0]} ({top_counts[0]} applications)"
        if len(top_areas) > 1:
            summary += f", followed by {top_areas[1]} ({top_counts[1]} applications)"
        if len(top_areas) > 2:
            summary += f", and {top_areas[2]} ({top_counts[2]} applications)"
        summary += "."
        
        # Only include relevant demographic factors if they significantly correlate with application counts
        relevant_features = [f for f in feature_importance if 
            any(term in f['feature'].lower() for term in ['household', 'income', 'population', 'density']) and
            abs(f.get('importance', 0)) > 0.3]  # Only include if correlation is significant
        
        if relevant_features:
            summary += f" These areas tend to have higher {relevant_features[0]['feature'].lower().replace('_', ' ')}"
            if len(relevant_features) > 1:
                summary += f" and {relevant_features[1]['feature'].lower().replace('_', ' ')}"
            summary += "."
        
        return {
            'summary': summary,
            'feature_importance': relevant_features,
            'query_type': 'topN'
        }
    
    # Handle other query types
    # ... existing code for other analysis types ...

# Example usage patterns for different analysis types:

ANALYSIS_EXAMPLES = {
    "demographic_focus": {
        "target_variable": "CONVERSION_RATE",
        "focus_area": "demographic",
        "demographic_filters": [
            {"field": "value_2024 Visible Minority Total Population (%)", "operator": "greaterThan", "value": 10}
        ],
        "analysis_type": "correlation",
        "query": "How do demographic factors affect conversion rates in diverse areas?"
    },
    
    "financial_analysis": {
        "target_variable": "SUM_FUNDED", 
        "focus_area": "financial",
        "demographic_filters": [
            {"field": "value_2024 Household Average Income (Current Year $)", "operator": "greaterThan", "value": 80000}
        ],
        "analysis_type": "ranking",
        "query": "Which high-income areas generate the most loan volume?"
    },
    
    "geographic_patterns": {
        "target_variable": "CONVERSION_RATE",
        "focus_area": "geographic", 
        "demographic_filters": [
            {"field": "value_2024 Structure Type Single-Detached House (%)", "operator": "greaterThan", "value": 50}
        ],
        "analysis_type": "correlation",
        "query": "How do housing types affect conversion rates?"
    },
    
    "frequency_analysis": {
        "target_variable": "FREQUENCY",
        "demographic_filters": [
            {"field": "value_2024 Total Population", "operator": "greaterThan", "value": 5000}
        ],
        "analysis_type": "ranking", 
        "query": "Which populated areas have the highest application frequency?"
    }
} 