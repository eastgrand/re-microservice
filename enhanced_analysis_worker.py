import pandas as pd
import numpy as np
import json
import os
import logging
from typing import List, Dict

# Import the query classifier
from query_classifier import QueryClassifier, process_query

# Set up logging
logger = logging.getLogger(__name__)

def select_model_for_analysis(query):
    """Select the best pre-calculated model based on the analysis request"""
    
    # For now, we only have one model available, but we'll use query classification
    # to determine how to analyze the data differently
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
    """Enhanced analysis worker with query-aware analysis"""
    
    try:
        # Extract the actual user query from the request
        user_query = query.get('query', '')
        analysis_type = query.get('analysis_type', 'correlation')
        conversation_context = query.get('conversationContext', '')
        
        logger.info(f"Processing query: {user_query}")
        logger.info(f"Analysis type: {analysis_type}")
        
        # Use query classifier to understand the query intent
        classifier = QueryClassifier()
        query_classification = process_query(user_query)
        
        logger.info(f"Query classification: {query_classification}")
        
        # Select appropriate model (for now, always 'conversion')
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
        
        # Apply query-aware filtering and analysis
        filtered_data, query_specific_features = apply_query_aware_analysis(
            precalc_df, query_classification, user_query, conversation_context
        )
        
        # Apply additional filters from request
        filters = query.get('demographic_filters', [])
        if filters:
            filtered_ids = apply_filters_to_get_ids(filtered_data, filters, target_variable)
            filtered_data = filtered_data[filtered_data['ID'].isin(filtered_ids)]
        
        # Get top areas based on query-specific ranking
        top_data = get_query_aware_top_areas(
            filtered_data, query_classification, target_variable, user_query
        )
        
        # Calculate query-aware feature importance
        feature_importance = calculate_query_aware_feature_importance(
            top_data, query_specific_features, query_classification
        )
        
        # Build results with query-specific fields
        results = build_query_aware_results(top_data, target_variable, query_classification)
        
        # Build SHAP values dict for relevant features
        shap_values_dict = {}
        for feature in query_specific_features[:10]:  # Top 10 most relevant features
            shap_col = f'shap_{feature}'
            if shap_col in top_data.columns:
                shap_values_dict[feature] = top_data[shap_col].tolist()[:10]
        
        return {
            "success": True,
            "analysisType": analysis_type,  # Include analysis type at top level for frontend
            "results": results,
            "summary": None,  # Let Claude handle all narrative explanations
            "feature_importance": feature_importance,
            "shap_values": shap_values_dict,
            "model_info": {
                "model_name": selected_model,
                "model_description": f"{model_info['description']} - Query-aware analysis for: {query_classification.get('query_type', 'general')}",
                "target_variable": target_variable,
                "feature_count": len(query_specific_features),
                "r2_score": model_info['r2_score'],
                "query_classification": query_classification
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced analysis error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

def apply_query_aware_analysis(df, query_classification, user_query, conversation_context):
    """Apply query-specific filtering and feature selection"""
    
    query_type = query_classification.get('query_type', 'unknown')
    target_variable = query_classification.get('target_variable', 'CONVERSION_RATE')
    
    logger.info(f"Applying query-aware analysis for type: {query_type}")
    
    # Start with all data
    filtered_data = df.copy()
    
    # Define query-specific feature priorities
    query_specific_features = []
    
    if 'diversity' in user_query.lower() or 'population' in user_query.lower():
        # Prioritize demographic features
        demographic_features = [f for f in df.columns if 'Visible Minority' in f or 'Population' in f]
        query_specific_features.extend(demographic_features)
        logger.info(f"Diversity query detected - prioritizing {len(demographic_features)} demographic features")
        
    if 'income' in user_query.lower() or 'financial' in user_query.lower():
        # Prioritize financial features
        financial_features = [f for f in df.columns if 'Income' in f or 'Financial' in f or 'Mortgage' in f]
        query_specific_features.extend(financial_features)
        logger.info(f"Income query detected - prioritizing {len(financial_features)} financial features")
        
    if 'housing' in user_query.lower() or 'condo' in user_query.lower() or 'structure' in user_query.lower():
        # Prioritize housing features
        housing_features = [f for f in df.columns if 'Structure' in f or 'Housing' in f or 'Condominium' in f or 'Tenure' in f]
        query_specific_features.extend(housing_features)
        logger.info(f"Housing query detected - prioritizing {len(housing_features)} housing features")
    
    # If no specific features identified, use top features by SHAP importance
    if not query_specific_features:
        # Get all SHAP columns and find the most important ones
        shap_columns = [col for col in df.columns if col.startswith('shap_')]
        if shap_columns:
            # Calculate average absolute SHAP values for each feature
            feature_importance_scores = {}
            for shap_col in shap_columns:
                feature_name = shap_col.replace('shap_', '')
                if feature_name in df.columns:
                    avg_importance = abs(df[shap_col]).mean()
                    feature_importance_scores[feature_name] = avg_importance
            
            # Sort by importance and take top features
            sorted_features = sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)
            query_specific_features = [f[0] for f in sorted_features[:20]]
            logger.info(f"Using top {len(query_specific_features)} features by SHAP importance")
    
    return filtered_data, query_specific_features

def get_query_aware_top_areas(df, query_classification, target_variable, user_query):
    """Get top areas using query-aware ranking"""
    
    query_type = query_classification.get('query_type', 'unknown')
    
    # Only apply ranking/limiting for ranking queries
    if query_type == 'ranking':
        # For diversity queries, we might want to weight areas with higher diversity
        if 'diversity' in user_query.lower():
            # Create a diversity score
            diversity_cols = [col for col in df.columns if 'Visible Minority' in col and col != 'value_2024 Visible Minority Total Population (%)']
            if diversity_cols and 'value_2024 Visible Minority Total Population (%)' in df.columns:
                # Calculate diversity index (higher when multiple groups are present)
                df_copy = df.copy()
                df_copy['diversity_score'] = df_copy['value_2024 Visible Minority Total Population (%)']
                
                # Combine with conversion rate for ranking
                df_copy['combined_score'] = (
                    df_copy[target_variable] * 0.7 + 
                    df_copy['diversity_score'] * 0.3
                )
                
                top_data = df_copy.nlargest(25, 'combined_score')
                logger.info("Applied diversity-aware ranking")
                return top_data
        
        # For income queries, weight by income levels
        elif 'income' in user_query.lower():
            income_col = 'value_2024 Household Average Income (Current Year $)'
            if income_col in df.columns:
                df_copy = df.copy()
                # Normalize income for scoring
                income_normalized = (df_copy[income_col] - df_copy[income_col].min()) / (df_copy[income_col].max() - df_copy[income_col].min())
                
                df_copy['combined_score'] = (
                    df_copy[target_variable] * 0.7 + 
                    income_normalized * 0.3
                )
                
                top_data = df_copy.nlargest(25, 'combined_score')
                logger.info("Applied income-aware ranking")
                return top_data
        
        # For housing queries, weight by housing characteristics
        elif 'condo' in user_query.lower():
            condo_col = 'value_2024 Condominium Status - In Condo (%)'
            if condo_col in df.columns:
                df_copy = df.copy()
                df_copy['combined_score'] = (
                    df_copy[target_variable] * 0.7 + 
                    df_copy[condo_col] * 0.3
                )
                
                top_data = df_copy.nlargest(25, 'combined_score')
                logger.info("Applied condo-aware ranking")
                return top_data
        
        # Default ranking by target variable only for ranking queries
        top_data = df.nlargest(25, target_variable)
        logger.info("Applied default ranking by target variable")
        return top_data
    
    # For non-ranking queries (correlation, etc.), return all data
    logger.info(f"Returning all data for {query_type} query type")
    return df

def calculate_query_aware_feature_importance(df, query_specific_features, query_classification):
    """Calculate feature importance with query-specific weighting"""
    
    feature_importance = []
    
    for feature in query_specific_features:
        shap_col = f'shap_{feature}'
        if shap_col in df.columns:
            importance = abs(df[shap_col]).mean()
            feature_importance.append({
                'feature': feature,
                'importance': float(importance)
            })
    
    # Sort by importance
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    # Limit to top 15 features for clarity
    return feature_importance[:15]

def build_query_aware_results(df, target_variable, query_classification):
    """Build results with query-specific field selection"""
    
    results = []
    for _, row in df.iterrows():
        result = {
            'geo_id': str(row['ID']),
            'FSA_ID': str(row['ID']),
            'ID': str(row['ID']),
            target_variable.lower(): float(row[target_variable])
        }
        
        # Always include conversion rate
        if 'CONVERSION_RATE' in row:
            result['conversion_rate'] = float(row['CONVERSION_RATE'])
        
        # Add query-specific fields
        # Demographic fields
        if 'value_2024 Visible Minority Total Population (%)' in row:
            result['visible_minority_population_pct'] = float(row['value_2024 Visible Minority Total Population (%)'])
        
        if 'value_2024 Total Population' in row:
            result['total_population'] = float(row['value_2024 Total Population'])
        
        # Financial fields
        if 'value_2024 Household Average Income (Current Year $)' in row:
            result['household_average_income'] = float(row['value_2024 Household Average Income (Current Year $)'])
        
        # Housing fields
        if 'value_2024 Structure Type Single-Detached House (%)' in row:
            result['single_detached_house_pct'] = float(row['value_2024 Structure Type Single-Detached House (%)'])
        
        if 'value_2024 Condominium Status - In Condo (%)' in row:
            result['condominium_pct'] = float(row['value_2024 Condominium Status - In Condo (%)'])
        
        # Add combined score if it was calculated
        if 'combined_score' in row:
            result['combined_score'] = float(row['combined_score'])
        
        results.append(result)
    
    return results

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
    """Analyze the user's query to understand their analytical intent
    
    Note: This function no longer generates summaries - that's handled by Claude in the narrative pass.
    It only provides query type classification and feature importance filtering.
    """
    
    query_lower = query.lower()
    
    # Determine query type based on keywords and patterns
    if target_field.lower() == 'frequency' and ('application' in query_lower or 'applications' in query_lower):
        query_type = 'topN'
        # Filter to most relevant demographic factors for application analysis
        relevant_features = [f for f in feature_importance if 
            any(term in f['feature'].lower() for term in ['household', 'income', 'population', 'density']) and
            abs(f.get('importance', 0)) > 0.3]
        
        return {
            'summary': None,  # No summary - Claude handles this
            'feature_importance': relevant_features,
            'query_type': query_type
        }
    
    # Default case - return all feature importance for Claude to analyze
    return {
        'summary': None,  # No summary - Claude handles this
        'feature_importance': feature_importance,
        'query_type': 'analysis'
    }

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