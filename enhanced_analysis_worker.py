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
        
        # Use target variable from query classification if available, otherwise use model default
        target_variable = query_classification.get('target_variable', model_info['target'])
        
        # Handle condominium target variable - map to the actual column name in data
        if target_variable == "condo_ownership_pct":
            # Map to the actual column name in the precalculated data
            if "value_2024 Condominium Status - In Condo (%)" in precalc_df.columns:
                target_variable = "value_2024 Condominium Status - In Condo (%)"
                logger.info(f"Mapped condo_ownership_pct to actual column: {target_variable}")
            else:
                logger.warning(f"Condominium target variable not found in data, using default: {model_info['target']}")
                target_variable = model_info['target']
        elif target_variable == "median_income":
            # Map to the actual column name in the precalculated data
            if "value_2024 Household Average Income (Current Year $)" in precalc_df.columns:
                target_variable = "value_2024 Household Average Income (Current Year $)"
                logger.info(f"Mapped median_income to actual column: {target_variable}")
            else:
                logger.warning(f"Income target variable not found in data, using default: {model_info['target']}")
                target_variable = model_info['target']
        elif target_variable == "visible_minority_population_pct":
            # Map to the actual column name in the precalculated data
            if "value_2024 Visible Minority Total Population (%)" in precalc_df.columns:
                target_variable = "value_2024 Visible Minority Total Population (%)"
                logger.info(f"Mapped visible_minority_population_pct to actual column: {target_variable}")
            else:
                logger.warning(f"Visible minority target variable not found in data, using default: {model_info['target']}")
                target_variable = model_info['target']
        elif target_variable == "conversion_rate":
            # Map to the actual column name in the precalculated data
            if "CONVERSION_RATE" in precalc_df.columns:
                target_variable = "CONVERSION_RATE"
                logger.info(f"Mapped conversion_rate to actual column: {target_variable}")
            else:
                logger.warning(f"Conversion rate target variable not found in data, using default: {model_info['target']}")
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
                # Use safe_float to handle NaN values in SHAP data
                shap_values_dict[feature] = [safe_float(val) for val in top_data[shap_col].tolist()[:10]]
        
        # ADD: Enhanced spatial and geographic context for Claude
        spatial_analysis = analyze_spatial_patterns(top_data, target_variable, query_specific_features)
        regional_clusters = identify_regional_clusters(top_data, target_variable)
        geographic_context = build_geographic_context(top_data, target_variable, user_query)
        
        return {
            "success": True,
            "analysisType": analysis_type,  # Include analysis type at top level for frontend
            "results": results,
            "summary": None,  # Let Claude handle all narrative explanations
            "feature_importance": feature_importance,
            "shap_values": shap_values_dict,
            "spatial_analysis": spatial_analysis,  # NEW: Spatial clustering and distribution info
            "regional_clusters": regional_clusters,  # NEW: Geographic clusters
            "geographic_context": geographic_context,  # NEW: Regional context and patterns
            "model_info": {
                "model_name": selected_model,
                "model_description": f"{model_info['description']} - Query-aware analysis for: {query_classification.get('query_type', 'general')}",
                "target_variable": target_variable,
                "feature_count": len(query_specific_features),
                "r2_score": model_info['r2_score'],
                "query_classification": query_classification,
                "total_areas_analyzed": len(top_data),
                "analysis_scope": "Regional patterns and spatial clustering analysis included"
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
        # Get the limit from query classification, default to 25 if not specified
        limit = query_classification.get('limit', 25)
        logger.info(f"Using limit: {limit} for ranking query")
        
        # Check if we're ranking by a specific target variable (like condominium ownership)
        # If so, rank directly by that variable instead of creating combined scores
        if target_variable == "value_2024 Condominium Status - In Condo (%)":
            # For condominium ownership queries, rank directly by the condo percentage
            top_data = df.nlargest(limit, target_variable)
            logger.info(f"Applied direct ranking by {target_variable} with limit {limit}")
            return top_data
        
        # For diversity queries, we might want to weight areas with higher diversity
        elif 'diversity' in user_query.lower():
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
                
                top_data = df_copy.nlargest(limit, 'combined_score')
                logger.info(f"Applied diversity-aware ranking with limit {limit}")
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
                
                top_data = df_copy.nlargest(limit, 'combined_score')
                logger.info(f"Applied income-aware ranking with limit {limit}")
                return top_data
        
        # Default ranking by target variable only for ranking queries
        top_data = df.nlargest(limit, target_variable)
        logger.info(f"Applied default ranking by {target_variable} with limit {limit}")
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
            # Handle NaN values - convert to None which becomes null in JSON
            importance_value = float(importance) if pd.notna(importance) else None
            if importance_value is not None:  # Only add non-null values
                feature_importance.append({
                    'feature': feature,
                    'importance': importance_value
                })
    
    # Sort by importance
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    # Limit to top 15 features for clarity
    return feature_importance[:15]

def safe_float(value):
    """Safely convert value to float, replacing NaN with None for JSON compatibility"""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (ValueError, TypeError):
        return None

def build_query_aware_results(df, target_variable, query_classification):
    """Build results with query-specific field selection"""
    
    results = []
    for _, row in df.iterrows():
        result = {
            'geo_id': str(row['ID']),
            'FSA_ID': str(row['ID']),
            'ID': str(row['ID'])
        }
        
        # Add the target variable with a clean field name
        if target_variable == "value_2024 Condominium Status - In Condo (%)":
            result['condo_ownership_pct'] = safe_float(row[target_variable])
            result['target_value'] = safe_float(row[target_variable])
        elif target_variable == "value_2024 Household Average Income (Current Year $)":
            result['median_income'] = safe_float(row[target_variable])
            result['target_value'] = safe_float(row[target_variable])
        elif target_variable == "value_2024 Visible Minority Total Population (%)":
            result['visible_minority_population_pct'] = safe_float(row[target_variable])
            result['target_value'] = safe_float(row[target_variable])
        elif target_variable == "CONVERSION_RATE":
            result['conversion_rate'] = safe_float(row[target_variable])
            result['target_value'] = safe_float(row[target_variable])
        else:
            result[target_variable.lower()] = safe_float(row[target_variable])
            result['target_value'] = safe_float(row[target_variable])
        
        # Always include conversion rate if available
        if 'CONVERSION_RATE' in row:
            result['conversion_rate'] = safe_float(row['CONVERSION_RATE'])
        
        # Add query-specific fields
        # Demographic fields
        if 'value_2024 Visible Minority Total Population (%)' in row:
            result['visible_minority_population_pct'] = safe_float(row['value_2024 Visible Minority Total Population (%)'])
        
        if 'value_2024 Total Population' in row:
            result['total_population'] = safe_float(row['value_2024 Total Population'])
        
        # Financial fields
        if 'value_2024 Household Average Income (Current Year $)' in row:
            result['household_average_income'] = safe_float(row['value_2024 Household Average Income (Current Year $)'])
        
        # Housing fields
        if 'value_2024 Structure Type Single-Detached House (%)' in row:
            result['single_detached_house_pct'] = safe_float(row['value_2024 Structure Type Single-Detached House (%)'])
        
        if 'value_2024 Condominium Status - In Condo (%)' in row:
            result['condominium_pct'] = safe_float(row['value_2024 Condominium Status - In Condo (%)'])
        
        # Add combined score if it was calculated
        if 'combined_score' in row:
            result['combined_score'] = safe_float(row['combined_score'])
        
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

def analyze_spatial_patterns(df, target_variable, features):
    """Analyze spatial distribution patterns and clustering"""
    
    if len(df) < 3:
        return {
            "distribution_type": "insufficient_data",
            "clustering_strength": 0,
            "spatial_autocorrelation": 0,
            "pattern_description": "Insufficient data for spatial pattern analysis"
        }
    
    # Calculate basic distribution statistics
    values = df[target_variable].dropna()
    if len(values) == 0:
        return {
            "distribution_type": "no_data",
            "clustering_strength": 0,
            "spatial_autocorrelation": 0,
            "pattern_description": "No valid data for spatial analysis"
        }
    
    # Analyze value distribution
    mean_val = values.mean()
    std_val = values.std()
    cv = std_val / mean_val if mean_val != 0 else 0
    
    # Determine clustering strength based on coefficient of variation
    if cv < 0.2:
        clustering_type = "uniform"
        clustering_strength = 0.2
    elif cv < 0.5:
        clustering_type = "moderate_clustering"
        clustering_strength = 0.6
    else:
        clustering_type = "high_clustering"
        clustering_strength = 0.9
    
    # Calculate quartile distribution
    q1 = values.quantile(0.25)
    q2 = values.quantile(0.50)
    q3 = values.quantile(0.75)
    
    # Identify outliers
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[target_variable] < lower_bound) | (df[target_variable] > upper_bound)]
    
    return {
        "distribution_type": clustering_type,
        "clustering_strength": safe_float(clustering_strength),
        "spatial_autocorrelation": safe_float(min(1.0, cv)),  # Use CV as proxy for spatial autocorrelation
        "value_statistics": {
            "mean": safe_float(mean_val),
            "median": safe_float(q2),
            "std_deviation": safe_float(std_val),
            "coefficient_variation": safe_float(cv),
            "quartiles": {
                "q1": safe_float(q1),
                "q2": safe_float(q2),
                "q3": safe_float(q3)
            }
        },
        "outlier_analysis": {
            "outlier_count": len(outliers),
            "outlier_percentage": safe_float((len(outliers) / len(df)) * 100),
            "outlier_threshold_low": safe_float(lower_bound),
            "outlier_threshold_high": safe_float(upper_bound)
        },
        "pattern_description": f"Data shows {clustering_type} with {len(outliers)} outliers ({(len(outliers)/len(df)*100):.1f}% of areas)"
    }

def identify_regional_clusters(df, target_variable):
    """Identify and analyze regional clusters based on values"""
    
    if len(df) < 5:
        return {
            "cluster_count": 1,
            "clusters": [{"name": "all_areas", "area_count": len(df), "avg_value": safe_float(df[target_variable].mean())}],
            "cluster_description": "Insufficient data for cluster analysis"
        }
    
    values = df[target_variable].dropna()
    if len(values) == 0:
        return {
            "cluster_count": 0,
            "clusters": [],
            "cluster_description": "No valid data for clustering"
        }
    
    # Use quartile-based clustering for simplicity
    q25 = values.quantile(0.25)
    q50 = values.quantile(0.50)
    q75 = values.quantile(0.75)
    
    # Define clusters based on quartiles
    clusters = []
    
    # High performing areas (top quartile)
    high_areas = df[df[target_variable] >= q75]
    if len(high_areas) > 0:
        clusters.append({
            "name": "high_performance",
            "label": "High Performance Areas",
            "area_count": len(high_areas),
            "avg_value": safe_float(high_areas[target_variable].mean()),
            "min_value": safe_float(high_areas[target_variable].min()),
            "max_value": safe_float(high_areas[target_variable].max()),
            "percentage_of_total": safe_float((len(high_areas) / len(df)) * 100),
            "areas": high_areas['ID'].tolist()[:10]  # First 10 area IDs
        })
    
    # Medium-high performing areas (3rd quartile)
    med_high_areas = df[(df[target_variable] >= q50) & (df[target_variable] < q75)]
    if len(med_high_areas) > 0:
        clusters.append({
            "name": "medium_high_performance",
            "label": "Medium-High Performance Areas",
            "area_count": len(med_high_areas),
            "avg_value": safe_float(med_high_areas[target_variable].mean()),
            "min_value": safe_float(med_high_areas[target_variable].min()),
            "max_value": safe_float(med_high_areas[target_variable].max()),
            "percentage_of_total": safe_float((len(med_high_areas) / len(df)) * 100),
            "areas": med_high_areas['ID'].tolist()[:10]
        })
    
    # Medium-low performing areas (2nd quartile)
    med_low_areas = df[(df[target_variable] >= q25) & (df[target_variable] < q50)]
    if len(med_low_areas) > 0:
        clusters.append({
            "name": "medium_low_performance",
            "label": "Medium-Low Performance Areas",
            "area_count": len(med_low_areas),
            "avg_value": safe_float(med_low_areas[target_variable].mean()),
            "min_value": safe_float(med_low_areas[target_variable].min()),
            "max_value": safe_float(med_low_areas[target_variable].max()),
            "percentage_of_total": safe_float((len(med_low_areas) / len(df)) * 100),
            "areas": med_low_areas['ID'].tolist()[:10]
        })
    
    # Low performing areas (bottom quartile)
    low_areas = df[df[target_variable] < q25]
    if len(low_areas) > 0:
        clusters.append({
            "name": "low_performance",
            "label": "Low Performance Areas",
            "area_count": len(low_areas),
            "avg_value": safe_float(low_areas[target_variable].mean()),
            "min_value": safe_float(low_areas[target_variable].min()),
            "max_value": safe_float(low_areas[target_variable].max()),
            "percentage_of_total": safe_float((len(low_areas) / len(df)) * 100),
            "areas": low_areas['ID'].tolist()[:10]
        })
    
    return {
        "cluster_count": len(clusters),
        "clusters": clusters,
        "cluster_method": "quartile_based",
        "cluster_description": f"Identified {len(clusters)} performance-based regional clusters across {len(df)} areas"
    }

def build_geographic_context(df, target_variable, user_query):
    """Build comprehensive geographic context for spatial analysis"""
    
    # Analyze geographic spread and diversity
    total_areas = len(df)
    values = df[target_variable].dropna()
    
    if len(values) == 0:
        return {
            "geographic_scope": "no_data",
            "area_coverage": 0,
            "regional_patterns": {},
            "context_description": "No geographic data available for analysis"
        }
    
    # Calculate geographic diversity metrics
    unique_ids = df['ID'].nunique()
    value_range = values.max() - values.min()
    mean_value = values.mean()
    
    # Identify top and bottom performing areas with geographic context
    top_5 = df.nlargest(5, target_variable)
    bottom_5 = df.nsmallest(5, target_variable)
    
    # Build regional patterns based on available demographic/economic indicators
    regional_patterns = {}
    
    # Check for diversity patterns
    diversity_col = 'value_2024 Visible Minority Total Population (%)'
    if diversity_col in df.columns:
        diversity_corr = df[[target_variable, diversity_col]].corr().iloc[0, 1]
        regional_patterns['diversity_relationship'] = {
            "correlation": safe_float(diversity_corr),
            "description": f"{'Positive' if diversity_corr > 0.1 else 'Negative' if diversity_corr < -0.1 else 'Weak'} relationship between diversity and {target_variable.lower()}"
        }
    
    # Check for income patterns
    income_col = 'value_2024 Household Average Income (Current Year $)'
    if income_col in df.columns:
        income_corr = df[[target_variable, income_col]].corr().iloc[0, 1]
        regional_patterns['income_relationship'] = {
            "correlation": safe_float(income_corr),
            "description": f"{'Positive' if income_corr > 0.1 else 'Negative' if income_corr < -0.1 else 'Weak'} relationship between income and {target_variable.lower()}"
        }
    
    # Check for housing patterns
    housing_col = 'value_2024 Structure Type Single-Detached House (%)'
    if housing_col in df.columns:
        housing_corr = df[[target_variable, housing_col]].corr().iloc[0, 1]
        regional_patterns['housing_relationship'] = {
            "correlation": safe_float(housing_corr),
            "description": f"{'Positive' if housing_corr > 0.1 else 'Negative' if housing_corr < -0.1 else 'Weak'} relationship between single-detached housing and {target_variable.lower()}"
        }
    
    # Determine geographic scope based on query and data
    if 'province' in user_query.lower() or 'provincial' in user_query.lower():
        geographic_scope = "provincial"
    elif 'city' in user_query.lower() or 'urban' in user_query.lower():
        geographic_scope = "urban"
    elif 'region' in user_query.lower() or 'regional' in user_query.lower():
        geographic_scope = "regional"
    else:
        geographic_scope = "multi_area"
    
    return {
        "geographic_scope": geographic_scope,
        "area_coverage": total_areas,
        "value_statistics": {
            "range": safe_float(value_range),
            "mean": safe_float(mean_value),
            "geographic_diversity": safe_float(value_range / mean_value if mean_value != 0 else 0)
        },
        "top_performers": {
            "count": len(top_5),
            "areas": top_5['ID'].tolist(),
            "avg_value": safe_float(top_5[target_variable].mean()),
            "value_range": safe_float(top_5[target_variable].max() - top_5[target_variable].min())
        },
        "bottom_performers": {
            "count": len(bottom_5),
            "areas": bottom_5['ID'].tolist(),
            "avg_value": safe_float(bottom_5[target_variable].mean()),
            "value_range": safe_float(bottom_5[target_variable].max() - bottom_5[target_variable].min())
        },
        "regional_patterns": regional_patterns,
        "adjacent_area_analysis": {
            "coverage_description": f"Analysis covers {total_areas} geographic areas",
            "spatial_continuity": "Available" if total_areas > 10 else "Limited",
            "pattern_reliability": "High" if total_areas > 20 else "Moderate" if total_areas > 10 else "Limited"
        },
        "context_description": f"Geographic analysis of {total_areas} areas showing {geographic_scope} patterns with {len(regional_patterns)} demographic/economic relationships identified"
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