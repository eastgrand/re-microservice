import pandas as pd
import numpy as np
import json
import os
import logging
from typing import List, Dict

# Import the query classifier
from query_processing.classifier import QueryClassifier, process_query

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
        
        # 1️⃣  Respect an explicit target_variable coming from the request – this lets the front-end
        # choose the exact metric (e.g. MP30034A_B) and avoids the legacy mortgage/income fall-backs
        requested_target = query.get('target_variable')
        
        # 🆕 NEW: Check for bivariate correlation analysis (Nike vs Adidas comparison)
        matched_fields = query.get('matched_fields', [])
        metrics = query.get('metrics', [])
        
        logger.info(f"DEBUG: matched_fields = {matched_fields}")
        logger.info(f"DEBUG: metrics = {metrics}")
        logger.info(f"DEBUG: analysis_type = {analysis_type}")
        
        # Detect bivariate correlation: if we have exactly 2 brand fields, do bivariate analysis
        all_brand_fields = [field for field in (matched_fields + metrics) if field and field.startswith('MP30') and '_' in field]
        # Remove duplicates while preserving order (set() can reorder elements)  
        seen = set()
        all_brand_fields = [field for field in all_brand_fields if not (field in seen or seen.add(field))]
        
        # Ensure target_variable is first if it's a brand field (for Nike vs Adidas, Nike should be primary)
        brand_fields = []
        if requested_target and requested_target.startswith('MP30') and '_' in requested_target and requested_target in all_brand_fields:
            brand_fields.append(requested_target)
            # Add other brand fields except the target
            brand_fields.extend([f for f in all_brand_fields if f != requested_target])
        else:
            brand_fields = all_brand_fields
        
        logger.info(f"DEBUG: detected brand_fields = {brand_fields}")
        logger.info(f"DEBUG: len(brand_fields) = {len(brand_fields)}")
        logger.info(f"DEBUG: analysis_type in ['correlation', 'comparison'] = {analysis_type in ['correlation', 'comparison']}")
        
        # Trigger bivariate correlation for both 'correlation' and 'comparison' analysis types
        if len(brand_fields) == 2 and analysis_type in ['correlation', 'comparison']:
            logger.info(f"Detected bivariate correlation request: {brand_fields}")
            return handle_bivariate_correlation(precalc_df, brand_fields, user_query, query_classification)
        else:
            logger.info(f"NOT triggering bivariate correlation. Conditions: len(brand_fields)={len(brand_fields)}, analysis_type='{analysis_type}'")

        # Add comprehensive field mappings to handle all supported fields
        # This prevents query failures by mapping frontend field codes to actual column names in precalculated data
        field_aliases = {
            # Geographic and basic fields
            "OBJECTID": "OBJECTID",
            "ID": "ID", 
            "DESCRIPTION": "DESCRIPTION",
            "ZIP_CODE": "ZIP_CODE",
            
            # Demographics - Population
            "TOTPOP_CY": "value_TOTPOP_CY",
            "HHPOP_CY": "value_HHPOP_CY", 
            "HHPOP_CY_P": "value_HHPOP_CY_P",
            "FAMPOP_CY": "value_FAMPOP_CY",
            "FAMPOP_CY_P": "value_FAMPOP_CY_P",
            
            # Demographics - Race/Ethnicity  
            "DIVINDX_CY": "value_DIVINDX_CY",
            "WHITE_CY": "value_WHITE_CY",
            "WHITE_CY_P": "value_WHITE_CY_P", 
            "BLACK_CY": "value_BLACK_CY",
            "BLACK_CY_P": "value_BLACK_CY_P",
            "AMERIND_CY": "value_AMERIND_CY",
            "AMERIND_CY_P": "value_AMERIND_CY_P",
            "ASIAN_CY": "value_ASIAN_CY",
            "ASIAN_CY_P": "value_ASIAN_CY_P",
            "PACIFIC_CY": "value_PACIFIC_CY", 
            "PACIFIC_CY_P": "value_PACIFIC_CY_P",
            "OTHRACE_CY": "value_OTHRACE_CY",
            "OTHRACE_CY_P": "value_OTHRACE_CY_P",
            "RACE2UP_CY": "value_RACE2UP_CY",
            "RACE2UP_CY_P": "value_RACE2UP_CY_P",
            
            # Demographics - Hispanic
            "HISPWHT_CY": "value_HISPWHT_CY",
            "HISPWHT_CY_P": "value_HISPWHT_CY_P",
            "HISPBLK_CY": "value_HISPBLK_CY", 
            "HISPBLK_CY_P": "value_HISPBLK_CY_P",
            "HISPAI_CY": "value_HISPAI_CY",
            "HISPAI_CY_P": "value_HISPAI_CY_P",
            "HISPPI_CY": "value_HISPPI_CY",
            "HISPPI_CY_P": "value_HISPPI_CY_P", 
            "HISPOTH_CY": "value_HISPOTH_CY",
            "HISPOTH_CY_P": "value_HISPOTH_CY_P",
            
            # Demographics - Generations
            "GENZ_CY": "value_GENZ_CY",
            "GENZ_CY_P": "value_GENZ_CY_P",
            "GENALPHACY": "value_GENALPHACY",
            "GENALPHACY_P": "value_GENALPHACY_P", 
            "MILLENN_CY": "value_MILLENN_CY",
            "MILLENN_CY_P": "value_MILLENN_CY_P",
            
            # Economics
            "MEDDI_CY": "value_MEDDI_CY",
            "WLTHINDXCY": "value_WLTHINDXCY",
            "median_income": "value_MEDDI_CY",  # Keep existing alias
            
            # Sports/Recreation Equipment
            "X9051_X": "value_X9051_X",
            "X9051_X_A": "value_X9051_X_A",
            
            # Athletic Shoe Purchases (MP codes)
            "MP30016A_B": "value_MP30016A_B",  # Athletic shoes
            "MP30016A_B_P": "value_MP30016A_B_P",
            "MP30018A_B": "value_MP30018A_B",  # Basketball shoes
            "MP30018A_B_P": "value_MP30018A_B_P", 
            "MP30019A_B": "value_MP30019A_B",  # Cross-training shoes
            "MP30019A_B_P": "value_MP30019A_B_P",
            "MP30021A_B": "value_MP30021A_B",  # Running shoes
            "MP30021A_B_P": "value_MP30021A_B_P",
            
            # Brand-specific Athletic Shoes
            "MP30029A_B": "value_MP30029A_B",  # Adidas
            "MP30029A_B_P": "value_MP30029A_B_P",
            "MP30030A_B": "value_MP30030A_B",  # Asics
            "MP30030A_B_P": "value_MP30030A_B_P",
            "MP30031A_B": "value_MP30031A_B",  # Converse
            "MP30031A_B_P": "value_MP30031A_B_P",
            "MP30032A_B": "value_MP30032A_B",  # Jordan
            "MP30032A_B_P": "value_MP30032A_B_P",
            "MP30033A_B": "value_MP30033A_B",  # New Balance
            "MP30033A_B_P": "value_MP30033A_B_P", 
            "MP30034A_B": "value_MP30034A_B",  # Nike
            "MP30034A_B_P": "value_MP30034A_B_P",
            "MP30035A_B": "value_MP30035A_B",  # Puma
            "MP30035A_B_P": "value_MP30035A_B_P",
            "MP30036A_B": "value_MP30036A_B",  # Reebok
            "MP30036A_B_P": "value_MP30036A_B_P",
            "MP30037A_B": "value_MP30037A_B",  # Skechers
            "MP30037A_B_P": "value_MP30037A_B_P",
            
            # Sports Clothing Purchases
            "MP07109A_B": "value_MP07109A_B",  # $300+ sports clothing
            "MP07109A_B_P": "value_MP07109A_B_P",
            "MP07111A_B": "value_MP07111A_B",  # $100+ athletic wear
            "MP07111A_B_P": "value_MP07111A_B_P",
            "PSIV7UMKVALM": "value_PSIV7UMKVALM",  # $200+ shoes
            
            # Store Shopping
            "MP31035A_B": "value_MP31035A_B",  # Dick's Sporting Goods
            "MP31035A_B_P": "value_MP31035A_B_P",
            "MP31042A_B": "value_MP31042A_B",  # Foot Locker
            "MP31042A_B_P": "value_MP31042A_B_P",
            
            # Sports Participation
            "MP33020A_B": "value_MP33020A_B",  # Jogging/Running
            "MP33020A_B_P": "value_MP33020A_B_P",
            "MP33032A_B": "value_MP33032A_B",  # Yoga
            "MP33032A_B_P": "value_MP33032A_B_P",
            "MP33031A_B": "value_MP33031A_B",  # Weight lifting
            "MP33031A_B_P": "value_MP33031A_B_P",
            
            # Sports Fandom (Super fans)
            "MP33104A_B": "value_MP33104A_B",  # MLB
            "MP33104A_B_P": "value_MP33104A_B_P",
            "MP33105A_B": "value_MP33105A_B",  # NASCAR
            "MP33105A_B_P": "value_MP33105A_B_P",
            "MP33106A_B": "value_MP33106A_B",  # NBA
            "MP33106A_B_P": "value_MP33106A_B_P",
            "MP33107A_B": "value_MP33107A_B",  # NFL
            "MP33107A_B_P": "value_MP33107A_B_P",
            "MP33108A_B": "value_MP33108A_B",  # NHL
            "MP33108A_B_P": "value_MP33108A_B_P",
            "MP33119A_B": "value_MP33119A_B",  # International Soccer
            "MP33119A_B_P": "value_MP33119A_B_P",
            "MP33120A_B": "value_MP33120A_B",  # MLS Soccer
            "MP33120A_B_P": "value_MP33120A_B_P",
            
            # Shape fields
            "Shape__Area": "Shape__Area",
            "Shape__Length": "Shape__Length",
            
            # Metadata fields
            "CreationDate": "CreationDate",
            "Creator": "Creator", 
            "EditDate": "EditDate",
            "Editor": "Editor",
            
            # Internal/computed fields
            "thematic_value": "thematic_value",
        }
        
        # Use request target if supplied, otherwise fall back to classifier suggestion, then model default
        target_variable = requested_target or query_classification.get('target_variable', 'MP30034A_B')
        original_target = target_variable  # Keep original for result naming
        
        # Step 1: Try to find the field in precalculated data with various naming patterns
        possible_column_names = []
        
        # First, check if we have a direct mapping in our comprehensive field_aliases
        if target_variable in field_aliases:
            possible_column_names.append(field_aliases[target_variable])
            logger.info(f"Found direct field mapping: {target_variable} -> {field_aliases[target_variable]}")
        
        # Add common naming patterns as fallbacks
        possible_column_names.extend([
            target_variable,                    # Exact match (e.g., "CONVERSION_RATE")
            f"value_{target_variable}",         # With value_ prefix (e.g., "value_MP30034A_B")
            f"value_2024 {target_variable}",    # With value_2024 prefix
            target_variable.upper(),            # Uppercase version
            target_variable.lower(),            # Lowercase version
        ])
        
        # Special handling for MP fields (athletic shoe data)
        if target_variable and target_variable.startswith('MP') and '_' in target_variable:
            value_target = f"value_{target_variable}"
            if value_target not in possible_column_names:
                possible_column_names.insert(1, value_target)  # High priority for MP fields
            logger.info(f"Added MP field pattern: {value_target}")
        
        # Special handling for demographic fields that might have different prefixes
        if any(keyword in target_variable.upper() for keyword in ['CY', 'POPULATION', 'INCOME', 'RACE']):
            demographic_patterns = [
                f"value_{target_variable}",
                f"value_2024 {target_variable}",
                f"{target_variable}_CY" if not target_variable.endswith('_CY') else target_variable,
            ]
            for pattern in demographic_patterns:
                if pattern not in possible_column_names:
                    possible_column_names.append(pattern)
        
        logger.info(f"Searching for target variable '{target_variable}' using patterns: {possible_column_names[:5]}...")  # Log first 5 patterns
        
        # Find the actual column name that exists in the data
        actual_target_column = None
        for candidate in possible_column_names:
            if candidate in precalc_df.columns:
                actual_target_column = candidate
                break
        
        if actual_target_column is None:
            # Enhanced error reporting with suggestions
            available_columns = list(precalc_df.columns)
            logger.error(f"Target variable '{target_variable}' not found in any expected format.")
            logger.error(f"Searched patterns: {possible_column_names}")
            logger.error(f"Total available columns: {len(available_columns)}")
            
            # Look for similar column names
            similar_columns = [col for col in available_columns 
                             if target_variable.lower() in col.lower() or 
                                any(pattern.lower() in col.lower() for pattern in possible_column_names[:3])]
            
            if similar_columns:
                logger.error(f"Similar columns found: {similar_columns[:10]}")
            
            # Show sample of available columns for debugging
            sample_columns = [col for col in available_columns if col.startswith('value_')][:20]
            logger.error(f"Sample 'value_' columns: {sample_columns}")
            
            # Fallback to model default with warning
            actual_target_column = model_info['target']
            logger.warning(f"Falling back to model default target: '{actual_target_column}'")
            
            # Return error if even the fallback doesn't exist
            if actual_target_column not in precalc_df.columns:
                logger.error(f"Even model default target '{actual_target_column}' not found in data!")
                return {"success": False, "error": f"Target variable '{target_variable}' and model default '{actual_target_column}' not found in dataset. Available columns: {len(available_columns)} total."}
        else:
            logger.info(f"Successfully mapped target variable '{target_variable}' to actual column: '{actual_target_column}'")
        
        # Use the actual column name for analysis
        target_variable = actual_target_column
        
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
            "analysis_type": analysis_type,  # Standardized field name for frontend consistency
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
    
    # Check both 'query_type' and 'analysis_type' keys for compatibility
    query_type = query_classification.get('query_type', query_classification.get('analysis_type', 'unknown'))
    target_variable = query_classification.get('target_variable', 'MP30034A_B')
    
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

def get_human_readable_field_name(field_code):
    """Convert field codes to human-readable names"""
    
    # Field aliases mapping from technical codes to human-readable names
    field_aliases = {
        # Brand fields
        "MP30029A_B": "Adidas Athletic Shoes",
        "MP30030A_B": "ASICS Athletic Shoes", 
        "MP30031A_B": "Converse Athletic Shoes",
        "MP30032A_B": "Jordan Athletic Shoes",
        "MP30033A_B": "New Balance Athletic Shoes",
        "MP30034A_B": "Nike Athletic Shoes",
        "MP30035A_B": "Puma Athletic Shoes",
        "MP30036A_B": "Reebok Athletic Shoes",
        "MP30037A_B": "Skechers Athletic Shoes",
        
        # Demographic fields
        "TOTPOP_CY": "Total Population",
        "MEDDI_CY": "Median Disposable Income",
        "DIVINDX_CY": "Diversity Index",
        "WHITE_CY": "White Population",
        "BLACK_CY": "Black Population",
        "ASIAN_CY": "Asian Population",
        "HISPWHT_CY": "Hispanic White Population",
        "GENZ_CY": "Gen Z Population",
        "MILLENN_CY": "Millennial Population",
        
        # Sports participation
        "MP33020A_B": "Jogging/Running Participation",
        "MP33032A_B": "Yoga Participation", 
        "MP33031A_B": "Weight Lifting Participation",
        
        # Sports fandom
        "MP33104A_B": "MLB Super Fan",
        "MP33105A_B": "NASCAR Super Fan",
        "MP33106A_B": "NBA Super Fan",
        "MP33107A_B": "NFL Super Fan",
        "MP33108A_B": "NHL Super Fan",
        "MP33119A_B": "International Soccer Super Fan",
        "MP33120A_B": "MLS Soccer Super Fan",
        
        # Store shopping
        "MP31035A_B": "Dick's Sporting Goods Shopping",
        "MP31042A_B": "Foot Locker Shopping",
    }
    
    # Handle correlation field patterns like "MP30034A_B_vs_MP30029A_B_correlation"
    if '_vs_' in field_code and field_code.endswith('_correlation'):
        parts = field_code.replace('_correlation', '').split('_vs_')
        if len(parts) == 2:
            field1_name = get_human_readable_field_name(parts[0])
            field2_name = get_human_readable_field_name(parts[1])
            return f"{field1_name} vs {field2_name}"
    
    # Handle value_ prefixed fields
    if field_code.startswith('value_'):
        base_field = field_code.replace('value_', '')
        if base_field in field_aliases:
            return field_aliases[base_field]
        field_code = base_field  # Continue with base field for other processing
    
    # Direct mapping
    if field_code in field_aliases:
        return field_aliases[field_code]
    
    # Fallback to prettified field name
    return field_code.replace('_', ' ').title()

def get_query_aware_top_areas(df, query_classification, target_variable, user_query):
    """Get top areas using query-aware ranking"""
    
    # Check both 'query_type' and 'analysis_type' keys for compatibility
    query_type = query_classification.get('query_type', query_classification.get('analysis_type', 'unknown'))
    
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
        
        # For income queries, weight by income levels
        if 'income' in user_query.lower():
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
                    'feature': get_human_readable_field_name(feature),  # Convert to human-readable name
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
    """Build results with query-specific fields"""
    results = []
    
    # Extract the original target name for clean field naming
    # This should be passed from the calling function, but we'll derive it here for now
    original_target = target_variable
    
    # Reverse lookup common mappings to get clean names (only for fields that actually exist)
    reverse_aliases = {
        "value_MEDDI_CY": "median_income",  # This field actually exists
        # Note: Other fields like conversion_rate, condo_ownership_pct don't exist in precalculated data
    }
    
    # Get clean field name for results
    if target_variable in reverse_aliases:
        clean_field_name = reverse_aliases[target_variable]
    elif target_variable.startswith('value_'):
        # For value_ prefixed fields, remove the prefix and convert to lowercase
        clean_field_name = target_variable.replace('value_', '').lower()
    else:
        # Use lowercase version of the field name
        clean_field_name = target_variable.lower()
    
    for _, row in df.iterrows():
        result = {
            'geo_id': str(row['ID']),
            'ZIP_CODE': str(row['ID']),
            'ID': str(row['ID'])
        }
        
        # Add DESCRIPTION field if it exists (contains ZIP + City like "10001 (New York)")
        if 'value_DESCRIPTION' in row and pd.notna(row['value_DESCRIPTION']):
            result['DESCRIPTION'] = str(row['value_DESCRIPTION'])
        
        # Add the target variable with both clean name and target_value
        target_val = safe_float(row[target_variable])
        result[clean_field_name] = target_val
        result['target_value'] = target_val
        
        # Add commonly useful fields if they exist in the data
        if 'TOTPOP_CY' in row:
            result['total_population'] = safe_float(row['TOTPOP_CY'])
        
        if 'value_MEDDI_CY' in row:
            result['median_income'] = safe_float(row['value_MEDDI_CY'])
        
        if 'value_WHITE_CY' in row:
            result['white_population'] = safe_float(row['value_WHITE_CY'])
        
        if 'value_ASIAN_CY' in row:
            result['asian_population'] = safe_float(row['value_ASIAN_CY'])
        
        if 'value_BLACK_CY' in row:
            result['black_population'] = safe_float(row['value_BLACK_CY'])
        
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
        "area_coverage": safe_float((unique_ids / total_areas) * 100),
        "regional_patterns": regional_patterns,
        "value_range": {
            "min": safe_float(values.min()),
            "max": safe_float(values.max()),
            "mean": safe_float(mean_value),
            "range_span": safe_float(value_range)
        },
        "top_areas": {
            "highest": [{"id": str(row['ID']), "value": safe_float(row[target_variable])} for _, row in top_5.iterrows()],
            "lowest": [{"id": str(row['ID']), "value": safe_float(row[target_variable])} for _, row in bottom_5.iterrows()]
        },
        "context_description": f"Analysis covers {unique_ids} areas with {geographic_scope} scope. "
                             f"Values range from {values.min():.2f} to {values.max():.2f} "
                             f"(mean: {mean_value:.2f})"
    }

def handle_bivariate_correlation(df, brand_fields, user_query, query_classification):
    """Handle bivariate correlation analysis for brand comparisons like Nike vs Adidas"""
    
    try:
        # Map field names to actual column names in the data
        field_aliases = {
            "MP30034A_B": "value_MP30034A_B",  # Nike Athletic Shoes
            "MP30029A_B": "value_MP30029A_B",  # Adidas Athletic Shoes
            "MP30032A_B": "value_MP30032A_B",  # Jordan Athletic Shoes
            "MP30031A_B": "value_MP30031A_B",  # Converse Athletic Shoes
            "MP30033A_B": "value_MP30033A_B",  # New Balance Athletic Shoes
            "MP30035A_B": "value_MP30035A_B",  # Puma Athletic Shoes
            "MP30036A_B": "value_MP30036A_B",  # Reebok Athletic Shoes
            "MP30037A_B": "value_MP30037A_B",  # Skechers Athletic Shoes
            "MP30016A_B": "value_MP30016A_B",  # Athletic Shoes (general)
        }
        
        # Get the actual column names
        field1_name = brand_fields[0]
        field2_name = brand_fields[1]
        
        col1 = field_aliases.get(field1_name, f"value_{field1_name}")
        col2 = field_aliases.get(field2_name, f"value_{field2_name}")
        
        logger.info(f"Bivariate correlation: {field1_name} ({col1}) vs {field2_name} ({col2})")
        
        # Check if columns exist in the data
        if col1 not in df.columns:
            logger.warning(f"Column {col1} not found in data. Available columns: {list(df.columns)}")
            return {"success": False, "error": f"Data not available for {field1_name}"}
        
        if col2 not in df.columns:
            logger.warning(f"Column {col2} not found in data. Available columns: {list(df.columns)}")
            return {"success": False, "error": f"Data not available for {field2_name}"}
        
        # Filter out rows with missing data
        valid_data = df[[col1, col2, 'ID']].dropna()
        
        if len(valid_data) < 3:
            logger.warning(f"Insufficient data for correlation: only {len(valid_data)} valid records")
            return {"success": False, "error": "Insufficient data for correlation analysis"}
        
        # Calculate correlation
        correlation_matrix = valid_data[[col1, col2]].corr()
        correlation_value = correlation_matrix.iloc[0, 1]
        
        logger.info(f"Calculated correlation: {correlation_value:.4f}")
        
        # Build results with both variables for each area
        results = []
        correlation_field_name = f'{field1_name}_vs_{field2_name}_correlation'
        
        for _, row in valid_data.iterrows():
            result = {
                'geo_id': str(row['ID']),
                'ZIP_CODE': str(row['ID']),
                'ID': str(row['ID']),
                'FSA_ID': str(row['ID']),  # Add FSA_ID for compatibility
                'primary_value': safe_float(row[col1]),      # First brand (e.g., Nike)
                'comparison_value': safe_float(row[col2]),   # Second brand (e.g., Adidas)
                'correlation_strength': safe_float(abs(correlation_value)),
                'correlation_score': safe_float(abs(correlation_value)),  # Alternative field name
                # Add the specific correlation field that frontend expects
                correlation_field_name: safe_float(abs(correlation_value)),
                # Add field names in multiple formats for frontend compatibility
                field1_name: safe_float(row[col1]),          # Original case (MP30034A_B)
                field2_name: safe_float(row[col2]),          # Original case (MP30029A_B)
                field1_name.lower(): safe_float(row[col1]),  # Lowercase (mp30034a_b)
                field2_name.lower(): safe_float(row[col2]),  # Lowercase (mp30029a_b)
                # Add snake_case versions for frontend compatibility
                field1_name.lower().replace('a_b', '_a_b'): safe_float(row[col1]),  # Snake case (mp30034_a_b)
                field2_name.lower().replace('a_b', '_a_b'): safe_float(row[col2]),  # Snake case (mp30029_a_b)
            }
            results.append(result)
        
        # Create feature importance showing the correlation
        feature_importance = [
            {
                'feature': f'{field1_name}_vs_{field2_name}_correlation',
                'importance': safe_float(abs(correlation_value)),
                'correlation': safe_float(correlation_value),
                'description': f'Correlation between {field1_name} and {field2_name}'
            }
        ]
        
        # Build summary
        brand1_name = get_brand_name_from_field(field1_name)
        brand2_name = get_brand_name_from_field(field2_name)
        
        if abs(correlation_value) > 0.7:
            strength = "strong"
        elif abs(correlation_value) > 0.3:
            strength = "moderate"
        else:
            strength = "weak"
        
        direction = "positive" if correlation_value > 0 else "negative"
        
        summary = f"Analysis shows a {strength} {direction} correlation ({correlation_value:.3f}) between {brand1_name} and {brand2_name} athletic shoe purchases across regions."
        
        # Add geographic context
        top_areas = valid_data.nlargest(5, col1)
        if len(top_areas) > 0:
            summary += f" Areas with high {brand1_name} purchases also tend to have {'high' if correlation_value > 0 else 'low'} {brand2_name} purchases."
        
        return {
            "success": True,
            "results": results,
            "summary": summary,
            "feature_importance": feature_importance,
            "correlation_analysis": {
                "correlation_coefficient": safe_float(correlation_value),
                "correlation_strength": strength,
                "correlation_direction": direction,
                "sample_size": len(valid_data),
                "field1": field1_name,
                "field2": field2_name,
                "brand1": brand1_name,
                "brand2": brand2_name
            },
            "analysis_type": "bivariate_correlation"
        }
        
    except Exception as e:
        logger.error(f"Error in bivariate correlation analysis: {str(e)}")
        return {"success": False, "error": f"Bivariate correlation analysis failed: {str(e)}"}

def get_brand_name_from_field(field_code):
    """Convert field codes to human-readable brand names"""
    brand_mapping = {
        "MP30034A_B": "Nike",
        "MP30029A_B": "Adidas", 
        "MP30032A_B": "Jordan",
        "MP30031A_B": "Converse",
        "MP30033A_B": "New Balance",
        "MP30035A_B": "Puma",
        "MP30036A_B": "Reebok",
        "MP30037A_B": "Skechers",
        "MP30016A_B": "Athletic Shoes"
    }
    return brand_mapping.get(field_code, field_code)

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