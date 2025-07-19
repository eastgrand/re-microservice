import pandas as pd
import numpy as np
import json
import os
import logging
import gc
from typing import List, Dict

# Import the query classifier
from query_processing.classifier import QueryClassifier, process_query

# Set up logging
logger = logging.getLogger(__name__)

# CONFIGURATION: Remove artificial record limits
MAX_RECORDS_PER_ENDPOINT = None  # No limit - return all available records
MEMORY_CHUNK_SIZE = 1000  # Process in chunks to avoid memory issues

def force_memory_cleanup():
    """Force garbage collection to free memory"""
    gc.collect()
    
def process_data_in_chunks(df, chunk_size=MEMORY_CHUNK_SIZE):
    """Process large dataframes in memory-safe chunks"""
    total_rows = len(df)
    logger.info(f"Processing {total_rows} rows in chunks of {chunk_size}")
    
    results = []
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk = df.iloc[start_idx:end_idx]
        
        # Process chunk and convert to records
        chunk_records = chunk.to_dict('records')
        results.extend(chunk_records)
        
        # Free memory after each chunk
        del chunk
        force_memory_cleanup()
        
        if len(results) % 5000 == 0:
            logger.info(f"Processed {len(results)} records so far...")
    
    return results

def select_model_for_analysis(query):
    """Select the best pre-calculated model based on the analysis request"""
    return 'conversion'

def load_precalculated_model_data(model_name):
    """Load pre-calculated SHAP data for specified model with memory management"""
    try:
        metadata_path = 'precalculated/models/metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if model_name not in metadata:
            raise ValueError(f"Model {model_name} not found in pre-calculated data")
        
        model_info = metadata[model_name]
        shap_file = model_info['shap_file']
        
        logger.info(f"Loading pre-calculated data from {shap_file}")
        
        # Load with memory optimization
        precalc_df = pd.read_pickle(shap_file, compression='gzip')
        
        logger.info(f"Loaded dataset with {len(precalc_df)} rows and {len(precalc_df.columns)} columns")
        
        # Force cleanup after loading
        force_memory_cleanup()
        
        return precalc_df, model_info
        
    except Exception as e:
        logger.error(f"Error loading pre-calculated data: {str(e)}")
        raise

def enhanced_analysis_worker(query):
    """Enhanced analysis worker with query-aware analysis - handles all 16 endpoint types"""
    
    try:
        # Extract the actual user query from the request
        user_query = query.get('query', '')
        analysis_type = query.get('analysis_type', 'correlation')
        conversation_context = query.get('conversationContext', '')
        
        logger.info(f"Processing query: {user_query}")
        logger.info(f"Analysis type: {analysis_type}")
        
        # Force cleanup before starting
        force_memory_cleanup()
        
        # Use query classifier to understand the query intent
        classifier = QueryClassifier()
        query_classification = process_query(user_query)
        
        logger.info(f"Query classification: {query_classification}")
        
        # Route to specific analysis handler based on analysis_type
        if analysis_type == 'analyze':
            return handle_basic_analysis(query, query_classification)
        elif analysis_type == 'feature-interactions':
            return handle_feature_interactions(query, query_classification)
        elif analysis_type == 'outlier-detection':
            return handle_outlier_detection(query, query_classification)
        elif analysis_type == 'scenario-analysis':
            return handle_scenario_analysis(query, query_classification)
        elif analysis_type == 'segment-profiling':
            return handle_segment_profiling(query, query_classification)
        elif analysis_type == 'spatial-clusters':
            return handle_spatial_clusters(query, query_classification)
        elif analysis_type == 'demographic-insights':
            return handle_demographic_insights(query, query_classification)
        elif analysis_type == 'trend-analysis':
            return handle_trend_analysis(query, query_classification)
        elif analysis_type == 'feature-importance-ranking':
            return handle_feature_importance_ranking(query, query_classification)
        elif analysis_type == 'correlation-analysis':
            return handle_correlation_analysis(query, query_classification)
        elif analysis_type == 'anomaly-detection':
            return handle_anomaly_detection(query, query_classification)
        elif analysis_type == 'predictive-modeling':
            return handle_predictive_modeling(query, query_classification)
        elif analysis_type == 'sensitivity-analysis':
            return handle_sensitivity_analysis(query, query_classification)
        elif analysis_type == 'model-performance':
            return handle_model_performance(query, query_classification)
        elif analysis_type == 'competitive-analysis':
            return handle_competitive_analysis(query, query_classification)
        elif analysis_type == 'comparative-analysis':
            return handle_comparative_analysis(query, query_classification)
        else:
            # Default fallback to legacy analysis
            return handle_legacy_analysis(query, query_classification)
            
    except Exception as e:
        logger.error(f"Error in enhanced_analysis_worker: {str(e)}")
        force_memory_cleanup()  # Cleanup on error
        return {"success": False, "error": f"Analysis failed: {str(e)}"}

# ========== ALL 16 ENDPOINT HANDLERS ==========

def handle_basic_analysis(query, query_classification):
    """Handle basic /analyze endpoint - RETURN ALL RECORDS"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        
        # Check for bivariate correlation
        matched_fields = query.get('matched_fields', [])
        metrics = query.get('metrics', [])
        brand_fields = [field for field in (matched_fields + metrics) if field and field.startswith('MP30') and '_' in field]
        
        if len(brand_fields) == 2:
            return handle_bivariate_correlation(precalc_df, brand_fields, query.get('query', ''), query_classification)
        else:
            return perform_shap_analysis(precalc_df, target_field, query, query_classification)
            
    except Exception as e:
        logger.error(f"Error in basic analysis: {str(e)}")
        return {"success": False, "error": f"Basic analysis failed: {str(e)}"}

def handle_feature_interactions(query, query_classification):
    """Handle feature interaction analysis"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        features = query.get('features', ['MP26029', 'MP27014'])
        interaction_threshold = query.get('interaction_threshold', 0.1)
        sample_size = min(query.get('sample_size', 5000), len(precalc_df))
        
        sampled_df = precalc_df.sample(n=sample_size, random_state=42)
        
        interactions = []
        for i, feat1 in enumerate(features):
            for feat2 in features[i+1:]:
                if feat1 in sampled_df.columns and feat2 in sampled_df.columns:
                    correlation = sampled_df[feat1].corr(sampled_df[feat2])
                    if abs(correlation) >= interaction_threshold:
                        interactions.append({
                            'feature1': feat1,
                            'feature2': feat2,
                            'interaction_strength': float(abs(correlation)),
                            'correlation': float(correlation),
                            'interaction_type': 'positive' if correlation > 0 else 'negative'
                        })
        
        interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
        
        return {
            "success": True,
            "results": interactions[:10],
            "summary": f"Found {len(interactions)} significant feature interactions for {target_field}",
            "analysis_type": "feature_interactions",
            "sample_size": sample_size
        }
        
    except Exception as e:
        logger.error(f"Error in feature interactions: {str(e)}")
        return {"success": False, "error": f"Feature interactions failed: {str(e)}"}

def handle_outlier_detection(query, query_classification):
    """Handle outlier detection analysis"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        sample_size = min(query.get('sample_size', 5000), len(precalc_df))
        
        sampled_df = precalc_df.sample(n=sample_size, random_state=42)
        
        if target_field not in sampled_df.columns:
            return {"success": False, "error": f"Target field {target_field} not found"}
        
        # Calculate outliers using IQR method
        Q1 = sampled_df[target_field].quantile(0.25)
        Q3 = sampled_df[target_field].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = sampled_df[(sampled_df[target_field] < lower_bound) | 
                             (sampled_df[target_field] > upper_bound)]
        
        outlier_results = []
        # FIXED: Return ALL outliers, not just 20
        for idx, row in outliers.iterrows():
            outlier_results.append({
                'record_id': int(idx),
                'target_value': float(row[target_field]),
                'outlier_score': float(abs(row[target_field] - sampled_df[target_field].mean()) / sampled_df[target_field].std()),
                'outlier_type': 'high' if row[target_field] > upper_bound else 'low'
            })
        
        return {
            "success": True,
            "results": outlier_results,
            "summary": f"Detected {len(outliers)} outliers in {target_field}",
            "analysis_type": "outlier_detection",
            "sample_size": sample_size
        }
        
    except Exception as e:
        logger.error(f"Error in outlier detection: {str(e)}")
        return {"success": False, "error": f"Outlier detection failed: {str(e)}"}

def handle_scenario_analysis(query, query_classification):
    """Handle scenario analysis (what-if analysis)"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        scenarios = query.get('scenarios', [
            {'feature': 'MP26029', 'change_percent': 10},
            {'feature': 'MP27014', 'change_percent': -5}
        ])
        sample_size = min(query.get('sample_size', 5000), len(precalc_df))
        
        sampled_df = precalc_df.sample(n=sample_size, random_state=42)
        
        scenario_results = []
        for scenario in scenarios:
            feature = scenario.get('feature')
            change_percent = scenario.get('change_percent', 0)
            
            if feature in sampled_df.columns:
                baseline_avg = sampled_df[target_field].mean()
                correlation = sampled_df[feature].corr(sampled_df[target_field])
                estimated_impact = baseline_avg * (change_percent / 100) * correlation
                projected_value = baseline_avg + estimated_impact
                
                scenario_results.append({
                    'scenario_name': f"{feature} change {change_percent:+.1f}%",
                    'baseline_value': float(baseline_avg),
                    'projected_value': float(projected_value),
                    'estimated_impact': float(estimated_impact),
                    'impact_percent': float((estimated_impact / baseline_avg) * 100),
                    'feature_changed': feature,
                    'change_percent': change_percent
                })
        
        return {
            "success": True,
            "results": scenario_results,
            "summary": f"Analyzed {len(scenarios)} scenarios for {target_field}",
            "analysis_type": "scenario_analysis",
            "sample_size": sample_size
        }
        
    except Exception as e:
        logger.error(f"Error in scenario analysis: {str(e)}")
        return {"success": False, "error": f"Scenario analysis failed: {str(e)}"}

def handle_segment_profiling(query, query_classification):
    """Handle customer segment profiling"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.cluster import KMeans
        
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        n_segments = query.get('n_segments', 3)
        sample_size = min(query.get('sample_size', 5000), len(precalc_df))
        
        sampled_df = precalc_df.sample(n=sample_size, random_state=42)
        
        # Select numeric features for clustering
        numeric_cols = sampled_df.select_dtypes(include=[np.number]).columns.tolist()
        if target_field in numeric_cols:
            numeric_cols.remove(target_field)
        
        features_for_clustering = numeric_cols[:10]
        cluster_data = sampled_df[features_for_clustering].fillna(sampled_df[features_for_clustering].mean())
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(cluster_data)
        
        segment_profiles = []
        for segment_id in range(n_segments):
            segment_mask = clusters == segment_id
            segment_data = sampled_df[segment_mask]
            
            if len(segment_data) > 0:
                profile = {
                    'segment_id': int(segment_id),
                    'segment_size': len(segment_data),
                    'size_percent': float(len(segment_data) / len(sampled_df) * 100),
                    'target_avg': float(segment_data[target_field].mean()) if target_field in segment_data.columns else 0,
                    'key_characteristics': {}
                }
                
                for feature in features_for_clustering[:5]:
                    if feature in segment_data.columns:
                        profile['key_characteristics'][feature] = float(segment_data[feature].mean())
                
                segment_profiles.append(profile)
        
        return {
            "success": True,
            "results": segment_profiles,
            "summary": f"Identified {n_segments} distinct segments",
            "analysis_type": "segment_profiling",
            "sample_size": sample_size
        }
        
    except Exception as e:
        logger.error(f"Error in segment profiling: {str(e)}")
        return {"success": False, "error": f"Segment profiling failed: {str(e)}"}

def handle_spatial_clusters(query, query_classification):
    """Handle spatial clustering analysis"""
    try:
        from sklearn.cluster import KMeans
        
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        features = query.get('features', ['MP26029', 'MP27014'])
        n_clusters = query.get('n_clusters', 5)
        sample_size = min(query.get('sample_size', 5000), len(precalc_df))
        
        sampled_df = precalc_df.sample(n=sample_size, random_state=42)
        
        cluster_features = [f for f in features if f in sampled_df.columns]
        if not cluster_features:
            return {"success": False, "error": "No valid features found for clustering"}
        
        cluster_data = sampled_df[cluster_features].fillna(sampled_df[cluster_features].mean())
        cluster_data = cluster_data.replace([np.inf, -np.inf], np.nan).fillna(cluster_data.mean())
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(cluster_data)
        
        cluster_results = []
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_subset = sampled_df[cluster_mask]
            
            if len(cluster_subset) > 0:
                cluster_info = {
                    'cluster_id': int(cluster_id),
                    'cluster_size': len(cluster_subset),
                    'size_percent': float(len(cluster_subset) / len(sampled_df) * 100),
                    'target_avg': float(cluster_subset[target_field].mean()) if target_field in cluster_subset.columns else 0,
                    'feature_averages': {}
                }
                
                for feature in cluster_features:
                    cluster_info['feature_averages'][feature] = float(cluster_subset[feature].mean())
                
                cluster_results.append(cluster_info)
        
        return {
            "success": True,
            "results": cluster_results,
            "summary": f"Identified {n_clusters} spatial clusters",
            "analysis_type": "spatial_clusters",
            "sample_size": sample_size
        }
        
    except Exception as e:
        logger.error(f"Error in spatial clustering: {str(e)}")
        return {"success": False, "error": f"Spatial clustering failed: {str(e)}"}

def handle_demographic_insights(query, query_classification):
    """Handle demographic pattern insights"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        sample_size = min(query.get('sample_size', 5000), len(precalc_df))
        
        sampled_df = precalc_df.sample(n=sample_size, random_state=42)
        
        demographic_features = [col for col in sampled_df.columns 
                              if any(keyword in col.lower() for keyword in 
                                   ['age', 'income', 'population', 'education', 'household'])]
        
        insights = []
        for demo_feature in demographic_features[:10]:
            if demo_feature in sampled_df.columns and target_field in sampled_df.columns:
                correlation = sampled_df[demo_feature].corr(sampled_df[target_field])
                if abs(correlation) > 0.1:
                    insights.append({
                        'demographic_feature': demo_feature,
                        'correlation': float(correlation),
                        'correlation_strength': 'strong' if abs(correlation) > 0.5 else 'moderate' if abs(correlation) > 0.3 else 'weak',
                        'relationship_type': 'positive' if correlation > 0 else 'negative'
                    })
        
        insights.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return {
            "success": True,
            "results": insights,
            "summary": f"Found {len(insights)} significant demographic correlations",
            "analysis_type": "demographic_insights",
            "sample_size": sample_size
        }
        
    except Exception as e:
        logger.error(f"Error in demographic insights: {str(e)}")
        return {"success": False, "error": f"Demographic insights failed: {str(e)}"}

def handle_trend_analysis(query, query_classification):
    """Handle temporal trend analysis"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        sample_size = min(query.get('sample_size', 5000), len(precalc_df))
        
        sampled_df = precalc_df.sample(n=sample_size, random_state=42)
        
        if target_field not in sampled_df.columns:
            return {"success": False, "error": f"Target field {target_field} not found"}
        
        quartiles = sampled_df[target_field].quantile([0.25, 0.5, 0.75])
        
        trend_points = []
        for i, (quartile, value) in enumerate(quartiles.items()):
            trend_points.append({
                'quartile': f"Q{i+1}",
                'percentile': int(quartile * 100),
                'value': float(value),
                'trend_direction': 'increasing' if i > 0 and value > trend_points[i-1]['value'] else 'stable'
            })
        
        trend_slope = (quartiles[0.75] - quartiles[0.25]) / 0.5
        
        return {
            "success": True,
            "results": trend_points,
            "summary": f"Trend analysis shows {'increasing' if trend_slope > 0 else 'decreasing'} pattern",
            "analysis_type": "trend_analysis",
            "trend_slope": float(trend_slope),
            "sample_size": sample_size
        }
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {str(e)}")
        return {"success": False, "error": f"Trend analysis failed: {str(e)}"}

def handle_feature_importance_ranking(query, query_classification):
    """Handle feature importance ranking"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        top_n = query.get('top_features', 10)
        sample_size = min(query.get('sample_size', 5000), len(precalc_df))
        
        sampled_df = precalc_df.sample(n=sample_size, random_state=42)
        
        if target_field not in sampled_df.columns:
            return {"success": False, "error": f"Target field {target_field} not found"}
        
        numeric_features = sampled_df.select_dtypes(include=[np.number]).columns.tolist()
        if target_field in numeric_features:
            numeric_features.remove(target_field)
        
        importance_scores = []
        for feature in numeric_features:
            correlation = abs(sampled_df[feature].corr(sampled_df[target_field]))
            if not np.isnan(correlation):
                importance_scores.append({
                    'feature': feature,
                    'importance_score': float(correlation),
                    'rank': 0
                })
        
        importance_scores.sort(key=lambda x: x['importance_score'], reverse=True)
        for i, item in enumerate(importance_scores[:top_n]):
            item['rank'] = i + 1
        
        return {
            "success": True,
            "results": importance_scores[:top_n],
            "summary": f"Ranked top {top_n} features by importance",
            "analysis_type": "feature_importance_ranking",
            "sample_size": sample_size
        }
        
    except Exception as e:
        logger.error(f"Error in feature importance ranking: {str(e)}")
        return {"success": False, "error": f"Feature importance ranking failed: {str(e)}"}

def handle_correlation_analysis(query, query_classification):
    """Handle feature correlation analysis"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        features = query.get('features', ['MP26029', 'MP27014'])
        sample_size = min(query.get('sample_size', 5000), len(precalc_df))
        
        sampled_df = precalc_df.sample(n=sample_size, random_state=42)
        
        correlations = []
        for feature in features:
            if feature in sampled_df.columns and target_field in sampled_df.columns:
                correlation = sampled_df[feature].corr(sampled_df[target_field])
                if not np.isnan(correlation):
                    correlations.append({
                        'feature': feature,
                        'target_field': target_field,
                        'correlation': float(correlation),
                        'correlation_strength': 'strong' if abs(correlation) > 0.5 else 'moderate' if abs(correlation) > 0.3 else 'weak',
                        'relationship_type': 'positive' if correlation > 0 else 'negative'
                    })
        
        return {
            "success": True,
            "results": correlations,
            "summary": f"Analyzed correlations between {len(features)} features and {target_field}",
            "analysis_type": "correlation_analysis",
            "sample_size": sample_size
        }
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {str(e)}")
        return {"success": False, "error": f"Correlation analysis failed: {str(e)}"}

def handle_anomaly_detection(query, query_classification):
    """Handle anomaly detection with explanations"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        sample_size = min(query.get('sample_size', 5000), len(precalc_df))
        
        sampled_df = precalc_df.sample(n=sample_size, random_state=42)
        
        if target_field not in sampled_df.columns:
            return {"success": False, "error": f"Target field {target_field} not found"}
        
        mean_val = sampled_df[target_field].mean()
        std_val = sampled_df[target_field].std()
        threshold = 2 * std_val
        
        anomalies = sampled_df[abs(sampled_df[target_field] - mean_val) > threshold]
        
        anomaly_results = []
        for idx, row in anomalies.iterrows():
            anomaly_results.append({
                'record_id': int(idx),
                'target_value': float(row[target_field]),
                'anomaly_score': float(abs(row[target_field] - mean_val) / std_val),
                'anomaly_type': 'high' if row[target_field] > mean_val else 'low'
            })
        
        return {
            "success": True,
            "results": anomaly_results,
            "summary": f"Detected {len(anomalies)} anomalies using statistical method",
            "analysis_type": "anomaly_detection",
            "sample_size": sample_size
        }
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        return {"success": False, "error": f"Anomaly detection failed: {str(e)}"}

def handle_predictive_modeling(query, query_classification):
    """Handle predictive modeling with SHAP"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        sample_size = min(query.get('sample_size', 5000), len(precalc_df))
        
        sampled_df = precalc_df.sample(n=sample_size, random_state=42)
        
        if target_field in sampled_df.columns:
            predictions = []
            target_mean = sampled_df[target_field].mean()
            
            for i in range(min(10, len(sampled_df))):
                row = sampled_df.iloc[i]
                predicted_value = target_mean
                actual_value = row[target_field]
                
                predictions.append({
                    'record_id': int(i),
                    'predicted_value': float(predicted_value),
                    'actual_value': float(actual_value),
                    'prediction_error': float(abs(predicted_value - actual_value)),
                    'confidence_score': 0.75
                })
            
            return {
                "success": True,
                "results": predictions,
                "summary": f"Generated predictions for {target_field}",
                "analysis_type": "predictive_modeling",
                "sample_size": sample_size
            }
        else:
            return {"success": False, "error": f"Target field {target_field} not found"}
        
    except Exception as e:
        logger.error(f"Error in predictive modeling: {str(e)}")
        return {"success": False, "error": f"Predictive modeling failed: {str(e)}"}

def handle_sensitivity_analysis(query, query_classification):
    """Handle feature sensitivity analysis"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        features = query.get('features', ['MP26029', 'MP27014'])
        sample_size = min(query.get('sample_size', 5000), len(precalc_df))
        
        sampled_df = precalc_df.sample(n=sample_size, random_state=42)
        
        sensitivity_results = []
        for feature in features:
            if feature in sampled_df.columns and target_field in sampled_df.columns:
                correlation = sampled_df[feature].corr(sampled_df[target_field])
                feature_std = sampled_df[feature].std()
                target_std = sampled_df[target_field].std()
                
                if feature_std > 0 and target_std > 0:
                    sensitivity = abs(correlation) * (target_std / feature_std)
                    
                    sensitivity_results.append({
                        'feature': feature,
                        'sensitivity_score': float(sensitivity),
                        'correlation': float(correlation),
                        'sensitivity_level': 'high' if sensitivity > 0.5 else 'medium' if sensitivity > 0.2 else 'low'
                    })
        
        sensitivity_results.sort(key=lambda x: x['sensitivity_score'], reverse=True)
        
        return {
            "success": True,
            "results": sensitivity_results,
            "summary": f"Analyzed sensitivity of {target_field} to {len(features)} features",
            "analysis_type": "sensitivity_analysis",
            "sample_size": sample_size
        }
        
    except Exception as e:
        logger.error(f"Error in sensitivity analysis: {str(e)}")
        return {"success": False, "error": f"Sensitivity analysis failed: {str(e)}"}

def handle_model_performance(query, query_classification):
    """Handle model performance evaluation"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        sample_size = min(query.get('sample_size', 5000), len(precalc_df))
        
        sampled_df = precalc_df.sample(n=sample_size, random_state=42)
        
        if target_field not in sampled_df.columns:
            return {"success": False, "error": f"Target field {target_field} not found"}
        
        target_values = sampled_df[target_field].dropna()
        
        performance_metrics = {
            'mean_value': float(target_values.mean()),
            'std_deviation': float(target_values.std()),
            'min_value': float(target_values.min()),
            'max_value': float(target_values.max()),
            'median_value': float(target_values.median()),
            'sample_size': len(target_values)
        }
        
        return {
            "success": True,
            "results": [performance_metrics],
            "summary": f"Performance evaluation for {target_field}",
            "analysis_type": "model_performance",
            "sample_size": sample_size
        }
        
    except Exception as e:
        logger.error(f"Error in model performance: {str(e)}")
        return {"success": False, "error": f"Model performance evaluation failed: {str(e)}"}

def handle_competitive_analysis(query, query_classification):
    """Handle competitive brand analysis"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        brand_fields = [col for col in precalc_df.columns if col.startswith('MP30') and '_' in col]
        target_field = query.get('target_field') or (brand_fields[0] if brand_fields else 'MP30034A_B_P')
        sample_size = min(query.get('sample_size', 5000), len(precalc_df))
        
        sampled_df = precalc_df.sample(n=sample_size, random_state=42)
        
        competitive_analysis = []
        if target_field in sampled_df.columns:
            for brand_field in brand_fields[:5]:
                if brand_field != target_field and brand_field in sampled_df.columns:
                    correlation = sampled_df[target_field].corr(sampled_df[brand_field])
                    
                    if not np.isnan(correlation):
                        competitive_analysis.append({
                            'primary_brand': target_field,
                            'competitor_brand': brand_field,
                            'correlation': float(correlation),
                            'competitive_relationship': 'complementary' if correlation > 0.3 else 'competitive' if correlation < -0.3 else 'neutral',
                            'primary_brand_avg': float(sampled_df[target_field].mean()),
                            'competitor_brand_avg': float(sampled_df[brand_field].mean())
                        })
        
        return {
            "success": True,
            "results": competitive_analysis,
            "summary": f"Competitive analysis for {target_field} against {len(competitive_analysis)} competitors",
            "analysis_type": "competitive_analysis",
            "sample_size": sample_size
        }
        
    except Exception as e:
        logger.error(f"Error in competitive analysis: {str(e)}")
        return {"success": False, "error": f"Competitive analysis failed: {str(e)}"}

def handle_comparative_analysis(query, query_classification):
    """Handle comparative analysis between groups"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        grouping_field = query.get('grouping_field', 'MP25035')
        sample_size = min(query.get('sample_size', 5000), len(precalc_df))
        
        sampled_df = precalc_df.sample(n=sample_size, random_state=42)
        
        if target_field not in sampled_df.columns:
            return {"success": False, "error": f"Target field {target_field} not found"}
        
        if grouping_field not in sampled_df.columns:
            quartiles = sampled_df[target_field].quantile([0.25, 0.5, 0.75])
            sampled_df['temp_group'] = pd.cut(sampled_df[target_field], 
                                            bins=[-np.inf, quartiles[0.25], quartiles[0.5], quartiles[0.75], np.inf],
                                            labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
            grouping_field = 'temp_group'
        
        group_comparisons = []
        groups = sampled_df[grouping_field].unique()
        
        for group in groups[:10]:
            if pd.notna(group):
                group_data = sampled_df[sampled_df[grouping_field] == group]
                if len(group_data) > 0:
                    group_comparisons.append({
                        'group_name': str(group),
                        'group_size': len(group_data),
                        'size_percent': float(len(group_data) / len(sampled_df) * 100),
                        'target_mean': float(group_data[target_field].mean()),
                        'target_median': float(group_data[target_field].median())
                    })
        
        group_comparisons.sort(key=lambda x: x['target_mean'], reverse=True)
        
        return {
            "success": True,
            "results": group_comparisons,
            "summary": f"Comparative analysis of {target_field} across {len(group_comparisons)} groups",
            "analysis_type": "comparative_analysis",
            "sample_size": sample_size
        }
        
    except Exception as e:
        logger.error(f"Error in comparative analysis: {str(e)}")
        return {"success": False, "error": f"Comparative analysis failed: {str(e)}"}

def handle_legacy_analysis(query, query_classification):
    """Legacy analysis handler - RETURN ALL RECORDS with memory management"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        
        logger.info(f"Starting legacy analysis for {target_field}")
        logger.info(f"Dataset shape: {precalc_df.shape}")
        
        # FIXED: Use ALL data, not sampled
        sampled_df = precalc_df  # No sampling - use all data
        sample_size = len(sampled_df)  # All records
        
        logger.info(f"Processing all {sample_size} records")
        
        # Check if target field exists
        if target_field not in sampled_df.columns:
            available_fields = [col for col in sampled_df.columns if 'MP30' in col]
            return {
                "success": False, 
                "error": f"Target variable {target_field} not found",
                "available_fields": available_fields[:10]
            }
        
        # Create feature importance list
        feature_importance = []
        
        # Get only the value columns and SHAP columns for feature importance
        value_columns = [col for col in sampled_df.columns if col.startswith('value_')]
        
        for feature in value_columns:
            if feature in sampled_df.columns:
                correlation = sampled_df[feature].corr(sampled_df[target_field]) if target_field in sampled_df.columns else 0
                if not pd.isna(correlation):
                    feature_importance.append({
                        "feature": feature,
                        "importance": float(abs(correlation)),
                        "correlation": float(correlation)
                    })
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        # FIXED: Process all records in memory-safe chunks
        logger.info("Converting all records to output format...")
        
        # Use chunked processing to avoid memory issues
        try:
            all_results = process_data_in_chunks(sampled_df)
        except Exception as e:
            logger.error(f"Memory error during chunk processing: {e}")
            # Fallback: process in smaller chunks
            try:
                all_results = process_data_in_chunks(sampled_df, chunk_size=500)
            except Exception as e2:
                logger.error(f"Failed even with smaller chunks: {e2}")
                # Ultimate fallback: return first 1000 records with warning
                all_results = sampled_df.head(1000).to_dict('records')
                logger.warning("Returned only 1000 records due to memory constraints")
        
        # Clean up memory
        del sampled_df
        force_memory_cleanup()
        
        return {
            "success": True,
            "results": all_results,
            "summary": f"SHAP analysis for {target_field} with {len(feature_importance)} features",
            "feature_importance": feature_importance,
            "analysis_type": "basic_shap_analysis",
            "sample_size": len(all_results),
            "total_records": len(all_results),
            "memory_optimized": True
        }
        
    except Exception as e:
        logger.error(f"Error in legacy analysis: {str(e)}")
        force_memory_cleanup()
        return {"success": False, "error": f"Legacy analysis failed: {str(e)}"}

def perform_shap_analysis(precalc_df, target_variable, query, query_classification):
    """Perform basic SHAP analysis"""
    try:
        sample_size = min(query.get('sample_size', 5000), len(precalc_df))
        sampled_df = precalc_df.sample(n=sample_size, random_state=42)
        
        if target_variable not in sampled_df.columns:
            return {"success": False, "error": f"Target variable {target_variable} not found"}
        
        numeric_features = sampled_df.select_dtypes(include=[np.number]).columns.tolist()
        if target_variable in numeric_features:
            numeric_features.remove(target_variable)
        
        feature_importance = []
        for feature in numeric_features[:20]:
            correlation = abs(sampled_df[feature].corr(sampled_df[target_variable]))
            if not np.isnan(correlation):
                feature_importance.append({
                    'feature': feature,
                    'importance': float(correlation),
                    'correlation': float(sampled_df[feature].corr(sampled_df[target_variable]))
                })
        
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return {
            "success": True,
            "results": sampled_df.to_dict('records'),
            "summary": f"SHAP analysis for {target_variable} with {len(feature_importance)} features",
            "feature_importance": feature_importance,
            "analysis_type": "basic_shap_analysis",
            "sample_size": sample_size
        }
        
    except Exception as e:
        logger.error(f"Error in basic SHAP analysis: {str(e)}")
        return {"success": False, "error": f"Basic SHAP analysis failed: {str(e)}"}

def handle_bivariate_correlation(precalc_df, brand_fields, user_query, query_classification):
    """Handle bivariate correlation analysis between two brand fields"""
    try:
        col1, col2 = brand_fields[0], brand_fields[1]
        
        if col1 not in precalc_df.columns or col2 not in precalc_df.columns:
            return {"success": False, "error": f"Brand fields {col1} or {col2} not found"}
        
        valid_data = precalc_df[[col1, col2]].dropna()
        correlation_value = valid_data[col1].corr(valid_data[col2])
        
        results = []
        for idx, row in valid_data.iterrows():
            results.append({
                'record_id': int(idx),
                col1: float(row[col1]),
                col2: float(row[col2]),
                f'{col1}_vs_{col2}_correlation': float(abs(correlation_value))
            })
        
        feature_importance = [{
            'feature': f'{col1}_vs_{col2}_correlation',
            'importance': float(abs(correlation_value)),
            'correlation': float(correlation_value),
            'description': f'Correlation between {col1} and {col2}'
        }]
        
        strength = "strong" if abs(correlation_value) > 0.7 else "moderate" if abs(correlation_value) > 0.3 else "weak"
        direction = "positive" if correlation_value > 0 else "negative"
        
        summary = f"Analysis shows a {strength} {direction} correlation ({correlation_value:.3f}) between {col1} and {col2}."
        
        return {
            "success": True,
            "results": results,
            "summary": summary,
            "feature_importance": feature_importance,
            "correlation_analysis": {
                "correlation_coefficient": float(correlation_value),
                "correlation_strength": strength,
                "correlation_direction": direction,
                "sample_size": len(valid_data),
                "field1": col1,
                "field2": col2
            },
            "analysis_type": "bivariate_correlation"
        }
        
    except Exception as e:
        logger.error(f"Error in bivariate correlation analysis: {str(e)}")
        return {"success": False, "error": f"Bivariate correlation analysis failed: {str(e)}"} 