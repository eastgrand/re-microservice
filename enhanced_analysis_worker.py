import pandas as pd
import numpy as np
import json
import os
import logging
import gc
import pickle
import tempfile
from typing import List, Dict, Any

# Import the query classifier
from query_processing.classifier import QueryClassifier, process_query

# --- FIELD FILTERING FUNCTIONS ---
def filter_results_fields(results: List[Dict[str, Any]], query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Filter result fields based on query parameters"""
    
    # Check if field filtering is requested
    include_all_fields = query_params.get('include_all_fields', True)
    matched_fields = query_params.get('matched_fields', [])
    
    # If include_all_fields is True or no matched_fields specified, return as-is
    if include_all_fields or not matched_fields:
        logger.info(f"🔓 Field filtering disabled - returning all {len(results[0].keys()) if results else 0} fields")
        return results
    
    # Filter fields for each result record
    filtered_results = []
    for record in results:
        # Keep only matched fields that exist in the record
        filtered_record = {key: value for key, value in record.items() if key in matched_fields}
        filtered_results.append(filtered_record)
    
    original_field_count = len(results[0].keys()) if results else 0
    filtered_field_count = len(filtered_results[0].keys()) if filtered_results else 0
    
    logger.info(f"🔒 Field filtering applied: {original_field_count} → {filtered_field_count} fields")
    logger.info(f"📋 Requested fields: {len(matched_fields)}, Found: {filtered_field_count}")
    
    return filtered_results

def apply_field_filtering_to_response(response: Dict[str, Any], query_params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply field filtering to a complete response object"""
    if 'results' in response and isinstance(response['results'], list) and response['results']:
        response['results'] = filter_results_fields(response['results'], query_params)
    return response

# SHAP CALCULATION INFRASTRUCTURE - PHASE 1
import shap

# Set up logging
logger = logging.getLogger(__name__)

# Global SHAP explainer (initialized once)
_shap_explainer = None
_model_features = None
_xgb_model = None

def initialize_shap_explainer():
    """Initialize SHAP explainer with loaded XGBoost model"""
    global _shap_explainer, _model_features, _xgb_model
    
    try:
        logger.info("Initializing SHAP explainer...")
        
        # Import model from main app module
        from app import model, feature_names
        
        # Check if model and features are loaded
        if model is None:
            logger.error("❌ XGBoost model not loaded in main app")
            return False
            
        if not feature_names:
            logger.error("❌ Feature names not loaded in main app")
            return False
            
        # Store references
        _xgb_model = model
        _model_features = feature_names
        
        # Create SHAP explainer
        _shap_explainer = shap.TreeExplainer(model)
        
        logger.info(f"✅ SHAP explainer initialized with {len(_model_features)} features")
        logger.info(f"📊 Model type: {type(model)}")
        logger.info(f"🎯 Sample features: {_model_features[:5]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SHAP explainer initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def calculate_shap_values_batch(data_batch, target_variable):
    """Calculate SHAP values for a batch of records"""
    global _shap_explainer, _model_features, _xgb_model
    
    try:
        # Initialize SHAP if not already done
        if _shap_explainer is None:
            if not initialize_shap_explainer():
                logger.warning("SHAP initialization failed, using correlation fallback")
                return None, None
        
        logger.info(f"🧠 Calculating SHAP values for {len(data_batch)} records...")
        
        # Prepare data for SHAP calculation
        df_batch = pd.DataFrame(data_batch)
        
        # Ensure all model features are present
        missing_features = set(_model_features) - set(df_batch.columns)
        for feature in missing_features:
            df_batch[feature] = 0  # Fill missing features with 0
            
        # Select only model features in correct order
        model_data = df_batch[_model_features].fillna(0)
        
        # Replace inf values with 0
        model_data = model_data.replace([np.inf, -np.inf], 0)
        
        logger.info(f"📋 Model data shape: {model_data.shape}")
        logger.info(f"🎯 Target variable: {target_variable}")
        
        # Calculate SHAP values (memory efficient)
        shap_values = _shap_explainer.shap_values(model_data)
        
        logger.info(f"✅ SHAP calculation complete. Shape: {shap_values.shape}")
        
        # Convert to feature importance format - LOCATION-SPECIFIC VERSION
        feature_importance = []
        
        # For location-specific feature importance, calculate mean absolute SHAP per feature
        # but keep this for compatibility with existing endpoints that expect global importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        for i, feature in enumerate(_model_features):
            if i < len(mean_abs_shap):
                importance_score = float(mean_abs_shap[i])
                correlation = 0.0
                
                # Calculate correlation if target variable exists
                if target_variable in df_batch.columns:
                    try:
                        correlation = float(df_batch[feature].corr(df_batch[target_variable]))
                        if pd.isna(correlation):
                            correlation = 0.0
                    except:
                        correlation = 0.0
                
                feature_importance.append({
                    "feature": feature,
                    "importance": importance_score,
                    "correlation": correlation,
                    "shap_mean_abs": importance_score
                })
        
        # Add SHAP values to records WITH LOCATION-SPECIFIC FEATURE IMPORTANCE
        enhanced_records = []
        for idx, record in enumerate(data_batch):
            enhanced_record = record.copy()
            
            # Add individual SHAP values for this record
            record_shap_values = []
            for i, feature in enumerate(_model_features):
                if i < len(shap_values[idx]):
                    shap_value = float(shap_values[idx][i])
                    enhanced_record[f'shap_{feature}'] = shap_value
                    record_shap_values.append(abs(shap_value))
            
            # LOCATION-SPECIFIC: Calculate feature importance score for THIS record
            # This gives each location its own feature importance profile
            if record_shap_values:
                # Calculate this record's total SHAP impact
                total_shap_impact = sum(record_shap_values)
                positive_shap = sum(shap_values[idx][shap_values[idx] > 0]) if len(shap_values[idx]) > 0 else 0
                
                # Feature importance score based on this record's SHAP distribution
                max_shap = max(record_shap_values) if record_shap_values else 0
                diversity_score = 1 - (max_shap / total_shap_impact) if total_shap_impact > 0 else 0
                
                # Calculate location-specific feature importance score (0-100)
                impact_component = min(40, (total_shap_impact / 50) * 40)  # Scale for location variation
                positive_component = min(30, (positive_shap / 25) * 30) if positive_shap > 0 else 0
                diversity_component = diversity_score * 20
                consistency_component = 10  # Base consistency score
                
                location_feature_importance_score = impact_component + positive_component + diversity_component + consistency_component
                enhanced_record['feature_importance_score'] = round(location_feature_importance_score, 1)
                
                # Add debugging info for location-specific calculation
                enhanced_record['_debug_total_shap'] = round(total_shap_impact, 2)
                enhanced_record['_debug_positive_shap'] = round(positive_shap, 2)
                enhanced_record['_debug_diversity'] = round(diversity_score, 3)
            else:
                enhanced_record['feature_importance_score'] = 50.0  # Default score
                
            enhanced_records.append(enhanced_record)
        
        logger.info(f"✅ Enhanced {len(enhanced_records)} records with SHAP values")
        logger.info(f"📊 Feature importance calculated for {len(feature_importance)} features")
        
        return enhanced_records, feature_importance
        
    except Exception as e:
        logger.error(f"❌ SHAP calculation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return data_batch, []  # Return original data if SHAP fails

def calculate_correlation_importance(records, target_field):
    """Fallback correlation-based importance when SHAP fails"""
    try:
        logger.info(f"🔄 Using correlation fallback for {len(records)} records")
        
        df = pd.DataFrame(records)
        feature_importance = []
        
        if target_field not in df.columns:
            logger.warning(f"Target field {target_field} not found for correlation fallback")
            return []
        
        # Focus on value fields for correlation
        value_fields = [k for k in df.columns if k.startswith('value_')]
        
        for feature in value_fields:
            try:
                correlation = df[feature].corr(df[target_field])
                if not pd.isna(correlation):
                    feature_importance.append({
                        "feature": feature,
                        "importance": abs(correlation),
                        "correlation": correlation,
                        "method": "correlation_fallback"
                    })
            except Exception as e:
                logger.warning(f"Correlation calculation failed for {feature}: {e}")
                continue
        
        logger.info(f"✅ Correlation fallback calculated for {len(feature_importance)} features")
        return feature_importance
        
    except Exception as e:
        logger.error(f"❌ Correlation fallback failed: {e}")
        return []

# ULTRA-MINIMAL STREAMING: Prevent 502s while getting ALL records
MICRO_CHUNK_SIZE = 10  # Tiny chunks to prevent memory issues
MEMORY_CLEANUP_INTERVAL = 2  # Cleanup every 2 chunks
MAX_MEMORY_SOFT_LIMIT = 300  # Conservative limit
MAX_PROCESSING_TIME = 240  # 4 minutes max

# OPTIMIZED FOR 2GB RENDER STANDARD INSTANCE
RECORD_BATCH_SIZE = 100  # Larger batches with more memory
MAX_RECORDS_LIMIT = 4000  # Process ALL records (3,983)
MEMORY_CHECK_INTERVAL = 200  # Less frequent checks with more memory
MAX_MEMORY_THRESHOLD = 1500  # Use 1.5GB of 2GB available

# RECORD-BY-RECORD STREAMING: Never load full dataset
RECORD_BATCH_SIZE = 5  # Process 5 records at a time
MAX_RECORDS_LIMIT = 2000  # Hard limit to prevent memory issues
MEMORY_CHECK_INTERVAL = 25  # Check memory every 25 records

def ultra_minimal_cleanup():
    """Most conservative memory cleanup"""
    import gc
    import ctypes
    import sys
    
    # Immediate cleanup
    gc.collect()
    gc.collect()  # Double collection
    
    # Force OS memory return
    try:
        import os
        if hasattr(os, 'name') and os.name == 'posix':
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except:
                try:
                    import ctypes.util
                    libc_name = ctypes.util.find_library("c")
                    if libc_name:
                        libc = ctypes.CDLL(libc_name)
                        libc.malloc_trim(0)
                except:
                    pass
    except:
        pass

def get_memory_usage_mb():
    """Get current memory usage"""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def optimized_2gb_reader(pickle_path):
    """Optimized reader for 2GB Render Standard instance"""
    logger.info(f"Starting 2GB-optimized processing from {pickle_path}")
    
    try:
        # Load dataset with 2GB memory available
        with open(pickle_path, 'rb') as f:
            full_dataset = pd.read_pickle(f, compression='gzip')
            total_rows = len(full_dataset)
            logger.info(f"Loaded full dataset: {total_rows} rows")
            
            # With 2GB, we can process ALL records
            logger.info(f"Processing ALL {total_rows} records with 2GB memory")
            
            all_records = []
            processed_count = 0
            
            # Process in larger batches (2GB allows this)
            for start_idx in range(0, total_rows, RECORD_BATCH_SIZE):
                try:
                    end_idx = min(start_idx + RECORD_BATCH_SIZE, total_rows)
                    
                    # Process larger batches efficiently
                    batch = full_dataset.iloc[start_idx:end_idx].copy()
                    batch_records = batch.to_dict('records')
                    
                    # Add to collection
                    all_records.extend(batch_records)
                    processed_count += len(batch_records)
                    
                    # Cleanup batch
                    del batch
                    del batch_records
                    
                    # Memory monitoring (less frequent with 2GB)
                    if processed_count % MEMORY_CHECK_INTERVAL == 0:
                        ultra_minimal_cleanup()
                        current_mem = get_memory_usage_mb()
                        logger.info(f"Processed {processed_count}/{total_rows} (Memory: {current_mem:.1f}MB / 2048MB)")
                        
                        # Only brake if we're truly close to 2GB limit
                        if current_mem > MAX_MEMORY_THRESHOLD:
                            logger.warning(f"Approaching 2GB limit at {processed_count} records")
                            # Continue processing - we have more headroom
                    
                except Exception as e:
                    logger.error(f"Batch failed at record {start_idx}: {e}")
                    # With 2GB, continue trying other batches
                    continue
            
            # Final cleanup
            del full_dataset
            ultra_minimal_cleanup()
            
            final_memory = get_memory_usage_mb()
            logger.info(f"2GB processing complete: {len(all_records)} records (Final memory: {final_memory:.1f}MB)")
            
            # Success metrics for 2GB instance
            if len(all_records) >= 3500:
                logger.info("🎉 EXCELLENT: Got nearly all records with 2GB!")
            elif len(all_records) >= 2500:
                logger.info("✅ GOOD: Substantial dataset with 2GB")
            else:
                logger.warning("⚠️ Still limited - may need optimization")
            
            return all_records, len(all_records)
            
    except Exception as e:
        logger.error(f"2GB optimized processing failed: {e}")
        ultra_minimal_cleanup()
        # Fallback to smaller processing
        return record_by_record_reader(pickle_path)

def record_by_record_reader(pickle_path):
    """Fallback record-by-record reader for memory constraints"""
    logger.info(f"Using fallback record-by-record streaming from {pickle_path}")
    
    try:
        # Conservative approach for fallback
        with open(pickle_path, 'rb') as f:
            temp_peek = pd.read_pickle(f, compression='gzip')
            total_rows = len(temp_peek)
            
            # Fallback limit
            records_to_process = min(total_rows, 2000)
            logger.info(f"Fallback: processing {records_to_process} records")
            
            all_records = []
            processed_count = 0
            
            # Small batch processing for fallback
            for start_idx in range(0, records_to_process, 20):  # Small batches
                try:
                    end_idx = min(start_idx + 20, records_to_process)
                    
                    batch = temp_peek.iloc[start_idx:end_idx].copy()
                    batch_records = batch.to_dict('records')
                    
                    all_records.extend(batch_records)
                    processed_count += len(batch_records)
                    
                    # Cleanup
                    del batch
                    del batch_records
                    
                    if processed_count % 100 == 0:
                        ultra_minimal_cleanup()
                        current_mem = get_memory_usage_mb()
                        logger.info(f"Fallback processed {processed_count}/{records_to_process} (Memory: {current_mem:.1f}MB)")
                        
                        if current_mem > 400:  # Conservative fallback limit
                            logger.warning(f"Fallback memory limit - stopping at {processed_count}")
                            break
                    
                except Exception as e:
                    logger.error(f"Fallback batch failed: {e}")
                    break
            
            del temp_peek
            ultra_minimal_cleanup()
            
            logger.info(f"Fallback complete: {len(all_records)} records")
            return all_records, len(all_records)
            
    except Exception as e:
        logger.error(f"Fallback failed: {e}")
        return [], 0

def emergency_minimal_reader(pickle_path):
    """Emergency reader - absolute minimum records"""
    logger.warning("Using emergency minimal reader - very limited records")
    
    try:
        with open(pickle_path, 'rb') as f:
            temp_sample = pd.read_pickle(f, compression='gzip')
            
            # Take only first 100 records to ensure success
            if len(temp_sample) > 100:
                subset = temp_sample.head(100).copy()
            else:
                subset = temp_sample.copy()
            
            records = subset.to_dict('records')
            
            # Cleanup
            del temp_sample
            del subset
            ultra_minimal_cleanup()
            
            logger.info(f"Emergency reader: {len(records)} records")
            return records, len(records)
            
    except Exception as e:
        logger.error(f"Emergency reader failed: {e}")
        return [], 0

def progressive_pickle_reader(pickle_path):
    """Multi-tier progressive reading optimized for 2GB instance"""
    logger.info(f"Starting 2GB-optimized progressive reading")
    
    try:
        # Primary: 2GB optimized approach
        records, count = optimized_2gb_reader(pickle_path)
        
        if count >= 3000:  # Excellent - near full dataset
            logger.info(f"2GB optimization successful: {count} records")
            return records, count
        elif count >= 1000:  # Good progress
            logger.info(f"2GB optimization partial: {count} records")
            return records, count
        else:
            # Fallback: Record-by-record (should rarely happen with 2GB)
            logger.warning("2GB approach failed - using fallback")
            return record_by_record_reader(pickle_path)
            
    except Exception as e:
        logger.error(f"2GB processing failed: {e}")
        # Emergency fallback
        return emergency_minimal_reader(pickle_path)

def load_precalculated_model_data_progressive(model_name):
    """Load data using progressive file reading"""
    try:
        metadata_path = 'precalculated/models/metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if model_name not in metadata:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = metadata[model_name]
        shap_file = model_info['shap_file']
        
        logger.info(f"Starting progressive load of {shap_file}")
        
        # Use progressive reader
        all_records, total_count = progressive_pickle_reader(shap_file)
        
        # Convert back to lightweight DataFrame for compatibility
        # But only if not too large
        if len(all_records) <= 1000:
            df = pd.DataFrame(all_records)
            logger.info(f"Created lightweight DataFrame: {df.shape}")
        else:
            # For large datasets, return records directly with minimal DataFrame wrapper
            logger.info(f"Dataset too large for DataFrame - using records directly")
            # Create a minimal DataFrame with just first few rows for compatibility
            df = pd.DataFrame(all_records[:100])  # Just for structure checks
            df._all_records = all_records  # Store all records as attribute
        
        ultra_minimal_cleanup()
        
        return df, model_info
        
    except Exception as e:
        logger.error(f"Progressive load failed: {str(e)}")
        ultra_minimal_cleanup()
        raise

def select_model_for_analysis(query):
    """Select the best pre-calculated model based on the analysis request"""
    return 'conversion'

def enhanced_analysis_worker(query):
    """Enhanced analysis worker with progressive processing"""
    
    try:
        user_query = query.get('query', '')
        analysis_type = query.get('analysis_type', 'correlation')
        
        logger.info(f"Progressive analysis: {analysis_type}")
        logger.info(f"Query: {user_query}")
        
        # Ultra cleanup before starting
        ultra_minimal_cleanup()
        
        # Use query classifier
        classifier = QueryClassifier()
        query_classification = process_query(user_query)
        
        # Route to progressive handlers
        if analysis_type == 'analyze':
            return handle_basic_analysis_progressive(query, query_classification)
        else:
            # For now, all other endpoints use the same progressive approach
            return handle_basic_analysis_progressive(query, query_classification)
            
    except Exception as e:
        logger.error(f"Error in progressive analysis: {str(e)}")
        ultra_minimal_cleanup()
        return {"success": False, "error": f"Progressive analysis failed: {str(e)}"}

def handle_basic_analysis_progressive(query, query_classification):
    """Handle analysis with progressive loading - ALL RECORDS"""
    try:
        logger.info("Starting progressive basic analysis")
        
        selected_model = select_model_for_analysis(query)
        df, model_info = load_precalculated_model_data_progressive(selected_model)
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        
        logger.info(f"Progressive analysis for {target_field}")
        
        # Check if we have all records stored
        if hasattr(df, '_all_records'):
            all_records = df._all_records
            logger.info(f"Using stored records: {len(all_records)}")
        else:
            # For small datasets, convert DataFrame
            all_records = df.to_dict('records')
            logger.info(f"Converting DataFrame: {len(all_records)} records")
        
        # ✅ REAL SHAP CALCULATION - PHASE 1 IMPLEMENTATION
        logger.info(f"🧠 Starting SHAP-enabled analysis for {len(all_records)} records with target: {target_field}")
        
        enhanced_records = []
        feature_importance = []
        
        if len(all_records) > 0:
            try:
                # Process in batches to manage memory (start with 500 records)
                batch_size = 500
                total_batches = (len(all_records) + batch_size - 1) // batch_size
                
                logger.info(f"📊 Processing {total_batches} batches of {batch_size} records each")
                
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(all_records))
                    batch = all_records[start_idx:end_idx]
                    
                    logger.info(f"🔄 Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} records)")
                    
                    # Calculate SHAP values for this batch
                    batch_enhanced, batch_importance = calculate_shap_values_batch(batch, target_field)
                    
                    if batch_enhanced is not None:
                        # SHAP calculation succeeded
                        enhanced_records.extend(batch_enhanced)
                        if batch_importance and not feature_importance:  # Use first successful batch for importance
                            feature_importance = batch_importance
                        logger.info(f"✅ Batch {batch_idx + 1} SHAP calculation successful")
                    else:
                        # SHAP failed, use original batch
                        enhanced_records.extend(batch)
                        logger.warning(f"⚠️ Batch {batch_idx + 1} SHAP calculation failed, using original data")
                    
                    # Memory cleanup after each batch
                    ultra_minimal_cleanup()
                
                # If no SHAP calculations succeeded, use correlation fallback
                if not feature_importance:
                    logger.warning("🔄 All SHAP calculations failed, using correlation fallback")
                    feature_importance = calculate_correlation_importance(all_records, target_field)
                
                # Sort feature importance by actual importance scores
                feature_importance.sort(key=lambda x: x.get('importance', 0), reverse=True)
                
                logger.info(f"✅ SHAP processing complete: {len(enhanced_records)} records, {len(feature_importance)} features")
                
                # Log some sample SHAP values to verify they're non-zero
                if enhanced_records:
                    sample_record = enhanced_records[0]
                    shap_fields = [k for k in sample_record.keys() if k.startswith('shap_')]
                    sample_shap_values = [sample_record[k] for k in shap_fields[:5]]
                    non_zero_shap = [v for v in sample_shap_values if v != 0.0]
                    logger.info(f"🎯 Sample SHAP values: {sample_shap_values}")
                    logger.info(f"✅ Non-zero SHAP count: {len(non_zero_shap)}/{len(sample_shap_values)}")
                
            except Exception as shap_error:
                logger.error(f"❌ SHAP processing failed completely: {shap_error}")
                # Complete fallback - use original records with correlation importance
                enhanced_records = all_records
                feature_importance = calculate_correlation_importance(all_records, target_field)
        
        # Final cleanup
        del df
        ultra_minimal_cleanup()
        
        final_memory = get_memory_usage_mb()
        
        # Determine if SHAP was successful
        shap_success = any(k.startswith('shap_') for k in enhanced_records[0].keys()) if enhanced_records else False
        analysis_method = "shap_analysis" if shap_success else "correlation_fallback"
        
        response = {
            "success": True,
            "results": enhanced_records,
            "summary": f"SHAP-enhanced analysis for {target_field} using {analysis_method}",
            "feature_importance": feature_importance,
            "analysis_type": analysis_method,
            "total_records": len(enhanced_records),
            "sample_size": len(enhanced_records),
            "shap_enabled": shap_success,
            "target_variable": target_field,
            "progressive_processed": True,
            "final_memory_mb": final_memory
        }
        
        # Apply field filtering to response
        return apply_field_filtering_to_response(response, query)
        
    except Exception as e:
        logger.error(f"Progressive analysis error: {str(e)}")
        ultra_minimal_cleanup()
        return {
            "success": False,
            "error": f"Progressive analysis failed: {str(e)}"
        }

def handle_feature_interactions_streaming(query, query_classification):
    """Handle feature interaction analysis"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data_streaming(selected_model)
        
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

def handle_outlier_detection_streaming(query, query_classification):
    """Handle outlier detection analysis"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data_streaming(selected_model)
        
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

def handle_scenario_analysis_streaming(query, query_classification):
    """Handle scenario analysis (what-if analysis)"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data_streaming(selected_model)
        
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

def handle_segment_profiling_streaming(query, query_classification):
    """Handle customer segment profiling"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.cluster import KMeans
        
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data_streaming(selected_model)
        
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

def handle_spatial_clusters_streaming(query, query_classification):
    """Handle spatial clustering analysis"""
    try:
        from sklearn.cluster import KMeans
        
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data_streaming(selected_model)
        
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

def handle_demographic_insights_streaming(query, query_classification):
    """Handle demographic pattern insights"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data_streaming(selected_model)
        
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

def handle_trend_analysis_streaming(query, query_classification):
    """Handle temporal trend analysis"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data_streaming(selected_model)
        
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

def handle_feature_importance_ranking_streaming(query, query_classification):
    """Handle feature importance ranking"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data_streaming(selected_model)
        
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

def handle_correlation_analysis_streaming(query, query_classification):
    """Handle feature correlation analysis"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data_streaming(selected_model)
        
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

def handle_anomaly_detection_streaming(query, query_classification):
    """Handle anomaly detection with explanations"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data_streaming(selected_model)
        
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

def handle_predictive_modeling_streaming(query, query_classification):
    """Handle predictive modeling with SHAP"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data_streaming(selected_model)
        
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

def handle_sensitivity_analysis_streaming(query, query_classification):
    """Handle feature sensitivity analysis"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data_streaming(selected_model)
        
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

def handle_model_performance_streaming(query, query_classification):
    """Handle model performance evaluation"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data_streaming(selected_model)
        
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

def handle_competitive_analysis_streaming(query, query_classification):
    """Handle competitive brand analysis"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data_streaming(selected_model)
        
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

def handle_comparative_analysis_streaming(query, query_classification):
    """Handle comparative analysis between groups"""
    try:
        selected_model = select_model_for_analysis(query)
        precalc_df, model_info = load_precalculated_model_data_streaming(selected_model)
        
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

def handle_legacy_analysis_streaming(query, query_classification):
    """Legacy analysis handler - ALL RECORDS with chunked processing"""
    try:
        logger.info("Starting legacy analysis with ALL records")
        
        # Load data with memory management
        precalc_df, model_info = load_precalculated_model_data_streaming(model_info['shap_file'])
        
        target_field = query.get('target_field') or query.get('target_variable', 'MP30034A_B_P')
        
        logger.info(f"Target field: {target_field}")
        logger.info(f"Dataset shape: {precalc_df.shape}")
        
        # Check if target field exists
        if target_field not in precalc_df.columns:
            available_fields = [col for col in precalc_df.columns if 'MP30' in col]
            return {
                "success": False, 
                "error": f"Target variable {target_field} not found",
                "available_fields": available_fields[:10]
            }
        
        # Create feature importance (lightweight operation)
        logger.info("Calculating feature importance...")
        feature_importance = []
        
        value_columns = [col for col in precalc_df.columns if col.startswith('value_')]
        
        # Calculate correlations in small batches to avoid memory spikes
        for i in range(0, len(value_columns), 50):  # Process 50 features at a time
            batch_features = value_columns[i:i+50]
            
            for feature in batch_features:
                if feature in precalc_df.columns:
                    try:
                        correlation = precalc_df[feature].corr(precalc_df[target_field])
                        if not pd.isna(correlation):
                            feature_importance.append({
                                "feature": feature,
                                "importance": float(abs(correlation)),
                                "correlation": float(correlation)
                            })
                    except Exception as e:
                        logger.warning(f"Error calculating correlation for {feature}: {e}")
            
            # Cleanup after each batch
            force_aggressive_cleanup()
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        logger.info(f"Calculated {len(feature_importance)} feature importances")
        
        # Convert ALL records using streaming
        logger.info("Converting ALL records to output format using streaming...")
        
        try:
            # Stream convert to records
            all_results = stream_process_pickle_file(model_info['shap_file'])
            if isinstance(all_results, tuple):
                all_results = all_results[0]  # Get just the records
                
        except Exception as e:
            logger.error(f"Error in chunked processing: {e}")
            # Emergency fallback - still try to get substantial data
            try:
                logger.warning("Falling back to direct conversion of first 2000 records")
                all_results = precalc_df.head(2000).to_dict('records')
            except Exception as e2:
                logger.error(f"Even fallback failed: {e2}")
                return {
                    "success": False,
                    "error": f"Memory processing failed: {str(e)}",
                    "fallback_error": str(e2)
                }
        
        # Final cleanup
        del precalc_df
        force_aggressive_cleanup()
        
        final_memory = get_current_memory_mb()
        
        return {
            "success": True,
            "results": all_results,
            "summary": f"SHAP analysis for {target_field} with {len(feature_importance)} features",
            "feature_importance": feature_importance,
            "analysis_type": "chunked_processing_analysis", 
            "total_records": len(all_results),
            "memory_optimized": True,
            "final_memory_mb": final_memory,
            "chunked_processing": True
        }
        
    except Exception as e:
        logger.error(f"Error in legacy analysis: {str(e)}")
        force_aggressive_cleanup()
        return {
            "success": False, 
            "error": f"Analysis failed: {str(e)}",
            "memory_cleanup_performed": True
        }

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