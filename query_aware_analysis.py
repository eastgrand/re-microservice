import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple

def analyze_query_intent(query: str, conversation_context: str = '') -> Dict[str, any]:
    """Analyze the user's query to understand their analytical intent"""
    
    query_lower = query.lower()
    
    # If we have conversation context, incorporate it for better intent detection
    if conversation_context:
        # Look for follow-up indicators in the query
        follow_up_indicators = ['why', 'how', 'what about', 'more like', 'similar to', 'based on that', 'tell me more']
        is_follow_up = any(indicator in query_lower for indicator in follow_up_indicators)
        
        if is_follow_up:
            # For follow-up questions, extract context from previous conversation
            context_lower = conversation_context.lower()
            
            # Inherit focus areas from context
            inherited_focus = []
            if any(term in context_lower for term in ['demographic', 'diversity', 'population', 'ethnic']):
                inherited_focus.append('demographic')
            if any(term in context_lower for term in ['income', 'financial', 'economic', 'mortgage']):
                inherited_focus.append('financial')
            if any(term in context_lower for term in ['housing', 'geographic', 'area', 'region']):
                inherited_focus.append('geographic')
                
            # Extract mentioned areas from context
            inherited_concepts = []
            if 'conversion' in context_lower or 'approval' in context_lower:
                inherited_concepts.append('conversion')
            if 'diversity' in context_lower:
                inherited_concepts.append('diversity')
            if 'income' in context_lower:
                inherited_concepts.append('income')
            if 'housing' in context_lower:
                inherited_concepts.append('housing')
    
    # Determine analysis type
    analysis_type = 'correlation'  # Default
    if any(word in query_lower for word in ['highest', 'top', 'best', 'rank', 'leading']):
        analysis_type = 'ranking'
    elif any(word in query_lower for word in ['compare', 'difference', 'between', 'vs', 'versus']):
        analysis_type = 'comparison'
    elif any(word in query_lower for word in ['correlate', 'relationship', 'relate', 'connection']):
        analysis_type = 'correlation'
    elif any(word in query_lower for word in ['distribution', 'spread', 'pattern', 'trend']):
        analysis_type = 'distribution'
    
    # Determine focus areas
    focus_areas = []
    if any(word in query_lower for word in ['demographic', 'diversity', 'population', 'ethnic', 'minority', 'race']):
        focus_areas.append('demographic')
    if any(word in query_lower for word in ['income', 'financial', 'economic', 'money', 'mortgage', 'loan']):
        focus_areas.append('financial')
    if any(word in query_lower for word in ['housing', 'geographic', 'area', 'region', 'location', 'structure']):
        focus_areas.append('geographic')
    
    # Merge with inherited focus from conversation context
    if conversation_context and 'inherited_focus' in locals():
        focus_areas.extend([f for f in inherited_focus if f not in focus_areas])
    
    # Extract key concepts
    key_concepts = []
    if any(word in query_lower for word in ['diversity', 'diverse']):
        key_concepts.append('diversity')
    if any(word in query_lower for word in ['income', 'earnings']):
        key_concepts.append('income')
    if any(word in query_lower for word in ['housing', 'house', 'apartment', 'structure']):
        key_concepts.append('housing')
    if any(word in query_lower for word in ['conversion', 'approval', 'rate']):
        key_concepts.append('conversion')
    if any(word in query_lower for word in ['application', 'applications']):
        key_concepts.append('applications')
    
    # Merge with inherited concepts from conversation context
    if conversation_context and 'inherited_concepts' in locals():
        key_concepts.extend([c for c in inherited_concepts if c not in key_concepts])
    
    return {
        'analysis_type': analysis_type,
        'focus_areas': focus_areas,
        'key_concepts': key_concepts,
        'confidence': 0.8 if focus_areas else 0.5,
        'is_follow_up': conversation_context and is_follow_up if 'is_follow_up' in locals() else False,
        'context_enhanced': bool(conversation_context)
    }

def get_relevant_features_by_intent(intent: Dict, all_features: List[str]) -> List[str]:
    """Get features most relevant to the user's intent"""
    
    relevant_features = []
    focus_areas = intent['focus_areas']
    key_concepts = intent.get('key_concepts', [])
    
    # First, identify specific concepts mentioned in the query
    specific_concepts = set()
    for concept in key_concepts:
        if 'diversity' in concept.lower():
            specific_concepts.update(['visible minority', 'population diversity'])
        elif 'income' in concept.lower():
            specific_concepts.update(['income', 'household income', 'discretionary income'])
        elif 'housing' in concept.lower():
            specific_concepts.update(['housing', 'structure type', 'tenure'])
        elif 'conversion' in concept.lower():
            specific_concepts.update(['conversion rate', 'mortgage approval'])
    
    # Then, select features based on both focus areas and specific concepts
    if 'demographic' in focus_areas:
        # Only include demographic features that match specific concepts
        demo_features = []
        for f in all_features:
            f_lower = f.lower()
            # Check if feature matches any specific concepts
            if any(concept in f_lower for concept in specific_concepts):
                demo_features.append(f)
            # If no specific concepts, use broader matching
            elif not specific_concepts and any(x in f_lower for x in 
                    ['visible minority', 'population', 'age', 'married', 'divorced', 'single', 'maintainer']):
                demo_features.append(f)
        relevant_features.extend(demo_features)
    
    if 'geographic' in focus_areas:
        # Only include geographic features that match specific concepts
        geo_features = []
        for f in all_features:
            f_lower = f.lower()
            if any(concept in f_lower for concept in specific_concepts):
                geo_features.append(f)
            elif not specific_concepts and any(x in f_lower for x in 
                   ['housing', 'construction', 'tenure', 'structure', 'condominium', 'apartment']):
                geo_features.append(f)
        relevant_features.extend(geo_features)
    
    if 'financial' in focus_areas:
        # Only include financial features that match specific concepts
        fin_features = []
        for f in all_features:
            f_lower = f.lower()
            if any(concept in f_lower for concept in specific_concepts):
                fin_features.append(f)
            elif not specific_concepts and any(x in f_lower for x in 
                   ['income', 'mortgage', 'property', 'employment', 'financial', 'shelter', 'tax']):
                fin_features.append(f)
        relevant_features.extend(fin_features)
    
    # If no specific focus or no features found, return top features by importance
    if not relevant_features:
        relevant_features = all_features
    
    # Remove duplicates and return
    return list(set(relevant_features))

def generate_intent_aware_summary(intent: Dict, feature_importance: List[Dict], 
                                 target_variable: str, results: List[Dict],
                                 conversation_context: str = '') -> str:
    """Generate a conversational summary that directly answers the user's question"""
    
    analysis_type = intent['analysis_type']
    focus_areas = intent['focus_areas']
    key_concepts = intent['key_concepts']
    is_follow_up = intent.get('is_follow_up', False)
    
    # Get top results for context
    top_results = results[:3] if results else []
    
    # If this is a follow-up question, reference the conversation context
    if is_follow_up and conversation_context:
        if 'why' in intent.get('query', '').lower():
            summary = "Building on our previous analysis, "
        elif 'what about' in intent.get('query', '').lower():
            summary = "Expanding on those findings, "
        elif 'more like' in intent.get('query', '').lower() or 'similar' in intent.get('query', '').lower():
            summary = "Looking for areas with similar characteristics, "
        else:
            summary = "To elaborate on that analysis, "
    else:
        # Start with a direct answer to their question
        if analysis_type == 'ranking' and 'diversity' in key_concepts and top_results:
            summary = f"Looking at areas with both high diversity and strong {target_variable.lower().replace('_', ' ')}, "
        elif analysis_type == 'correlation' and len(key_concepts) >= 2:
            summary = f"When examining the relationship between {' and '.join(key_concepts)}, "
        elif analysis_type == 'ranking' and 'income' in key_concepts:
            summary = f"Focusing on areas with high income levels and strong {target_variable.lower().replace('_', ' ')}, "
        elif analysis_type == 'ranking':
            summary = f"Looking at the top-performing areas for {target_variable.lower().replace('_', ' ')}, "
        else:
            summary = f"Analyzing the data for {target_variable.lower().replace('_', ' ')}, "
    
    # Add specific insights based on top factors
    if feature_importance and len(feature_importance) > 0:
        top_factor = feature_importance[0]['feature']
        
        # Make it conversational and human-readable
        if 'income' in top_factor.lower():
            factor_description = "income levels"
        elif 'filipino' in top_factor.lower():
            factor_description = "Filipino population concentration"
        elif 'south asian' in top_factor.lower():
            factor_description = "South Asian community presence"
        elif 'chinese' in top_factor.lower():
            factor_description = "Chinese population density"
        elif any(term in top_factor.lower() for term in ['minority', 'population']):
            factor_description = "demographic composition"
        elif 'housing' in top_factor.lower() or 'structure' in top_factor.lower():
            factor_description = "housing characteristics"
        elif 'apartment' in top_factor.lower():
            factor_description = "apartment-style housing"
        elif 'employment' in top_factor.lower():
            factor_description = "employment patterns"
        else:
            factor_description = top_factor.lower().replace('_', ' ')
        
        # Different phrasing for follow-ups vs initial questions
        if is_follow_up:
            summary += f"the key driver appears to be {factor_description}. "
        else:
            summary += f"areas with strong {factor_description} consistently show the best performance. "
        
        # Add context-aware insights
        if len(feature_importance) >= 2:
            second_factor = feature_importance[1]['feature']
            if 'income' in second_factor.lower() and 'income' not in factor_description:
                summary += "Combined with solid income levels, "
            elif any(term in second_factor.lower() for term in ['housing', 'structure']) and 'housing' not in factor_description:
                summary += "Along with favorable housing characteristics, "
            elif any(term in second_factor.lower() for term in ['minority', 'population']) and 'demographic' not in factor_description:
                summary += "Paired with specific demographic patterns, "
            else:
                summary += f"Working together with {second_factor.lower().replace('_', ' ')}, "
            
            summary += "these factors create a strong foundation for mortgage success."
    
    # Add specific area mentions if available
    if top_results and not is_follow_up:
        # Mention specific high-performing areas
        top_area_ids = [str(result.get('zip_code', '')) for result in top_results[:2] if result.get('zip_code')]
        if top_area_ids:
            if len(top_area_ids) == 1:
                summary += f" {top_area_ids[0]} stands out as a particularly strong performer."
            else:
                summary += f" Areas like {', '.join(top_area_ids)} exemplify these successful patterns."
    
    # Add forward-looking insight for conversation continuity
    if not is_follow_up and len(feature_importance) >= 3:
        summary += " This multi-factor approach suggests that successful areas typically have several complementary strengths rather than relying on just one characteristic."
    
    return summary

def enhanced_query_aware_analysis(query: str, precalc_df: pd.DataFrame, 
                                 feature_importance: List[Dict], results: List[Dict],
                                 target_variable: str = 'CONVERSION_RATE',
                                 conversation_context: str = '') -> Dict:
    """Enhanced analysis that interprets results based on query intent and conversation context"""
    
    # Analyze query intent, enriched with conversation context
    intent = analyze_query_intent(query, conversation_context)
    
    # Get all available features from SHAP columns
    all_features = [col.replace('shap_', '') for col in precalc_df.columns if col.startswith('shap_')]
    
    # Filter feature importance to match query intent
    relevant_features = get_relevant_features_by_intent(intent, all_features)
    
    # Re-rank feature importance based on relevance
    relevant_importance = []
    for factor in feature_importance:
        if factor['feature'] in relevant_features:
            # Boost importance score for highly relevant features
            boost_multiplier = 1.0
            if intent['focus_areas']:
                feature_name = factor['feature'].lower()
                if 'demographic' in intent['focus_areas'] and any(x in feature_name for x in ['minority', 'population']):
                    boost_multiplier = 1.5
                elif 'financial' in intent['focus_areas'] and any(x in feature_name for x in ['income', 'mortgage']):
                    boost_multiplier = 1.5
                elif 'geographic' in intent['focus_areas'] and any(x in feature_name for x in ['housing', 'structure']):
                    boost_multiplier = 1.5
            
            relevant_importance.append({
                'feature': factor['feature'],
                'importance': factor['importance'] * boost_multiplier,
                'relevance_score': boost_multiplier
            })
    
    # Sort by boosted importance
    relevant_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    # Generate intent-aware summary with conversation context
    summary = generate_intent_aware_summary(intent, relevant_importance, target_variable, results, conversation_context)
    
    return {
        'query_intent': intent,
        'relevant_features': relevant_features[:20],  # Top 20 relevant features
        'enhanced_feature_importance': relevant_importance[:15],  # Top 15 with relevance
        'intent_aware_summary': summary,
        'analysis_focus': intent['focus_areas'],
        'key_concepts': intent['key_concepts'],
        'conversation_context': conversation_context  # Include context in response
    }

# Example usage:
EXAMPLE_QUERIES = {
    "diversity_analysis": {
        "query": "Which areas have the highest rates of diversity and conversion rate?",
        "expected_focus": ["demographic"],
        "expected_concepts": ["diversity"]
    },
    "housing_analysis": {
        "query": "How do different housing types affect mortgage approval rates?", 
        "expected_focus": ["geographic"],
        "expected_concepts": ["housing", "mortgage"]
    },
    "income_analysis": {
        "query": "What impact does household income have on loan conversion?",
        "expected_focus": ["financial"], 
        "expected_concepts": ["income"]
    },
    "comprehensive": {
        "query": "Show me the top areas for mortgage conversion",
        "expected_focus": [],  # No specific focus - comprehensive analysis
        "expected_concepts": ["mortgage"]
    }
} 