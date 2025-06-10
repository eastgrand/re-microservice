import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple

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
    analysis_type = 'correlation'
    if any(word in query_lower for word in ['highest', 'top', 'best', 'rank', 'leading', 'most']):
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
        'context_enhanced': bool(conversation_context),
        'query': query  # Add the original query for reference
    }

def generate_intent_aware_summary(intent: Dict, feature_importance: List[Dict], 
                                 target_variable: str, results: List[Dict],
                                 conversation_context: str = '') -> str:
    """Generate a conversational summary that directly answers the user's question"""
    
    analysis_type = intent['analysis_type']
    focus_areas = intent['focus_areas']
    key_concepts = intent['key_concepts']
    is_follow_up = intent.get('is_follow_up', False)
    query = intent.get('query', '').lower()
    
    # Get top results for context
    top_results = results[:3] if results else []
    
    # Special handling for application count queries
    if 'application' in key_concepts and analysis_type == 'ranking':
        if top_results:
            # Get the top areas and their application counts
            top_areas = [result.get('FSA_ID', result.get('ID', 'Unknown Area')) for result in top_results[:3]]
            top_counts = [int(result.get('FREQUENCY', 0)) for result in top_results[:3]]
            
            # Format the summary focusing on application counts
            summary = f"The areas with the most mortgage applications are {top_areas[0]} ({top_counts[0]} applications), "
            if len(top_areas) > 1:
                summary += f"{top_areas[1]} ({top_counts[1]} applications), "
            if len(top_areas) > 2:
                summary += f"and {top_areas[2]} ({top_counts[2]} applications). "
            
            # Add context about what makes these areas successful
            if feature_importance and len(feature_importance) >= 2:
                summary += f"These areas tend to have strong {feature_importance[0]['feature'].lower().replace('_', ' ')} "
                summary += f"and {feature_importance[1]['feature'].lower().replace('_', ' ')}."
            
            return summary
    
    # If this is a follow-up question, reference the conversation context
    if is_follow_up and conversation_context:
        if 'why' in query:
            summary = "Building on our previous analysis, "
        elif 'what about' in query:
            summary = "Expanding on those findings, "
        elif 'more like' in query or 'similar' in query:
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
    
    # Add analysis details
    if top_results:
        # Format results based on analysis type
        if analysis_type == 'ranking':
            top_areas = [result.get('FSA_ID', result.get('ID', 'Unknown Area')) for result in top_results[:3]]
            summary += f"the top performing areas are {', '.join(top_areas[:-1])} and {top_areas[-1]}. "
        elif analysis_type == 'correlation':
            summary += "we found significant correlations in the following areas: "
            summary += f"{', '.join([result.get('FSA_ID', result.get('ID', 'Unknown Area')) for result in top_results[:2]])}. "
    
    # Add insights about contributing factors
    if feature_importance and len(feature_importance) >= 2:
        summary += f"Key factors include {feature_importance[0]['feature'].lower().replace('_', ' ')} "
        summary += f"and {feature_importance[1]['feature'].lower().replace('_', ' ')}."
    
    return summary

def get_relevant_features_by_intent(intent: Dict, all_features: List[str]) -> List[str]:
    """Filter and prioritize features based on query intent"""
    
    # Start with all features
    relevant_features = set(all_features)
    
    # Prioritize features based on key concepts
    if 'applications' in intent['key_concepts']:
        # For application queries, prioritize application-related features
        priority_terms = ['frequency', 'application', 'approval', 'conversion']
        relevant_features = {f for f in all_features if any(term in f.lower() for term in priority_terms)}
        
        # Add demographic and geographic features as secondary factors
        secondary_terms = ['population', 'income', 'housing', 'location']
        relevant_features.update({f for f in all_features if any(term in f.lower() for term in secondary_terms)})
    
    # Convert back to list and sort alphabetically for consistency
    return sorted(list(relevant_features))

def enhanced_query_aware_analysis(query: str, precalc_df: pd.DataFrame, 
                                 feature_importance: List[Dict], results: List[Dict],
                                 target_variable: str = 'CONVERSION_RATE',
                                 conversation_context: str = '') -> Dict:
    """Enhanced analysis that interprets results based on query intent and conversation context"""
    
    # Analyze query intent, enriched with conversation context
    intent = analyze_query_intent(query, target_variable, results, feature_importance, conversation_context)
    
    # For application queries, use the specialized analysis
    if target_variable.lower() == 'frequency' and ('application' in query.lower() or 'applications' in query.lower()):
        return intent
    
    # For other query types, continue with standard analysis
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
            
            # Special handling for application queries
            if 'applications' in intent['key_concepts']:
                if any(x in factor['feature'].lower() for x in ['frequency', 'application', 'approval']):
                    boost_multiplier *= 2.0  # Double boost for application-related features
            
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
    },
    "applications": {
        "query": "Which areas have the most applications?",
        "expected_focus": [],  # No specific focus
        "expected_concepts": ["applications"]
    }
} 