import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple

def analyze_query_intent(query: str) -> Dict[str, any]:
    """Analyze the user's query to understand their analytical intent"""
    
    query_lower = query.lower()
    
    # Intent categories
    intent = {
        'focus_areas': [],
        'analysis_type': 'correlation',
        'key_concepts': [],
        'target_emphasis': 'conversion_rate'
    }
    
    # Demographic focus keywords
    demographic_keywords = [
        'diversity', 'diverse', 'minority', 'visible minority', 'population', 
        'demographic', 'age', 'married', 'single', 'divorced', 'family',
        'ethnic', 'cultural', 'community', 'residents'
    ]
    
    # Geographic/Housing focus keywords  
    geographic_keywords = [
        'housing', 'house', 'apartment', 'condo', 'condominium', 'structure',
        'construction', 'built', 'neighborhood', 'area', 'location', 'geography',
        'detached', 'semi-detached', 'building', 'property', 'tenure', 'owned', 'rented'
    ]
    
    # Financial focus keywords
    financial_keywords = [
        'income', 'salary', 'financial', 'mortgage', 'loan', 'money', 'funding',
        'employment', 'job', 'work', 'tax', 'property tax', 'shelter', 'cost',
        'economic', 'wealth', 'affluent', 'high-income', 'low-income'
    ]
    
    # Volume/Frequency analysis keywords
    volume_keywords = [
        'volume', 'amount', 'total', 'sum', 'funding', 'loan amount',
        'frequency', 'applications', 'activity', 'market activity'
    ]
    
    # Analysis type keywords
    ranking_keywords = ['top', 'best', 'highest', 'lowest', 'rank', 'which areas']
    correlation_keywords = ['affect', 'influence', 'impact', 'drive', 'factor', 'relationship']
    comparison_keywords = ['compare', 'versus', 'vs', 'difference', 'between']
    
    # Detect focus areas
    if any(keyword in query_lower for keyword in demographic_keywords):
        intent['focus_areas'].append('demographic')
        
    if any(keyword in query_lower for keyword in geographic_keywords):
        intent['focus_areas'].append('geographic')
        
    if any(keyword in query_lower for keyword in financial_keywords):
        intent['focus_areas'].append('financial')
    
    # Detect analysis type
    if any(keyword in query_lower for keyword in ranking_keywords):
        intent['analysis_type'] = 'ranking'
    elif any(keyword in query_lower for keyword in correlation_keywords):
        intent['analysis_type'] = 'correlation'
    elif any(keyword in query_lower for keyword in comparison_keywords):
        intent['analysis_type'] = 'comparison'
    
    # Detect target emphasis
    if any(keyword in query_lower for keyword in volume_keywords):
        intent['target_emphasis'] = 'volume'
    elif 'frequency' in query_lower or 'applications' in query_lower:
        intent['target_emphasis'] = 'frequency'
    
    # Extract key concepts
    concepts = []
    if 'diversity' in query_lower or 'diverse' in query_lower:
        concepts.append('diversity')
    if 'income' in query_lower:
        concepts.append('income')
    if 'housing' in query_lower or 'house' in query_lower:
        concepts.append('housing')
    if 'mortgage' in query_lower or 'loan' in query_lower:
        concepts.append('mortgage')
    
    intent['key_concepts'] = concepts
    
    return intent

def get_relevant_features_by_intent(intent: Dict, all_features: List[str]) -> List[str]:
    """Get features most relevant to the user's intent"""
    
    relevant_features = []
    focus_areas = intent['focus_areas']
    
    if 'demographic' in focus_areas:
        demo_features = [f for f in all_features if any(x in f.lower() for x in 
                        ['visible minority', 'population', 'age', 'married', 'divorced', 'single', 'maintainer'])]
        relevant_features.extend(demo_features)
    
    if 'geographic' in focus_areas:
        geo_features = [f for f in all_features if any(x in f.lower() for x in 
                       ['housing', 'construction', 'tenure', 'structure', 'condominium', 'apartment'])]
        relevant_features.extend(geo_features)
    
    if 'financial' in focus_areas:
        fin_features = [f for f in all_features if any(x in f.lower() for x in 
                       ['income', 'mortgage', 'property', 'employment', 'financial', 'shelter', 'tax'])]
        relevant_features.extend(fin_features)
    
    # If no specific focus, return top features by importance
    if not relevant_features:
        relevant_features = all_features
    
    return list(set(relevant_features))  # Remove duplicates

def generate_intent_aware_summary(intent: Dict, feature_importance: List[Dict], 
                                 target_variable: str, results: List[Dict]) -> str:
    """Generate a summary that addresses the user's specific query intent"""
    
    analysis_type = intent['analysis_type']
    focus_areas = intent['focus_areas']
    key_concepts = intent['key_concepts']
    
    # Start with context
    if focus_areas:
        focus_text = " and ".join(focus_areas)
        summary = f"Analysis focused on {focus_text} factors affecting {target_variable}. "
    else:
        summary = f"Comprehensive analysis of factors affecting {target_variable}. "
    
    # Add findings based on analysis type
    if analysis_type == 'ranking' and results:
        top_area = results[0]['zip_code']
        top_value = results[0].get(target_variable.lower(), 'N/A')
        summary += f"The top-performing area is {top_area} with a {target_variable} of {top_value:.3f}. "
    
    # Highlight relevant factors
    if feature_importance:
        # Filter feature importance to match intent
        relevant_factors = []
        for factor in feature_importance[:10]:  # Top 10 factors
            feature_name = factor['feature'].lower()
            
            # Check if factor matches intent
            is_relevant = False
            if 'demographic' in focus_areas and any(x in feature_name for x in ['minority', 'population', 'age', 'married']):
                is_relevant = True
            elif 'geographic' in focus_areas and any(x in feature_name for x in ['housing', 'construction', 'structure']):
                is_relevant = True
            elif 'financial' in focus_areas and any(x in feature_name for x in ['income', 'mortgage', 'employment']):
                is_relevant = True
            elif not focus_areas:  # If no specific focus, include all
                is_relevant = True
            
            if is_relevant:
                relevant_factors.append(factor)
        
        # Mention top relevant factors
        if relevant_factors:
            top_factor = relevant_factors[0]['feature']
            summary += f"The most significant factor is {top_factor}."
            
            if len(relevant_factors) >= 3:
                summary += f" Other key factors include {relevant_factors[1]['feature']} and {relevant_factors[2]['feature']}."
    
    # Add concept-specific insights
    if 'diversity' in key_concepts:
        diversity_factors = [f for f in feature_importance if 'minority' in f['feature'].lower()]
        if diversity_factors:
            summary += f" Diversity metrics show {diversity_factors[0]['feature']} has significant impact."
    
    if 'income' in key_concepts:
        income_factors = [f for f in feature_importance if 'income' in f['feature'].lower()]
        if income_factors:
            summary += f" Income-related factors, particularly {income_factors[0]['feature']}, are influential."
    
    return summary

def enhanced_query_aware_analysis(query: str, precalc_df: pd.DataFrame, 
                                 feature_importance: List[Dict], results: List[Dict],
                                 target_variable: str = 'CONVERSION_RATE') -> Dict:
    """Enhanced analysis that interprets results based on query intent"""
    
    # Analyze query intent
    intent = analyze_query_intent(query)
    
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
    
    # Generate intent-aware summary
    summary = generate_intent_aware_summary(intent, relevant_importance, target_variable, results)
    
    return {
        'query_intent': intent,
        'relevant_features': relevant_features[:20],  # Top 20 relevant features
        'enhanced_feature_importance': relevant_importance[:15],  # Top 15 with relevance
        'intent_aware_summary': summary,
        'analysis_focus': intent['focus_areas'],
        'key_concepts': intent['key_concepts']
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