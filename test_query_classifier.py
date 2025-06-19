#!/usr/bin/env python3
"""
Test script for the query classifier module.
This script tests the classifier with a variety of queries and prints the results.
"""

import json
import sys
from query_processing.classifier import process_query, QueryType

def colorize(text, color):
    """Add ANSI color codes to text for terminal output"""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'end': '\033[0m'
    }
    
    # Only colorize if output is to a terminal
    if sys.stdout.isatty():
        return f"{colors.get(color, '')}{text}{colors['end']}"
    return text

def format_confidence(confidence):
    """Format confidence score with appropriate color"""
    if confidence > 0.5:
        return colorize(f"{confidence:.2f}", 'green')
    elif confidence > 0.2:
        return colorize(f"{confidence:.2f}", 'yellow')
    else:
        return colorize(f"{confidence:.2f}", 'red')

def test_classifier():
    """Test the query classifier with various example queries"""
    
    test_cases = [
        # Correlation queries
        "What factors influence mortgage approvals?",
        "Which variables affect approval rates the most?",
        "What is the relationship between income and mortgage approvals?",
        "Key factors contributing to high approval rates",
        "What causes the variation in mortgage approval rates?",
        "How does income level affect the likelihood of mortgage approval?",
        "Is there a correlation between population density and mortgage approvals?",
        
        # Ranking queries
        "Show me the top 10 areas with highest approval rates",
        "Which regions have the lowest income levels?",
        "Rank areas by mortgage approval rates",
        "What are the top 5 areas for homeownership?",
        "List areas with highest rental rates",
        "Which cities have the highest mortgage approval rates?",
        "What are the 3 best performing regions for mortgage approvals?",
        
        # Comparison queries
        "Compare mortgage approvals between urban and rural areas",
        "How do high income areas compare to low income areas?",
        "What's the difference between young and senior populations in terms of approvals?",
        "Compare homeowners vs renters in approval rates",
        "Contrast approval rates in dense vs sparse population areas",
        "Are mortgage approvals higher in urban or rural regions?",
        "Show the difference in approval rates for areas above and below $50,000 income",
        
        # Trend queries
        "How have mortgage approvals changed over time?",
        "Show the trend in approval rates in high income areas",
        "Has homeownership increased over the past years?",
        "Track the evolution of approval rates since 2020",
        "Historical pattern of approvals in urban areas",
        "What's the mortgage approval trend since 2015?",
        "How have approval rates changed over the past decade?",
        
        # Geographic queries
        "Show me a map of mortgage approvals",
        "Geographic distribution of approval rates",
        "Where are approval rates highest?",
        "Visualize approvals by region",
        "Show approval patterns across different locations",
        "Create a heat map of mortgage approvals in Ontario",
        "Map the approval rates by province",
        
        # Mixed queries
        "Compare the trend of approval rates between urban and rural areas since 2020",
        "Show me a map of the top 5 regions with highest approval rates",
        "What factors influence approval rates in high income vs low income areas?",
        "How have the key drivers of mortgage approvals changed over time?",
        "Rank provinces by approval rate and show them on a map",
        
        # Complex and ambiguous queries
        "What areas have high approvals but low income?",
        "Is there a correlation between population density and approval rates?",
        "Why do some urban areas have low approval rates despite high income?",
        "Which demographic factors predict high approval rates?",
        "How does income above 75000 affect approvals?",
        "What's the relationship between unemployment and mortgage denials?"
    ]
    
    results = []
    
    print("\n===== ENHANCED QUERY CLASSIFIER TEST RESULTS =====\n")
    
    for query in test_cases:
        result = process_query(query)
        results.append({
            "query": query,
            "classified_as": result["analysis_type"],
            "confidence": round(result["confidence"], 2),
            "target": result.get("target_variable", "Not specified"),
            "parameters": {k: v for k, v in result.items() 
                          if k not in ["query_text", "analysis_type", "confidence", "target_variable"]}
        })
        
        # Print individual result with colored formatting
        query_type = result['analysis_type'].upper()
        confidence = result['confidence']
        
        # Color-code query types
        type_colors = {
            'CORRELATION': 'blue',
            'RANKING': 'green',
            'COMPARISON': 'purple',
            'TREND': 'cyan',
            'GEOGRAPHIC': 'yellow',
            'MIXED': 'white',
            'UNKNOWN': 'red'
        }
        
        colored_type = colorize(query_type, type_colors.get(query_type, 'white'))
        
        print(f"QUERY: {query}")
        print(f"  → Type: {colored_type} (confidence: {format_confidence(confidence)})")
        print(f"  → Target: {colorize(result.get('target_variable', 'Not specified'), 'green')}")
        
        # Print extracted parameters
        params = {k: v for k, v in result.items() 
                if k not in ["query_text", "analysis_type", "confidence", "target_variable"]}
        if params:
            print(f"  → Parameters: {json.dumps(params, indent=2)}")
        print()
    
    # Print summary statistics
    query_types = {}
    for r in results:
        query_type = r["classified_as"]
        query_types[query_type] = query_types.get(query_type, 0) + 1
    
    print("\n===== CLASSIFICATION SUMMARY =====")
    total = len(results)
    for query_type, count in sorted(query_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        color = type_colors.get(query_type.upper(), 'white')
        print(f"{colorize(query_type.upper(), color)}: {count} queries ({percentage:.1f}%)")
    
    # Calculate confidence statistics
    confidences = [r["confidence"] for r in results]
    avg_confidence = sum(confidences) / len(confidences)
    high_confidence = sum(1 for c in confidences if c > 0.5)
    medium_confidence = sum(1 for c in confidences if 0.2 < c <= 0.5)
    low_confidence = sum(1 for c in confidences if c <= 0.2)
    
    print(f"\nAverage confidence: {format_confidence(avg_confidence)}")
    print(f"High confidence (>0.5): {high_confidence} queries ({high_confidence/total*100:.1f}%)")
    print(f"Medium confidence (0.2-0.5): {medium_confidence} queries ({medium_confidence/total*100:.1f}%)")
    print(f"Low confidence (≤0.2): {low_confidence} queries ({low_confidence/total*100:.1f}%)")
    
    # Check for unknowns
    unknowns = [r["query"] for r in results if r["classified_as"] == "unknown"]
    if unknowns:
        print(f"\n{colorize('Unclassified Queries:', 'red')}")
        for query in unknowns:
            print(f"  - {query}")
    
    print("\nNOTE: Please review the classifications manually to identify any misclassified queries.")

if __name__ == "__main__":
    test_classifier()
