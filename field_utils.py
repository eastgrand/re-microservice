"""
Field resolution utilities for the SHAP microservice.
Provides consistent field name resolution across all endpoints.
"""

import re
import logging

logger = logging.getLogger(__name__)

def resolve_field_name(field_name, available_columns, master_schema=None):
    """
    Resolve field names from aliases and common variations.
    
    Args:
        field_name (str): The field name to resolve
        available_columns (list): List of available column names in the dataset
        master_schema (dict): Optional master schema for additional aliases
    
    Returns:
        str: The resolved field name
    """
    logger.info(f"Attempting to resolve field: '{field_name}'")
    
    if field_name in available_columns:
        logger.info(f"Field '{field_name}' found directly in available columns")
        return field_name
    
    # Handle common field aliases that the frontend might use
    common_aliases = {
        'household_average_income': 'median_income',
        'household_income': 'median_income',
        'average_income': 'median_income',
        'income': 'median_income',
        'household_median_income': 'median_income',
        'disposable_household_income': 'disposable_income',
        'mortgage_approval': 'mortgage_approvals',
        'mortgage_approval_rate': 'mortgage_approvals',
        'approval_rate': 'mortgage_approvals',
    }
    
    if field_name.lower() in common_aliases:
        resolved_field = common_aliases[field_name.lower()]
        logger.info(f"Resolved common alias: '{field_name}' -> '{resolved_field}'")
        return resolved_field
    
    # Dynamic field resolution for all fields
    field_lower = field_name.lower()
    logger.info(f"Searching for field with lowercase: '{field_lower}'")
    
    # First, try exact match in MASTER_SCHEMA aliases (if provided)
    if master_schema:
        for canonical_name, details in master_schema.items():
            if field_lower in [alias.lower() for alias in details.get('aliases', [])]:
                logger.info(f"Found field '{field_name}' in MASTER_SCHEMA aliases, mapping to '{details['raw_mapping']}'")
                return details['raw_mapping']
    
    # If not found in aliases, try pattern matching for all fields
    for col in available_columns:
        col_lower = col.lower()
        
        # Direct match
        if field_lower == col_lower:
            logger.info(f"Direct match found: '{field_name}' -> '{col}'")
            return col
        
        # Try snake_case conversion of the column name (preserve hyphens as underscores)
        col_snake = re.sub(r'[^\w\s-]', '', col_lower)  # Keep hyphens
        col_snake = re.sub(r'[-\s]+', '_', col_snake.strip())  # Convert hyphens and spaces to underscores
        if field_lower == col_snake:
            logger.info(f"Snake case match found: '{field_name}' -> '{col}' (snake: '{col_snake}')")
            return col
        
        # Try without year prefixes
        col_clean = re.sub(r'\b(2024|2023|2022|2021|census|current|year|\$|%)\b', '', col_lower)
        col_clean = re.sub(r'[^\w\s-]', '', col_clean)  # Keep hyphens
        col_clean = re.sub(r'[-\s]+', '_', col_clean.strip())  # Convert hyphens and spaces to underscores
        if field_lower == col_clean:
            logger.info(f"Clean match found: '{field_name}' -> '{col}' (clean: '{col_clean}')")
            return col
        
        # For percentage fields, try with _pct suffix
        if field_name.endswith('_pct') and '(%)' in col:
            base_field = field_name[:-4]  # Remove _pct
            col_base = col.replace('(%)', '').strip()
            col_base_snake = re.sub(r'[^\w\s-]', '', col_base.lower())  # Keep hyphens
            col_base_snake = re.sub(r'[-\s]+', '_', col_base_snake.strip())  # Convert hyphens and spaces to underscores
            
            if base_field == col_base_snake:
                logger.info(f"Percentage match found: '{field_name}' -> '{col}' (base: '{col_base_snake}')")
                return col
            
            # Try without year prefixes
            col_base_clean = re.sub(r'\b(2024|2023|2022|2021)\b', '', col_base).strip()
            if col_base_clean:
                col_base_clean_snake = re.sub(r'[^\w\s-]', '', col_base_clean.lower())  # Keep hyphens
                col_base_clean_snake = re.sub(r'[-\s]+', '_', col_base_clean_snake.strip())  # Convert hyphens and spaces to underscores
                if base_field == col_base_clean_snake:
                    logger.info(f"Clean percentage match found: '{field_name}' -> '{col}' (clean base: '{col_base_clean_snake}')")
                    return col
                
                # Try partial matching - check if the base field is contained in the clean snake case
                if base_field in col_base_clean_snake:
                    # Additional validation: check if key terms match
                    base_terms = set(base_field.split('_'))
                    col_terms = set(col_base_clean_snake.split('_'))
                    if base_terms.issubset(col_terms):
                        logger.info(f"Partial percentage match found: '{field_name}' -> '{col}' (partial: '{base_field}' in '{col_base_clean_snake}')")
                        return col
        
        # Try partial matching for key terms
        field_terms = set(re.findall(r'\w+', field_lower))
        col_terms = set(re.findall(r'\w+', col_lower))
        
        # If field has specific housing/demographic terms, check for matches
        key_terms = {'single', 'detached', 'house', 'apartment', 'condominium', 'condo', 
                    'visible', 'minority', 'income', 'population', 'structure', 'type'}
        
        if field_terms & key_terms:  # If field contains key terms
            # Check if most important terms match
            important_field_terms = field_terms & key_terms
            important_col_terms = col_terms & key_terms
            
            if important_field_terms and important_field_terms.issubset(important_col_terms):
                # Additional check for percentage fields
                if (field_name.endswith('_pct') and '(%)' in col) or (not field_name.endswith('_pct') and '(%)' not in col):
                    logger.info(f"Key terms match found: '{field_name}' -> '{col}' (terms: {important_field_terms})")
                    return col
    
    logger.warning(f"No match found for field: '{field_name}'")
    return field_name 