#!/usr/bin/env python3
"""
Script to add MP field mapping to enhanced_analysis_worker.py
This maps MP field codes (like MP30034A_B) to their value_ prefixed versions in precalculated data
"""

def add_mp_mapping():
    # Read the original file
    with open('enhanced_analysis_worker.py', 'r') as f:
        content = f.read()
    
    # Find the insertion point after the conversion_rate mapping
    insertion_marker = '                target_variable = model_info[\'target\']\n        \n        features = model_info[\'features\']'
    
    if insertion_marker not in content:
        print("Could not find insertion point. Looking for alternative marker...")
        # Try alternative marker
        alt_marker = 'target_variable = model_info[\'target\']\n        \n        features'
        if alt_marker in content:
            insertion_marker = alt_marker
        else:
            print("Could not find insertion point. Showing context around conversion_rate:")
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'conversion_rate' in line and 'target_variable' in line:
                    for j in range(max(0, i-3), min(len(lines), i+8)):
                        print(f"{j+1:3}: {lines[j]}")
            return False
    
    # The MP field mapping code to insert
    mp_mapping_code = '''        elif target_variable and target_variable.startswith('MP') and '_' in target_variable:
            # Handle MP field codes (e.g., MP30034A_B) - map to value_ prefixed version
            value_target = f"value_{target_variable}"
            if value_target in precalc_df.columns:
                logger.info(f"Mapped MP field '{target_variable}' to actual column: '{value_target}'")
                target_variable = value_target
            else:
                logger.warning(f"MP field '{target_variable}' not found in data (tried '{value_target}'), using default: {model_info['target']}")
                target_variable = model_info['target']
        '''
    
    # Insert the MP mapping code
    new_content = content.replace(insertion_marker, mp_mapping_code + insertion_marker)
    
    # Check if the insertion was successful
    if new_content == content:
        print("No changes made - insertion point not found")
        return False
    
    # Write the updated content
    with open('enhanced_analysis_worker.py', 'w') as f:
        f.write(new_content)
    
    print("MP field mapping added successfully")
    return True

if __name__ == '__main__':
    add_mp_mapping()
