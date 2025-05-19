#!/usr/bin/env python3
"""
Search script to find all occurrences of deprecated NumPy APIs in the SHAP library
"""
import os
import re
import sys
from pathlib import Path

# Path to the SHAP library in the virtual environment
VENV_PATH = Path("/Users/voldeck/code/shap-microservice/venv/lib/python3.13/site-packages/shap")

def search_file(file_path, patterns):
    """Search for patterns in a file and return matches with line numbers"""
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                for pattern, pattern_name in patterns:
                    if re.search(pattern, line):
                        results.append((pattern_name, i, line.strip()))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return results

def search_directory(directory, patterns):
    """Search all Python files in a directory tree for the patterns"""
    all_results = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                results = search_file(file_path, patterns)
                if results:
                    rel_path = os.path.relpath(file_path, directory)
                    all_results[rel_path] = results
    return all_results

def main():
    patterns = [
        (r'np\.bool8', 'bool8'),
        (r'numpy\.bool8', 'bool8'),
        (r'np\.obj2sctype', 'obj2sctype'),
        (r'numpy\.obj2sctype', 'obj2sctype')
    ]
    
    print(f"Searching for deprecated NumPy APIs in {VENV_PATH}...")
    results = search_directory(VENV_PATH, patterns)
    
    if not results:
        print("No matches found.")
        return
    
    print(f"\nFound {sum(len(v) for v in results.values())} matches in {len(results)} files:")
    for file_path, file_results in results.items():
        print(f"\n{file_path}:")
        for pattern_name, line_num, line in file_results:
            print(f"  Line {line_num} ({pattern_name}): {line}")

if __name__ == "__main__":
    main()
