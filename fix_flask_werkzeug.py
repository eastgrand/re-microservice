#!/usr/bin/env python
"""
Flask-Werkzeug Compatibility Check

This script checks for compatibility between Flask and Werkzeug versions
and fixes potential issues by ensuring compatible versions are installed.
"""

import subprocess
import pkg_resources
import sys

def check_and_fix_werkzeug():
    """Check Flask and Werkzeug compatibility and fix if needed."""
    try:
        flask_version = pkg_resources.get_distribution('flask').version
        werkzeug_version = pkg_resources.get_distribution('werkzeug').version
        
        print(f"Detected Flask version: {flask_version}")
        print(f"Detected Werkzeug version: {werkzeug_version}")
        
        # Flask 2.2.x requires Werkzeug < 3.0.0
        if flask_version.startswith('2.2.') and werkzeug_version.startswith('3.'):
            print("Incompatible Flask and Werkzeug versions detected!")
            print("Flask 2.2.x requires Werkzeug < 3.0.0")
            
            # Install the correct Werkzeug version
            print("Installing compatible Werkzeug version (2.2.3)...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'werkzeug==2.2.3', '--force-reinstall'])
            
            werkzeug_version = pkg_resources.get_distribution('werkzeug').version
            print(f"Werkzeug version after fix: {werkzeug_version}")
            return True
        else:
            print("Flask and Werkzeug versions appear compatible.")
            return False
    except Exception as e:
        print(f"Error checking Flask/Werkzeug compatibility: {e}")
        return False

if __name__ == "__main__":
    check_and_fix_werkzeug()
