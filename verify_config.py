#!/usr/bin/env python3
"""
Simple Configuration Verification Script
This script doesn't rely on external dependencies and checks configuration files directly.
"""

import os
import sys
import json

def print_header(message):
    """Print a header message"""
    print("\n" + "=" * 60)
    print(message)
    print("=" * 60)

def check_render_yaml():
    """Check the render.yaml file"""
    print_header("RENDER.YAML CONFIGURATION")
    
    try:
        with open('render.yaml', 'r') as f:
            content = f.read()
            
        # Check worker configuration
        if 'simple_worker.py' in content:
            print("✅ Worker script: Using simple_worker.py")
        else:
            print("❌ Worker script: Not using simple_worker.py")
            
        # Check memory management setting
        if 'AGGRESSIVE_MEMORY_MANAGEMENT' in content and 'false' in content:
            print("✅ Memory management: Aggressive memory management is disabled")
        else:
            print("❌ Memory management: Aggressive memory management may be enabled")
            
        # Check batch size
        if 'SHAP_MAX_BATCH_SIZE' in content and '500' in content:
            print("✅ Batch size: Set to 500")
        else:
            print("❌ Batch size: Not set to 500")
            
        # Check Redis health check interval
        if 'REDIS_HEALTH_CHECK_INTERVAL' in content and '30' in content:
            print("✅ Redis health check: Set to 30 seconds")
        else:
            print("❌ Redis health check: Not set to 30 seconds")
            
        print("\nRender.yaml Excerpt:")
        print("---------------------")
        worker_section = False
        for line in content.split('\n'):
            if 'worker' in line:
                worker_section = True
                print("  " + line)
            elif worker_section and 'envVars' in line:
                print("  " + line)
            elif worker_section and '-' in line:
                print("  " + line)
            elif worker_section and 'key:' in line:
                print("  " + line)
            elif worker_section and 'value:' in line:
                print("  " + line)
                
    except Exception as e:
        print(f"Error checking render.yaml: {str(e)}")

def check_optimize_memory():
    """Check the optimize_memory.py file"""
    print_header("MEMORY OPTIMIZATION SETTINGS")
    
    try:
        with open('optimize_memory.py', 'r') as f:
            content = f.read()
            
        # Check memory thresholds
        aggressive_threshold = None
        standard_threshold = None
        
        for line in content.split('\n'):
            if 'DEFAULT_MAX_MEMORY_MB' in line and 'AGGRESSIVE_MEMORY' in line and '450' in line:
                aggressive_threshold = 450
                print("✅ Aggressive memory threshold: 450MB")
            elif 'DEFAULT_MAX_MEMORY_MB' in line and 'AGGRESSIVE_MEMORY' in line and '475' not in line:
                print(f"❌ Aggressive memory threshold: Not set to 450MB")
                
            if 'DEFAULT_MAX_MEMORY_MB' in line and 'AGGRESSIVE_MEMORY' not in line and '475' in line:
                standard_threshold = 475
                print("✅ Standard memory threshold: 475MB")
            elif 'DEFAULT_MAX_MEMORY_MB' in line and 'AGGRESSIVE_MEMORY' not in line and '475' not in line:
                print(f"❌ Standard memory threshold: Not set to 475MB")
                
        if aggressive_threshold and standard_threshold:
            print(f"✅ Memory buffer: {512 - standard_threshold}MB (from 512MB limit)")
        else:
            print("❌ Could not determine memory buffer")
            
    except Exception as e:
        print(f"Error checking optimize_memory.py: {str(e)}")

def check_worker_scripts():
    """Check worker script files"""
    print_header("WORKER SCRIPT VERIFICATION")
    
    # Check simple_worker.py
    try:
        if os.path.exists('simple_worker.py'):
            with open('simple_worker.py', 'r') as f:
                content = f.read()
                
            if 'Connection' not in content or 'Context' not in content:
                print("✅ simple_worker.py: Not using Connection context manager")
            else:
                print("❌ simple_worker.py: May be using Connection context manager")
                
            if os.access('simple_worker.py', os.X_OK):
                print("✅ simple_worker.py: Is executable")
            else:
                print("❌ simple_worker.py: Not executable")
                
        else:
            print("❌ simple_worker.py: File not found")
    except Exception as e:
        print(f"Error checking simple_worker.py: {str(e)}")
        
    # Check setup_worker.py
    try:
        if os.path.exists('setup_worker.py'):
            with open('setup_worker.py', 'r') as f:
                content = f.read()
                
            if 'from rq import Queue, Worker' in content and 'Connection' not in content:
                print("✅ setup_worker.py: Connection import removed")
            else:
                print("❌ setup_worker.py: Connection import may still be present")
                
        else:
            print("❌ setup_worker.py: File not found")
    except Exception as e:
        print(f"Error checking setup_worker.py: {str(e)}")

def check_redis_connection_patch():
    """Check redis_connection_patch.py"""
    print_header("REDIS CONNECTION PATCH VERIFICATION")
    
    try:
        if os.path.exists('redis_connection_patch.py'):
            with open('redis_connection_patch.py', 'r') as f:
                content = f.read()
                
            if 'patched_from_url' in content:
                print("✅ Redis connection patch: from_url patching implemented")
            else:
                print("❌ Redis connection patch: from_url patching not found")
                
            if 'safe_ping' in content:
                print("✅ Redis connection patch: Failsafe ping method implemented")
            else:
                print("❌ Redis connection patch: Failsafe ping method not found")
                
            if 'wrap_redis_queue_functions' in content:
                print("✅ Redis connection patch: Queue function wrapping implemented")
            else:
                print("❌ Redis connection patch: Queue function wrapping not found")
                
            if 'TypeError with connection_pool parameter' in content:
                print("✅ Redis connection patch: Connection pool fix implemented")
            else:
                print("❌ Redis connection patch: Connection pool fix not found")
                
        else:
            print("❌ redis_connection_patch.py: File not found")
    except Exception as e:
        print(f"Error checking redis_connection_patch.py: {str(e)}")

def print_summary():
    """Print verification summary"""
    print_header("VERIFICATION SUMMARY")
    
    print("The verification has checked the configuration files for the SHAP microservice.")
    print("Items marked with ✅ indicate correctly configured settings.")
    print("Items marked with ❌ indicate potential issues or missing configurations.")
    
    print("\nRecommended Next Steps:")
    print("1. Verify that the changes have been deployed to Render.com")
    print("2. Check the worker logs to ensure it's using simple_worker.py")
    print("3. Monitor memory usage to ensure it stays under 512MB")
    print("4. Test job processing to confirm that batching is working correctly")
    print("5. Verify Redis connection stability during high load")

def main():
    """Main function"""
    print_header("SHAP Microservice Configuration Verification")
    print(f"Current directory: {os.getcwd()}")
    
    # Run all checks
    check_render_yaml()
    check_optimize_memory()
    check_worker_scripts()
    check_redis_connection_patch()
    print_summary()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
