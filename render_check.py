#!/usr/bin/env python3
"""
Render port check helper script.
This script helps debug port binding issues with Render deployments.
"""

import os
import socket
import time
import sys

def check_port_binding():
    """Check if the port specified in the PORT environment variable is actively listening."""
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    
    print(f"Checking if port {port} is bound to {host}...")
    
    # Try to create a server socket to see if the port is already in use
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        print(f"Port {port} is available (not in use)")
        s.close()
        return False
    except socket.error:
        print(f"Port {port} is already in use - this is good if our app is running")
        s.close()
        return True
        
def check_service_responding():
    """Check if a service is responding on localhost at the PORT."""
    port = int(os.environ.get('PORT', 5000))
    
    print(f"Checking if service is responding on port {port}...")
    
    # Try to connect to the service
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)  # 5 second timeout
    try:
        s.connect(('localhost', port))
        print(f"Successfully connected to service on port {port}")
        s.close()
        return True
    except socket.error as e:
        print(f"Could not connect to service: {e}")
        s.close()
        return False

if __name__ == "__main__":
    print("Render Port Check Helper")
    print("------------------------")
    print(f"PORT environment variable: {os.environ.get('PORT', 'not set')}")
    
    is_bound = check_port_binding()
    is_responding = check_service_responding()
    
    if is_bound and is_responding:
        print("SUCCESS: Port is bound and service is responding")
        sys.exit(0)
    elif is_bound and not is_responding:
        print("WARNING: Port is bound but service is not responding")
        sys.exit(1)
    else:
        print("ERROR: Port is not bound")
        sys.exit(2)
