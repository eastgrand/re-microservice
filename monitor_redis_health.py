#!/usr/bin/env python3
"""
Redis Health Monitor for SHAP Microservice

This script continuously monitors the Redis connection health
and reports any issues detected. It's designed to run in the
background and provide alerts for Redis connection problems.

Usage:
  python3 monitor_redis_health.py

Options:
  --interval SECONDS   Monitoring interval in seconds (default: 60)
  --verbose            Enable verbose output
  --alert              Enable email/webhook alerts
"""

import os
import sys
import time
import json
import argparse
import urllib.request
import logging
from datetime import datetime
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("redis_monitor.log")
    ]
)
logger = logging.getLogger("redis-monitor")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Redis Health Monitor for SHAP Microservice")
    parser.add_argument("--interval", type=int, default=60, help="Monitoring interval in seconds (default: 60)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--alert", action="store_true", help="Enable alerts")
    parser.add_argument("--webhook", type=str, help="Webhook URL for alerts")
    return parser.parse_args()

def check_redis_health(base_url, api_key):
    """Check Redis health using the service endpoints"""
    results = {}
    
    # Check endpoints that provide Redis health information
    endpoints = {
        "ping": "/admin/redis_ping",
        "health": "/health"
    }
    
    for name, path in endpoints.items():
        url = urljoin(base_url, path)
        req = urllib.request.Request(
            url, 
            headers={
                "x-api-key": api_key,
                "User-Agent": "Redis-Health-Monitor/1.0"
            }
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    content = response.read().decode("utf-8")
                    results[name] = {
                        "success": True,
                        "data": json.loads(content)
                    }
                else:
                    results[name] = {
                        "success": False,
                        "status": response.status,
                        "error": f"HTTP status: {response.status}"
                    }
        except Exception as e:
            results[name] = {
                "success": False,
                "error": str(e)
            }
    
    # Determine overall Redis health
    redis_connected = False
    if results.get("ping", {}).get("success") and results["ping"]["data"].get("success"):
        redis_connected = True
    elif results.get("health", {}).get("success") and results["health"]["data"].get("redis_connected"):
        redis_connected = True
    
    return {
        "timestamp": datetime.now().isoformat(),
        "redis_connected": redis_connected,
        "details": results
    }

def send_alert(message, webhook_url=None):
    """Send an alert about Redis health issues"""
    logger.warning(f"ALERT: {message}")
    
    if webhook_url:
        try:
            data = json.dumps({
                "text": f"SHAP Microservice Redis Alert: {message}",
                "timestamp": datetime.now().isoformat()
            }).encode("utf-8")
            
            req = urllib.request.Request(
                webhook_url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Redis-Health-Monitor/1.0"
                },
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status != 200:
                    logger.error(f"Failed to send webhook alert: HTTP {response.status}")
                else:
                    logger.info("Alert webhook sent successfully")
        except Exception as e:
            logger.error(f"Error sending webhook alert: {str(e)}")

def main():
    """Main monitoring function"""
    args = parse_arguments()
    
    # Set log level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get configuration from environment or use defaults
    base_url = os.environ.get("SHAP_SERVICE_URL", "https://nesto-mortgage-analytics.onrender.com")
    api_key = os.environ.get("API_KEY", "HFqkccbN3LV5CaB")
    webhook_url = args.webhook or os.environ.get("ALERT_WEBHOOK_URL")
    
    logger.info(f"Starting Redis health monitor (checking every {args.interval} seconds)")
    logger.info(f"Monitoring service at: {base_url}")
    
    # Track consecutive failures
    consecutive_failures = 0
    max_failures_before_alert = 3
    last_alert_time = None
    alert_cooldown_seconds = 300  # 5 minutes between alerts
    
    try:
        while True:
            check_time = datetime.now()
            
            try:
                health_result = check_redis_health(base_url, api_key)
                
                if args.verbose or not health_result["redis_connected"]:
                    if health_result["redis_connected"]:
                        logger.debug(f"Redis health check: CONNECTED")
                        consecutive_failures = 0
                    else:
                        logger.warning(f"Redis health check: DISCONNECTED")
                        consecutive_failures += 1
                        
                        # Send alert if we've had multiple failures
                        if (args.alert and consecutive_failures >= max_failures_before_alert and
                            (last_alert_time is None or 
                             (datetime.now() - last_alert_time).total_seconds() > alert_cooldown_seconds)):
                            alert_message = f"Redis connection is down ({consecutive_failures} consecutive failures)"
                            send_alert(alert_message, webhook_url)
                            last_alert_time = datetime.now()
                
                # Every 10 checks, log the full result regardless of status
                if datetime.now().minute % 10 == 0:
                    logger.info(f"Status report: Redis connected: {health_result['redis_connected']}")
            except Exception as e:
                logger.error(f"Error during health check: {str(e)}")
                consecutive_failures += 1
            
            # Calculate sleep time to maintain consistent interval
            elapsed_seconds = (datetime.now() - check_time).total_seconds()
            sleep_time = max(1, args.interval - elapsed_seconds)
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        logger.info("Redis health monitor stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
