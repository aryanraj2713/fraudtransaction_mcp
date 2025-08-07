#!/usr/bin/env python3
"""
Test script to verify Logfire integration is working
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def test_logfire_integration():
    """Test Logfire integration"""
    print("🔍 Testing Logfire Integration...")
    
    # Get token
    logfire_token = os.getenv("LOGFIRE_TOKEN")
    if not logfire_token:
        print("❌ LOGFIRE_TOKEN not found in environment")
        return False
    
    print(f"✅ LOGFIRE_TOKEN found: {logfire_token[:10]}...")
    
    try:
        import logfire
        print("✅ Logfire library imported successfully")
        
        # Configure Logfire
        logfire.configure(
            token=logfire_token,
            service_name="fraud-detection-test",
            service_version="1.0.0"
        )
        print("✅ Logfire configured successfully")
        
        # Create a test logger
        logger = logging.getLogger("fraud_detection_test")
        logfire_handler = logfire.LogfireLoggingHandler()
        logger.addHandler(logfire_handler)
        logger.setLevel(logging.INFO)
        
        # Send test log messages
        logger.info("🧪 Test log message from fraud detection system")
        logger.info(
            "💰 Test transaction processed", 
            extra={
                "transaction_id": "test_12345",
                "amount": 1000.00,
                "fraud_score": 0.75,
                "decision": "decline",
                "event": "test_fraud_analysis"
            }
        )
        
        print("✅ Test logs sent to Logfire successfully!")
        print("🌐 Check your Logfire dashboard at: https://logfire.pydantic.dev/")
        return True
        
    except ImportError:
        print("❌ Logfire library not installed")
        print("Run: pip install logfire")
        return False
    except Exception as e:
        print(f"❌ Error configuring Logfire: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_logfire_integration()
    if success:
        print("\n🎉 Logfire integration is working!")
        print("Your fraud detection system logs should now appear in Logfire.")
    else:
        print("\n💔 Logfire integration failed.")
        print("Please check your configuration and try again.")