#!/usr/bin/env python3
"""
Simple test to verify the fraud detection system works
"""

import asyncio
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_system():
    """Test basic system functionality"""
    try:
        logger.info("🧪 Testing Fraud Detection System...")
        
        # Import the system
        from src.fraud_detection_system import get_fraud_detection_system
        from src.utils.synthetic_data import generate_sample_transaction
        
        # Initialize system
        logger.info("Initializing system...")
        system = await get_fraud_detection_system()
        
        # Generate test transaction
        logger.info("Generating test transaction...")
        test_transaction = generate_sample_transaction(fraud=False)
        
        # Process transaction
        logger.info("Processing transaction...")
        result = await system.process_transaction(test_transaction)
        
        # Check result
        if "error" in result:
            logger.error(f"❌ Transaction processing failed: {result['error']}")
            return False
        
        # Extract key metrics
        processing_time = result.get("processing_time_ms", 0)
        fraud_decision = result.get("fraud_decision", {})
        fraud_score = fraud_decision.get("fraud_score", 0)
        decision = fraud_decision.get("decision", "unknown")
        
        logger.info(f"✅ Transaction processed successfully!")
        logger.info(f"   Processing Time: {processing_time:.2f}ms")
        logger.info(f"   Fraud Score: {fraud_score:.3f}")
        logger.info(f"   Decision: {decision}")
        
        # Verify performance requirement
        if processing_time < 100:
            logger.info("✅ Sub-100ms requirement: PASSED")
        else:
            logger.warning("⚠️  Sub-100ms requirement: FAILED")
        
        # Test system status
        logger.info("Checking system status...")
        status = await system.get_system_status()
        
        if status["system_status"] == "running":
            logger.info("✅ System status: HEALTHY")
        else:
            logger.warning("⚠️  System status: UNHEALTHY")
        
        # Clean shutdown
        await system.shutdown()
        
        logger.info("🎉 System test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ System test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    success = await test_system()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())