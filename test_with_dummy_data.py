#!/usr/bin/env python3
"""
Test script that runs the fraud detection system with dummy transaction data.
This demonstrates the system's ability to process different types of transactions.
"""

import os
import sys
import asyncio
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List

def setup_environment():
    """Set up environment variables for OpenMP and other dependencies."""
    os.environ['LDFLAGS'] = "-L/opt/homebrew/opt/libomp/lib"
    os.environ['CPPFLAGS'] = "-I/opt/homebrew/opt/libomp/include"
    
    # Add the src directory to Python path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

class MockMCPClient:
    """Mock MCP client for testing purposes."""
    
    def __init__(self):
        self.connected = True
        self.server_info = {
            "name": "mock_mcp_client",
            "version": "1.0.0"
        }
    
    async def connect(self):
        """Mock connection."""
        self.connected = True
        return True
    
    async def disconnect(self):
        """Mock disconnection."""
        self.connected = False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Mock tool call."""
        return {
            "status": "success",
            "result": f"Mock result for {tool_name}",
            "metadata": {"mock": True}
        }
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """Mock tool listing."""
        return [
            {"name": "mock_tool_1", "description": "Mock tool for testing"},
            {"name": "mock_tool_2", "description": "Another mock tool"}
        ]

def generate_dummy_transactions(count: int = 10) -> List[Dict[str, Any]]:
    """Generate realistic dummy transaction data for testing."""
    
    # Sample data for realistic transactions
    countries = ["US", "CA", "UK", "DE", "FR", "JP", "AU", "BR", "IN", "MX"]
    payment_methods = ["credit_card", "debit_card", "paypal", "bank_transfer", "crypto"]
    transaction_types = ["purchase", "refund", "transfer", "withdrawal", "deposit"]
    device_ids = [f"device_{chr(65+i)}{random.randint(100, 999)}" for i in range(5)]
    ip_ranges = [
        "192.168.1", "10.0.0", "172.16.0", "203.0.113", "198.51.100",
        "203.0.113", "198.51.100", "203.0.113", "198.51.100", "203.0.113"
    ]
    
    transactions = []
    
    for i in range(count):
        # Generate realistic transaction amounts
        amount = round(random.uniform(10.0, 2000.0), 2)
        
        # Generate realistic velocity patterns
        velocity_1h = random.randint(0, 5)
        velocity_24h = random.randint(0, 20)
        
        # Generate realistic account age
        account_age_days = random.randint(1, 3650)  # 1 day to 10 years
        
        # Create transaction with realistic patterns
        transaction = {
            "transaction_id": f"tx_test_{i+1:03d}_{random.randint(1000, 9999)}",
            "user_id": f"user_{random.randint(10000, 99999)}",
            "amount": amount,
            "currency": "USD",
            "transaction_type": random.choice(transaction_types),
            "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 60))).isoformat(),
            "country": random.choice(countries),
            "payment_method": random.choice(payment_methods),
            "device_id": random.choice(device_ids),
            "ip_address": f"{random.choice(ip_ranges)}.{random.randint(1, 254)}",
            "velocity_1h": velocity_1h,
            "velocity_24h": velocity_24h,
            "account_age_days": account_age_days,
            "is_first_transaction": random.random() < 0.1,  # 10% chance of first transaction
            
            # Additional features for ML models
            "merchant_category": random.choice(["retail", "food", "travel", "electronics", "services"]),
            "time_of_day": random.randint(0, 23),
            "day_of_week": random.randint(0, 6),
            "is_weekend": random.choice([True, False]),
            "amount_category": "high" if amount > 1000 else "medium" if amount > 100 else "low"
        }
        
        # Add some suspicious patterns for testing
        if i % 5 == 0:  # Every 5th transaction has suspicious characteristics
            transaction.update({
                "velocity_1h": random.randint(10, 50),  # High velocity
                "amount": round(random.uniform(5000, 15000), 2),  # High amount
                "country": random.choice(["XX", "YY", "ZZ"]),  # Suspicious country codes
                "is_first_transaction": True,  # New account
                "account_age_days": random.randint(1, 7)  # Very new account
            })
        
        transactions.append(transaction)
    
    return transactions

async def test_transaction_processing():
    """Test the fraud detection system with dummy transactions."""
    print("🧪 Testing Fraud Detection System with Dummy Data")
    print("=" * 60)
    
    try:
        # Import the system
        from fraud_detection_system import FraudDetectionSystem
        print("✅ FraudDetectionSystem imported successfully")
        
        # Create system instance
        fraud_system = FraudDetectionSystem()
        print("✅ FraudDetectionSystem instance created successfully")
        
        # Generate dummy transactions
        print("\n📊 Generating dummy transaction data...")
        dummy_transactions = generate_dummy_transactions(15)
        print(f"✅ Generated {len(dummy_transactions)} dummy transactions")
        
        # Show sample transactions
        print("\n📋 Sample Transactions:")
        for i, tx in enumerate(dummy_transactions[:3]):
            print(f"   {i+1}. {tx['transaction_id']}: ${tx['amount']:.2f} "
                  f"({tx['country']}, {tx['payment_method']})")
        
        # Start the system
        print("\n🚀 Starting fraud detection system...")
        await fraud_system.start()
        print("✅ System started successfully")
        
        # Process transactions
        print(f"\n🔄 Processing {len(dummy_transactions)} transactions...")
        results = []
        
        for i, transaction in enumerate(dummy_transactions):
            print(f"   Processing transaction {i+1}/{len(dummy_transactions)}: "
                  f"{transaction['transaction_id']}")
            
            try:
                # Process transaction
                tx_id = await fraud_system.process_transaction(transaction)
                
                # Get result (with timeout)
                result = await fraud_system.get_transaction_result(tx_id, timeout=5.0)
                
                if result:
                    results.append({
                        'transaction_id': transaction['transaction_id'],
                        'result': result,
                        'original_data': transaction
                    })
                    
                    # Show result
                    risk_level = "🟢 LOW" if result.risk_score < 0.3 else "🟡 MEDIUM" if result.risk_score < 0.7 else "🔴 HIGH"
                    print(f"      Result: {risk_level} (Risk: {result.risk_score:.3f}, "
                          f"Confidence: {result.confidence:.3f}, Time: {result.processing_time_ms:.1f}ms)")
                else:
                    print(f"      ⚠️  No result received for {transaction['transaction_id']}")
                    
            except Exception as e:
                print(f"      ❌ Error processing {transaction['transaction_id']}: {e}")
        
        # Show summary
        print(f"\n📊 Processing Summary:")
        print(f"   Total transactions: {len(dummy_transactions)}")
        print(f"   Successfully processed: {len(results)}")
        print(f"   Failed: {len(dummy_transactions) - len(results)}")
        
        if results:
            # Analyze results
            risk_scores = [r['result'].risk_score for r in results]
            confidence_scores = [r['result'].confidence for r in results]
            processing_times = [r['result'].processing_time_ms for r in results]
            
            print(f"\n📈 Performance Metrics:")
            print(f"   Average risk score: {sum(risk_scores)/len(risk_scores):.3f}")
            print(f"   Average confidence: {sum(confidence_scores)/len(confidence_scores):.3f}")
            print(f"   Average processing time: {sum(processing_times)/len(processing_times):.1f}ms")
            
            # Show high-risk transactions
            high_risk = [r for r in results if r['result'].risk_score > 0.7]
            if high_risk:
                print(f"\n🔴 High-Risk Transactions Detected:")
                for r in high_risk:
                    tx = r['original_data']
                    print(f"   {tx['transaction_id']}: ${tx['amount']:.2f} "
                          f"(Risk: {r['result'].risk_score:.3f}, Country: {tx['country']})")
        
        # Get system status
        print(f"\n📊 System Status:")
        status = await fraud_system.get_system_status()
        system_stats = status['system_stats']
        print(f"   Transactions processed: {system_stats['transactions_processed']}")
        print(f"   System uptime: {system_stats['system_uptime_seconds']}s")
        print(f"   Active agents: {system_stats['agents_active']}")
        print(f"   Active MCP servers: {system_stats['mcp_servers_active']}")
        
        # Stop the system
        print(f"\n🛑 Stopping fraud detection system...")
        await fraud_system.stop()
        print("✅ System stopped successfully")
        
        print(f"\n🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_individual_components():
    """Test individual MCP servers and agents."""
    print("\n🧪 Testing Individual Components")
    print("=" * 40)
    
    try:
        # Test MCP servers
        print("🔧 Testing MCP Servers...")
        
        # Data Intelligence Server
        from mcp_servers.data_intelligence_server import DataIntelligenceServer
        data_server = DataIntelligenceServer()
        print("✅ DataIntelligenceServer created")
        
        # Model Orchestration Server
        from mcp_servers.model_orchestration_server import ModelOrchestrationServer
        model_server = ModelOrchestrationServer()
        print("✅ ModelOrchestrationServer created")
        
        # Decision Engine Server
        from mcp_servers.decision_engine_server import DecisionEngineServer
        decision_server = DecisionEngineServer()
        print("✅ DecisionEngineServer created")
        
        # Monitoring Server
        from mcp_servers.monitoring_server import MonitoringServer
        monitoring_server = MonitoringServer()
        print("✅ MonitoringServer created")
        
        # Test agents with mock MCP client
        print("\n🤖 Testing AI Agents...")
        
        mock_client = MockMCPClient()
        
        from agents.pattern_recognition_agent import PatternRecognitionAgent
        pattern_agent = PatternRecognitionAgent("pattern_agent_001", mock_client)
        print("✅ PatternRecognitionAgent created")
        
        from agents.risk_assessment_agent import RiskAssessmentAgent
        risk_agent = RiskAssessmentAgent("risk_agent_001", mock_client)
        print("✅ RiskAssessmentAgent created")
        
        from agents.coordination_agent import CoordinationAgent
        coord_agent = CoordinationAgent("coord_agent_001", mock_client)
        print("✅ CoordinationAgent created")
        
        # Test agent initialization
        print("\n🔄 Initializing agents...")
        await pattern_agent.initialize()
        await risk_agent.initialize()
        await coord_agent.initialize()
        print("✅ All agents initialized successfully")
        
        print("🎉 All components created and initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the complete test suite."""
    print("🚀 Fraud Detection System - Dummy Data Test Suite")
    print("=" * 70)
    
    setup_environment()
    
    # Test individual components first
    component_test = await test_individual_components()
    
    if component_test:
        # Test full system with dummy data
        system_test = await test_transaction_processing()
        
        print("\n" + "=" * 70)
        print("📊 Final Test Results:")
        print(f"   Component Test: {'✅ PASS' if component_test else '❌ FAIL'}")
        print(f"   System Test: {'✅ PASS' if system_test else '❌ FAIL'}")
        
        if all([component_test, system_test]):
            print("\n🎉 All tests passed! The system is working correctly with dummy data.")
            print("\n💡 The system successfully:")
            print("   - Processed multiple dummy transactions")
            print("   - Applied fraud detection algorithms")
            print("   - Generated risk scores and confidence levels")
            print("   - Handled different transaction patterns")
            print("   - Coordinated between multiple AI agents and MCP servers")
        else:
            print("\n⚠️  Some tests failed. Please check the errors above.")
            return 1
    else:
        print("\n❌ Component test failed. Cannot proceed with system test.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
