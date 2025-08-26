#!/usr/bin/env python3
"""
Test script to verify the fraud detection system can start up correctly.
"""

import os
import sys
import asyncio

def setup_environment():
    """Set up environment variables for OpenMP and other dependencies."""
    os.environ['LDFLAGS'] = "-L/opt/homebrew/opt/libomp/lib"
    os.environ['CPPFLAGS'] = "-I/opt/homebrew/opt/libomp/include"
    
    # Add the src directory to Python path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

async def test_system_startup():
    """Test if the system can start up correctly."""
    print("🧪 Testing Fraud Detection System Startup...")
    
    try:
        # Import the system
        from fraud_detection_system import FraudDetectionSystem
        print("✅ FraudDetectionSystem imported successfully")
        
        # Create system instance
        fraud_system = FraudDetectionSystem()
        print("✅ FraudDetectionSystem instance created successfully")
        
        # Test configuration loading
        config = fraud_system.config
        print(f"✅ Configuration loaded: {len(config)} sections")
        
        # Test MCP servers configuration
        mcp_servers = config.get('mcp_servers', {})
        print(f"✅ MCP Servers configured: {list(mcp_servers.keys())}")
        
        # Test agents configuration
        agents = config.get('agents', {})
        print(f"✅ Agents configured: {list(agents.keys())}")
        
        # Test processing configuration
        processing = config.get('processing', {})
        print(f"✅ Processing configured: {len(processing)} parameters")
        
        print("\n🎉 All tests passed! The system is ready to run.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mcp_servers():
    """Test if MCP servers can be imported."""
    print("\n🧪 Testing MCP Server Imports...")
    
    servers = [
        ("data_intelligence_server", "DataIntelligenceServer"),
        ("model_orchestration_server", "ModelOrchestrationServer"),
        ("decision_engine_server", "DecisionEngineServer"),
        ("monitoring_server", "MonitoringServer")
    ]
    
    all_passed = True
    
    for server_file, class_name in servers:
        try:
            module = __import__(f"mcp_servers.{server_file}", fromlist=[class_name])
            server_class = getattr(module, class_name)
            print(f"✅ {class_name} imported successfully")
        except Exception as e:
            print(f"❌ Failed to import {class_name}: {e}")
            all_passed = False
    
    if all_passed:
        print("🎉 All MCP servers imported successfully!")
    else:
        print("⚠️  Some MCP servers failed to import")
    
    return all_passed

async def test_agents():
    """Test if agents can be imported."""
    print("\n🧪 Testing Agent Imports...")
    
    agents = [
        ("pattern_recognition_agent", "PatternRecognitionAgent"),
        ("risk_assessment_agent", "RiskAssessmentAgent"),
        ("coordination_agent", "CoordinationAgent")
    ]
    
    all_passed = True
    
    for agent_file, class_name in agents:
        try:
            module = __import__(f"agents.{agent_file}", fromlist=[class_name])
            agent_class = getattr(module, class_name)
            print(f"✅ {class_name} imported successfully")
        except Exception as e:
            print(f"❌ Failed to import {class_name}: {e}")
            all_passed = False
    
    if all_passed:
        print("🎉 All agents imported successfully!")
    else:
        print("⚠️  Some agents failed to import")
    
    return all_passed

async def main():
    """Run all tests."""
    print("🚀 Fraud Detection System Test Suite")
    print("=" * 50)
    
    setup_environment()
    
    # Run tests
    system_test = await test_system_startup()
    mcp_test = await test_mcp_servers()
    agent_test = await test_agents()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   System Startup: {'✅ PASS' if system_test else '❌ FAIL'}")
    print(f"   MCP Servers: {'✅ PASS' if mcp_test else '❌ FAIL'}")
    print(f"   Agents: {'✅ PASS' if agent_test else '❌ FAIL'}")
    
    if all([system_test, mcp_test, agent_test]):
        print("\n🎉 All tests passed! The system is ready to run.")
        print("\n💡 To run the system:")
        print("   python run_fraud_system.py")
        print("\n💡 To run individual components:")
        print("   python run_fraud_system.py --data-intel")
        print("   python run_fraud_system.py --monitoring")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
