#!/usr/bin/env python3
"""
Simple script to run the fraud detection system with proper environment setup.
"""

import os
import sys
import asyncio
import subprocess

def setup_environment():
    """Set up environment variables for OpenMP and other dependencies."""
    os.environ['LDFLAGS'] = "-L/opt/homebrew/opt/libomp/lib"
    os.environ['CPPFLAGS'] = "-I/opt/homebrew/opt/libomp/include"
    
    # Add the src directory to Python path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

def run_fraud_system():
    """Run the main fraud detection system."""
    print("🚀 Starting Fraud Detection System...")
    print("📁 Setting up environment...")
    
    setup_environment()
    
    try:
        # Import and run the main system
        from fraud_detection_system import main
        
        print("✅ System imported successfully")
        print("🔄 Starting system...")
        print("💡 Press Ctrl+C to stop the system")
        print("-" * 50)
        
        # Run the main function
        asyncio.run(main())
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the virtual environment: source venv/bin/activate")
        return 1
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
        return 0
    except Exception as e:
        print(f"❌ System error: {e}")
        return 1

def run_mcp_server(server_name):
    """Run a specific MCP server."""
    print(f"🚀 Starting {server_name} MCP Server...")
    print("📁 Setting up environment...")
    
    setup_environment()
    
    try:
        if server_name == "data_intelligence":
            from mcp_servers.data_intelligence_server import main
        elif server_name == "model_orchestration":
            from mcp_servers.model_orchestration_server import main
        elif server_name == "decision_engine":
            from mcp_servers.decision_engine_server import main
        elif server_name == "monitoring":
            from mcp_servers.monitoring_server import main
        else:
            print(f"❌ Unknown server: {server_name}")
            return 1
        
        print(f"✅ {server_name} server imported successfully")
        print("🔄 Starting server...")
        print("💡 Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Run the server
        asyncio.run(main())
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the virtual environment: source venv/bin/activate")
        return 1
    except KeyboardInterrupt:
        print(f"\n🛑 {server_name} server stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Server error: {e}")
        return 1

def show_help():
    """Show help information."""
    print("""
🔍 Fraud Detection System Runner

Usage:
  python run_fraud_system.py [option]

Options:
  --help, -h          Show this help message
  --system, -s        Run the main fraud detection system
  --data-intel        Run the Data Intelligence MCP Server
  --model-orch        Run the Model Orchestration MCP Server
  --decision-engine   Run the Decision Engine MCP Server
  --monitoring        Run the Monitoring MCP Server

Examples:
  python run_fraud_system.py --system
  python run_fraud_system.py --data-intel
  python run_fraud_system.py --monitoring

Default: Runs the main fraud detection system
""")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ['--help', '-h']:
            show_help()
        elif arg in ['--data-intel']:
            run_mcp_server("data_intelligence")
        elif arg in ['--model-orch']:
            run_mcp_server("model_orchestration")
        elif arg in ['--decision-engine']:
            run_mcp_server("decision_engine")
        elif arg in ['--monitoring']:
            run_mcp_server("monitoring")
        elif arg in ['--system', '-s']:
            run_fraud_system()
        else:
            print(f"❌ Unknown option: {arg}")
            show_help()
            sys.exit(1)
    else:
        # Default: run the main system
        run_fraud_system()
