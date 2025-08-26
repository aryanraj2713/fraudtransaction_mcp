# 🚀 Running the Fraud Detection System

This guide will help you get the fraud detection system up and running on your machine.

## 📋 Prerequisites

- Python 3.9 or higher
- macOS (tested on macOS 14.6.0)
- Homebrew (for installing OpenMP runtime)

## 🛠️ Setup Instructions

### 1. Install OpenMP Runtime (macOS only)

The system uses XGBoost which requires OpenMP runtime on macOS:

```bash
brew install libomp
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install the project in development mode
pip install -e .

# Install additional requirements
pip install -r requirements.txt

# Install missing dependencies
pip install aiofiles psutil
```

### 4. Set Environment Variables (macOS only)

```bash
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
```

## 🎯 Running the System

### Option 1: Use the Convenience Script (Recommended)

We've created a convenient script that handles all the setup automatically:

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the main fraud detection system
python run_fraud_system.py

# Or run individual MCP servers
python run_fraud_system.py --data-intel
python run_fraud_system.py --model-orch
python run_fraud_system.py --decision-engine
python run_fraud_system.py --monitoring

# Show help
python run_fraud_system.py --help
```

### Option 2: Run Directly

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Set environment variables
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

# Run the main system
python src/fraud_detection_system.py

# Or run individual MCP servers
python src/mcp_servers/data_intelligence_server.py
python src/mcp_servers/model_orchestration_server.py
python src/mcp_servers/decision_engine_server.py
python src/mcp_servers/monitoring_server.py
```

## 🔧 What Each Component Does

### Main Fraud Detection System
- Orchestrates all components
- Processes real-time transactions
- Coordinates between agents and MCP servers
- Provides system monitoring and status

### MCP Servers
- **Data Intelligence Server**: Handles data quality, drift detection, and feature engineering
- **Model Orchestration Server**: Manages ML models, training, and deployment
- **Decision Engine Server**: Makes fraud decisions using ensemble models
- **Monitoring Server**: Tracks system performance and generates alerts

### AI Agents
- **Pattern Recognition Agent**: Identifies suspicious patterns in transactions
- **Risk Assessment Agent**: Evaluates risk levels and generates scores
- **Coordination Agent**: Coordinates between different agents and components

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Fraud Detection System                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Transaction │  │   Pattern   │  │    Risk     │        │
│  │ Processor   │  │ Recognition │  │ Assessment  │        │
│  └─────────────┘  │   Agent     │  │   Agent     │        │
│                   └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    Data     │  │    Model    │  │  Decision   │        │
│  │Intelligence │  │Orchestration│  │   Engine    │        │
│  │   Server    │  │   Server    │  │   Server    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                 Monitoring Server                          │
└─────────────────────────────────────────────────────────────┘
```

## 🚨 Troubleshooting

### Common Issues

1. **XGBoost Import Error**: Make sure OpenMP is installed and environment variables are set
2. **Module Not Found**: Ensure virtual environment is activated and project is installed
3. **Permission Errors**: Make sure the run script is executable (`chmod +x run_fraud_system.py`)

### Getting Help

- Check the help: `python run_fraud_system.py --help`
- Verify virtual environment is activated: `which python` should point to `venv/bin/python`
- Check environment variables: `echo $LDFLAGS $CPPFLAGS`

## 🎉 Next Steps

Once the system is running, you can:

1. **Monitor the logs** to see transaction processing
2. **Send test transactions** to see fraud detection in action
3. **Explore individual components** by running specific MCP servers
4. **Customize the configuration** in the system initialization

## 📝 Notes

- The system runs asynchronously and handles multiple transactions concurrently
- Press `Ctrl+C` to gracefully stop the system
- All components are designed to be production-ready with proper error handling
- The system includes comprehensive monitoring and alerting capabilities
