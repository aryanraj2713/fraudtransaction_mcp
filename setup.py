from setuptools import setup, find_packages

setup(
    name="fraud-detection-mcp-system",
    version="1.0.0",
    description="Production-ready agentic AI fraud detection system with MCP integration",
    author="AI Fraud Detection Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "mcp>=1.0.0",
        "pydantic>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "openai>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "redis>=5.0.0",
        "aiohttp>=3.8.0",
        "sqlalchemy>=2.0.0",
        "prometheus-client>=0.17.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.9.0",
            "isort>=5.12.0",
            "mypy>=1.6.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "lightgbm>=4.0.0",
        ],
        "monitoring": [
            "opentelemetry-api>=1.20.0",
            "opentelemetry-sdk>=1.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fraud-detection-server=api.fraud_detection_api:main",
            "mcp-server=mcp_servers.data_intelligence_server:main",
        ],
    },
)