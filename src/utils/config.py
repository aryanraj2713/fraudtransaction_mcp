import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Logfire Configuration
    LOGFIRE_TOKEN: str = os.getenv("LOGFIRE_TOKEN", "")
    
    # MCP Server Configuration
    MCP_SERVER_HOST: str = os.getenv("MCP_SERVER_HOST", "localhost")
    MCP_SERVER_PORT_DATA: int = int(os.getenv("MCP_SERVER_PORT_DATA", "8001"))
    MCP_SERVER_PORT_MODEL: int = int(os.getenv("MCP_SERVER_PORT_MODEL", "8002"))
    MCP_SERVER_PORT_DECISION: int = int(os.getenv("MCP_SERVER_PORT_DECISION", "8003"))
    MCP_SERVER_PORT_MONITORING: int = int(os.getenv("MCP_SERVER_PORT_MONITORING", "8004"))
    
    # Fraud Detection Configuration
    FRAUD_THRESHOLD: float = float(os.getenv("FRAUD_THRESHOLD", "0.7"))
    MODEL_UPDATE_INTERVAL: int = int(os.getenv("MODEL_UPDATE_INTERVAL", "3600"))
    
    # Vector Database Configuration
    VECTOR_DB_SIZE: int = int(os.getenv("VECTOR_DB_SIZE", "10000"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Performance Configuration
    MAX_PROCESSING_TIME_MS: float = float(os.getenv("MAX_PROCESSING_TIME_MS", "100"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "1000"))
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        required_fields = ["OPENAI_API_KEY"]
        missing_fields = []
        
        for field in required_fields:
            if not getattr(cls, field):
                missing_fields.append(field)
        
        if missing_fields:
            print(f"Missing required configuration: {', '.join(missing_fields)}")
            return False
        
        return True
    
    @classmethod
    def configure_logfire(cls) -> bool:
        """Configure Logfire for observability"""
        if cls.LOGFIRE_TOKEN:
            try:
                import logfire
                
                # Configure Logfire
                logfire.configure(
                    token=cls.LOGFIRE_TOKEN,
                    service_name="fraud-detection-system",
                    service_version="1.0.0",
                    environment="production" if os.getenv("ENVIRONMENT") == "production" else "development"
                )
                
                # Set up structured logging
                logfire_handler = logfire.LogfireLoggingHandler()
                
                # Configure root logger
                root_logger = logging.getLogger()
                root_logger.addHandler(logfire_handler)
                root_logger.setLevel(logging.INFO)
                
                # Configure specific loggers for our application
                for logger_name in ["src.agents", "src.web_interface", "src.fraud_detection_system"]:
                    logger = logging.getLogger(logger_name)
                    logger.addHandler(logfire_handler)
                    logger.setLevel(logging.INFO)
                
                print(f"✅ Logfire configured successfully for observability")
                return True
                
            except ImportError:
                print("⚠️  Logfire not installed, using standard logging")
                return False
            except Exception as e:
                print(f"❌ Failed to configure Logfire: {str(e)}")
                return False
        else:
            print("⚠️  LOGFIRE_TOKEN not set, using standard logging")
            return False

config = Config()