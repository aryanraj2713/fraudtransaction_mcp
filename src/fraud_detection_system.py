import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import signal
import sys
import os

from core.mcp_client import MCPClient
from core.transaction_processor import RealTimeFraudProcessor, ProcessingResult
from mcp_servers.data_intelligence_server import DataIntelligenceServer
from mcp_servers.model_orchestration_server import ModelOrchestrationServer  
from mcp_servers.decision_engine_server import DecisionEngineServer
from mcp_servers.monitoring_server import MonitoringServer
from agents.pattern_recognition_agent import PatternRecognitionAgent
from agents.risk_assessment_agent import RiskAssessmentAgent
from agents.coordination_agent import CoordinationAgent

logger = logging.getLogger(__name__)


class FraudDetectionSystem:
    """Main fraud detection system orchestrator."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_default_config()
        
        # Core components
        self.mcp_client: Optional[MCPClient] = None
        self.transaction_processor: Optional[RealTimeFraudProcessor] = None
        
        # MCP Servers
        self.mcp_servers = {}
        
        # Agents
        self.agents = {}
        
        # System state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Performance tracking
        self.system_stats = {
            "transactions_processed": 0,
            "system_uptime_seconds": 0,
            "agents_active": 0,
            "mcp_servers_active": 0,
            "avg_processing_time_ms": 0.0
        }
        
        logger.info("Fraud Detection System initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default system configuration."""
        
        return {
            "mcp_servers": {
                "data_intelligence": {"host": "0.0.0.0", "port": 8001},
                "model_orchestration": {"host": "0.0.0.0", "port": 8002},
                "decision_engine": {"host": "0.0.0.0", "port": 8003},
                "monitoring": {"host": "0.0.0.0", "port": 8004}
            },
            "agents": {
                "pattern_recognition": {"instances": 2},
                "risk_assessment": {"instances": 2}, 
                "coordination": {"instances": 1}
            },
            "processing": {
                "max_concurrent_transactions": 1000,
                "processing_timeout_seconds": 30,
                "num_workers": 10
            },
            "performance": {
                "target_latency_ms": 100,
                "max_queue_size": 50000
            }
        }
    
    async def start(self):
        """Start the complete fraud detection system."""
        
        if self.is_running:
            logger.warning("System is already running")
            return
        
        try:
            self.start_time = datetime.now()
            logger.info("Starting Fraud Detection System...")
            
            # Step 1: Initialize MCP Client
            await self._initialize_mcp_client()
            
            # Step 2: Start MCP Servers
            await self._start_mcp_servers()
            
            # Step 3: Initialize and register agents
            await self._initialize_agents()
            
            # Step 4: Start real-time processor
            await self._start_transaction_processor()
            
            # Step 5: Register MCP servers with client
            await self._register_mcp_servers()
            
            self.is_running = True
            
            logger.info(
                f"Fraud Detection System started successfully! "
                f"Servers: {len(self.mcp_servers)}, Agents: {len(self.agents)}"
            )
            
            # Update system stats
            self.system_stats.update({
                "agents_active": len(self.agents),
                "mcp_servers_active": len(self.mcp_servers)
            })
            
        except Exception as e:
            logger.error(f"Failed to start Fraud Detection System: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the fraud detection system gracefully."""
        
        if not self.is_running:
            return
        
        logger.info("Stopping Fraud Detection System...")
        
        try:
            # Stop transaction processor
            if self.transaction_processor:
                await self.transaction_processor.stop()
            
            # Shutdown agents
            for agent in self.agents.values():
                await agent.shutdown()
            
            # Stop MCP servers (they handle their own cleanup)
            for server in self.mcp_servers.values():
                await server.stop()
            
            # Close MCP client
            if self.mcp_client:
                await self.mcp_client.__aexit__(None, None, None)
            
            self.is_running = False
            
            logger.info("Fraud Detection System stopped gracefully")
            
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")
    
    async def _initialize_mcp_client(self):
        """Initialize MCP client for server communication."""
        
        self.mcp_client = MCPClient(connection_pool_size=20)
        await self.mcp_client.__aenter__()
        
        logger.info("MCP Client initialized")
    
    async def _start_mcp_servers(self):
        """Start all MCP servers."""
        
        server_configs = self.config["mcp_servers"]
        
        # Start servers in parallel
        server_tasks = []
        
        # Data Intelligence Server
        if "data_intelligence" in server_configs:
            config = server_configs["data_intelligence"]
            server = DataIntelligenceServer(config["host"], config["port"])
            self.mcp_servers["data_intelligence"] = server
            server_tasks.append(asyncio.create_task(server.start()))
        
        # Model Orchestration Server  
        if "model_orchestration" in server_configs:
            config = server_configs["model_orchestration"]
            server = ModelOrchestrationServer(config["host"], config["port"])
            self.mcp_servers["model_orchestration"] = server
            server_tasks.append(asyncio.create_task(server.start()))
        
        # Decision Engine Server
        if "decision_engine" in server_configs:
            config = server_configs["decision_engine"]
            server = DecisionEngineServer(config["host"], config["port"])
            self.mcp_servers["decision_engine"] = server
            server_tasks.append(asyncio.create_task(server.start()))
        
        # Monitoring Server
        if "monitoring" in server_configs:
            config = server_configs["monitoring"]
            server = MonitoringServer(config["host"], config["port"])
            self.mcp_servers["monitoring"] = server
            server_tasks.append(asyncio.create_task(server.start()))
        
        # Give servers time to start
        await asyncio.sleep(2)
        
        logger.info(f"Started {len(self.mcp_servers)} MCP servers")
    
    async def _initialize_agents(self):
        """Initialize and register all fraud detection agents."""
        
        agent_configs = self.config["agents"]
        
        # Pattern Recognition Agents
        if "pattern_recognition" in agent_configs:
            instances = agent_configs["pattern_recognition"]["instances"]
            for i in range(instances):
                agent_id = f"pattern_recognition_agent_{i}"
                agent = PatternRecognitionAgent(agent_id, self.mcp_client)
                await agent.initialize()
                self.agents[agent_id] = agent
        
        # Risk Assessment Agents
        if "risk_assessment" in agent_configs:
            instances = agent_configs["risk_assessment"]["instances"]
            for i in range(instances):
                agent_id = f"risk_assessment_agent_{i}"
                agent = RiskAssessmentAgent(agent_id, self.mcp_client)
                await agent.initialize()
                self.agents[agent_id] = agent
        
        # Coordination Agent (usually just one)
        if "coordination" in agent_configs:
            instances = agent_configs["coordination"]["instances"]
            for i in range(instances):
                agent_id = f"coordination_agent_{i}"
                agent = CoordinationAgent(agent_id, self.mcp_client)
                
                # Register other agents with coordinator
                for other_agent in self.agents.values():
                    agent.register_agent(other_agent)
                
                await agent.initialize()
                self.agents[agent_id] = agent
        
        logger.info(f"Initialized {len(self.agents)} fraud detection agents")
    
    async def _start_transaction_processor(self):
        """Start the real-time transaction processor."""
        
        processing_config = self.config["processing"]
        
        self.transaction_processor = RealTimeFraudProcessor(
            mcp_client=self.mcp_client,
            max_concurrent_transactions=processing_config["max_concurrent_transactions"],
            processing_timeout_seconds=processing_config["processing_timeout_seconds"]
        )
        
        # Register agents with processor
        for agent in self.agents.values():
            self.transaction_processor.register_agent(agent)
        
        # Start processor
        await self.transaction_processor.start(
            num_workers=processing_config["num_workers"]
        )
        
        logger.info("Real-time transaction processor started")
    
    async def _register_mcp_servers(self):
        """Register MCP servers with the client."""
        
        server_configs = self.config["mcp_servers"]
        
        # Register each server
        for server_name, config in server_configs.items():
            endpoint = f"http://{config['host']}:{config['port']}"
            
            try:
                await self.mcp_client.register_server(
                    server_id=server_name,
                    endpoint=endpoint,
                    capabilities=["fraud_detection", "real_time_processing"],
                    tools=self._get_server_tools(server_name)
                )
                
                logger.info(f"Registered MCP server: {server_name} at {endpoint}")
                
            except Exception as e:
                logger.error(f"Failed to register MCP server {server_name}: {e}")
    
    def _get_server_tools(self, server_name: str) -> List[str]:
        """Get list of tools for a server."""
        
        server_tools = {
            "data_intelligence": [
                "ingest_transaction", "validate_data", 
                "engineer_features", "detect_drift"
            ],
            "model_orchestration": [
                "train_model", "predict_fraud", 
                "update_model_weights", "a_b_test_models"
            ],
            "decision_engine": [
                "assess_risk", "make_decision", 
                "generate_explanation", "escalate_for_review"
            ],
            "monitoring": [
                "log_performance_metric", "trigger_alert", 
                "health_check", "auto_scale_resources"
            ]
        }
        
        return server_tools.get(server_name, [])
    
    async def process_transaction(self, transaction: Dict[str, Any], priority: str = "normal") -> str:
        """Process a transaction through the fraud detection system."""
        
        if not self.is_running:
            raise RuntimeError("Fraud Detection System is not running")
        
        if not self.transaction_processor:
            raise RuntimeError("Transaction processor not available")
        
        # Add system metadata
        transaction.update({
            "system_received_at": datetime.now().isoformat(),
            "system_version": "1.0.0"
        })
        
        # Queue for processing
        transaction_id = await self.transaction_processor.process_transaction_stream(
            transaction, priority
        )
        
        # Update stats
        self.system_stats["transactions_processed"] += 1
        
        logger.debug(f"Queued transaction {transaction_id} for processing")
        
        return transaction_id
    
    async def get_transaction_result(self, transaction_id: str, timeout: float = 30.0) -> Optional[ProcessingResult]:
        """Get the result of a processed transaction."""
        
        if not self.transaction_processor:
            return None
        
        return await self.transaction_processor.get_processing_result(transaction_id, timeout)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        # Update uptime
        if self.start_time:
            self.system_stats["system_uptime_seconds"] = int(
                (datetime.now() - self.start_time).total_seconds()
            )
        
        status = {
            "system_status": "running" if self.is_running else "stopped",
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "system_stats": self.system_stats.copy(),
            "mcp_servers": {},
            "agents": {},
            "transaction_processor": {}
        }
        
        # Get MCP server health
        if self.mcp_client:
            status["mcp_servers"] = await self.mcp_client.get_all_servers_health()
        
        # Get agent status
        for agent_id, agent in self.agents.items():
            status["agents"][agent_id] = agent.get_agent_status()
        
        # Get transaction processor status
        if self.transaction_processor:
            status["transaction_processor"] = self.transaction_processor.get_system_status()
            
            # Update average processing time
            processor_stats = status["transaction_processor"].get("performance_metrics", {})
            self.system_stats["avg_processing_time_ms"] = processor_stats.get("avg_processing_time_ms", 0.0)
        
        return status
    
    async def train_model(self, training_data: List[Dict[str, Any]], model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a new fraud detection model."""
        
        if not self.mcp_client:
            raise RuntimeError("MCP client not available")
        
        try:
            result = await self.mcp_client.call_tool(
                "model_orchestration",
                "train_model",
                {
                    "model_config": model_config,
                    "training_data": training_data
                }
            )
            
            logger.info(f"Model training initiated: {result.get('model_id', 'unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_fraud_explanation(self, transaction_id: str, explanation_type: str = "business") -> Dict[str, Any]:
        """Get fraud detection explanation for a transaction."""
        
        # This would typically retrieve the decision from storage
        # For now, we'll return a placeholder
        return {
            "transaction_id": transaction_id,
            "explanation_type": explanation_type,
            "explanation": "Fraud risk assessment completed using multi-agent analysis",
            "generated_at": datetime.now().isoformat()
        }


async def main():
    """Main entry point for the fraud detection system."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create system instance
    fraud_system = FraudDetectionSystem()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(fraud_system.stop())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the system
        await fraud_system.start()
        
        # Keep the system running
        logger.info("Fraud Detection System is running. Press Ctrl+C to stop.")
        
        # Example transaction processing
        sample_transaction = {
            "transaction_id": "tx_demo_001",
            "user_id": "user_12345",
            "amount": 299.99,
            "currency": "USD",
            "transaction_type": "purchase",
            "timestamp": datetime.now().isoformat(),
            "country": "US",
            "payment_method": "credit_card",
            "device_id": "device_abc123",
            "ip_address": "192.168.1.100",
            "velocity_1h": 1,
            "velocity_24h": 3,
            "account_age_days": 90,
            "is_first_transaction": False
        }
        
        # Process demo transaction
        logger.info("Processing demo transaction...")
        tx_id = await fraud_system.process_transaction(sample_transaction)
        
        # Get result
        result = await fraud_system.get_transaction_result(tx_id, timeout=10.0)
        if result:
            logger.info(
                f"Demo transaction result: {result.decision} "
                f"(risk: {result.risk_score:.3f}, confidence: {result.confidence:.3f}, "
                f"time: {result.processing_time_ms:.1f}ms)"
            )
        
        # Monitor system status
        while True:
            await asyncio.sleep(30)  # Status check every 30 seconds
            
            status = await fraud_system.get_system_status()
            logger.info(
                f"System Status - Processed: {status['system_stats']['transactions_processed']}, "
                f"Uptime: {status['system_stats']['system_uptime_seconds']}s, "
                f"Avg Time: {status['system_stats']['avg_processing_time_ms']:.1f}ms"
            )
    
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    
    except Exception as e:
        logger.error(f"System error: {e}")
    
    finally:
        await fraud_system.stop()


if __name__ == "__main__":
    asyncio.run(main())