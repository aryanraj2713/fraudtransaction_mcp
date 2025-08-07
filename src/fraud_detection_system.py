import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import uuid

from .servers.data_intelligence_server import DataIntelligenceServer
from .servers.model_orchestration_server import ModelOrchestrationServer
from .servers.decision_engine_server import DecisionEngineServer
from .servers.monitoring_server import MonitoringResponseServer

from .agents.pattern_recognition_agent import PatternRecognitionAgent
from .agents.risk_assessment_agent import RiskAssessmentAgent
from .agents.coordination_agent import CoordinationAgent

from .schemas.transaction_schema import TransactionSchema, FraudScoreSchema
from .utils.config import config
from .utils.vector_db import InMemoryVectorDB

logger = logging.getLogger(__name__)

class FraudDetectionSystem:
    """Main fraud detection system that orchestrates MCP servers and AI agents"""
    
    def __init__(self):
        # Initialize MCP servers
        self.data_server = DataIntelligenceServer()
        self.model_server = ModelOrchestrationServer()
        self.decision_server = DecisionEngineServer()
        self.monitoring_server = MonitoringResponseServer()
        
        # Initialize AI agents
        self.pattern_agent = PatternRecognitionAgent()
        self.risk_agent = RiskAssessmentAgent()
        self.coordination_agent = CoordinationAgent()
        
        # System state
        self.is_running = False
        self.transaction_count = 0
        self.system_metrics = {
            "total_transactions": 0,
            "fraud_detected": 0,
            "false_positives": 0,
            "avg_processing_time": 0.0,
            "system_uptime": 0.0
        }
        self.start_time = None
        
    async def initialize(self) -> None:
        """Initialize the fraud detection system"""
        try:
            logger.info("Initializing Fraud Detection System...")
            
            # Register agents with coordination agent
            await self.coordination_agent.register_agent(self.pattern_agent)
            await self.coordination_agent.register_agent(self.risk_agent)
            
            # Initialize default models if needed
            await self._initialize_default_models()
            
            # Start monitoring
            self.start_time = datetime.utcnow()
            self.is_running = True
            
            logger.info("Fraud Detection System initialized successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            raise
    
    async def process_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single transaction through the fraud detection pipeline"""
        if not self.is_running:
            raise RuntimeError("System not initialized")
        
        start_time = datetime.utcnow()
        transaction_id = transaction_data.get("transaction_id", str(uuid.uuid4()))
        
        try:
            logger.info(f"Processing transaction {transaction_id}")
            
            # Step 1: Data ingestion and validation
            data_result = await self._process_data_intelligence(transaction_data)
            
            # Step 2: Multi-agent analysis
            agent_result = await self._process_agent_coordination(transaction_data, data_result)
            
            # Step 3: Decision engine processing
            decision_result = await self._process_decision_engine(transaction_data, agent_result)
            
            # Step 4: Update monitoring metrics
            await self._update_monitoring_metrics(transaction_data, decision_result, start_time)
            
            # Compile final result
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            final_result = {
                "transaction_id": transaction_id,
                "timestamp": start_time.isoformat(),
                "processing_time_ms": processing_time,
                "data_intelligence": data_result,
                "agent_analysis": agent_result,
                "fraud_decision": decision_result,
                "system_status": "success"
            }
            
            self.transaction_count += 1
            self.system_metrics["total_transactions"] += 1
            
            # Update average processing time
            current_avg = self.system_metrics["avg_processing_time"]
            self.system_metrics["avg_processing_time"] = (
                (current_avg * (self.transaction_count - 1) + processing_time) / self.transaction_count
            )
            
            logger.info(f"Transaction {transaction_id} processed in {processing_time:.2f}ms")
            return final_result
            
        except Exception as e:
            logger.error(f"Transaction processing failed for {transaction_id}: {str(e)}")
            return {
                "transaction_id": transaction_id,
                "timestamp": start_time.isoformat(),
                "error": str(e),
                "system_status": "error"
            }
    
    async def _process_data_intelligence(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process transaction through data intelligence server"""
        results = {}
        
        # Ingest transaction
        ingest_result = await self.data_server.server.call_tool("ingest_transaction", transaction_data)
        results["ingestion"] = ingest_result
        
        # Validate data quality
        validate_result = await self.data_server.server.call_tool("validate_data", transaction_data)
        results["validation"] = validate_result
        
        # Engineer features
        features_result = await self.data_server.server.call_tool("engineer_features", transaction_data)
        results["feature_engineering"] = features_result
        
        return results
    
    async def _process_agent_coordination(self, transaction_data: Dict[str, Any], data_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process transaction through coordinated AI agents"""
        
        # Prepare input for agents
        agent_input = {
            "transaction": transaction_data,
            "data_intelligence": data_result,
            "task_type": "fraud_detection"
        }
        
        # Get historical data for pattern analysis
        historical_data = await self._get_historical_data(transaction_data.get("user_id"))
        if historical_data:
            agent_input["historical_data"] = historical_data
        
        # Get user profile for behavioral analysis
        user_profile = await self._get_user_profile(transaction_data.get("user_id"))
        if user_profile:
            agent_input["user_profile"] = user_profile
        
        # Process through coordination agent
        coordination_result = await self.coordination_agent.process(agent_input)
        
        return coordination_result
    
    async def _process_decision_engine(self, transaction_data: Dict[str, Any], agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process through decision engine for final fraud determination"""
        
        # Extract fraud score from agent analysis
        coordinated_decision = agent_result.get("coordinated_decision", {})
        fraud_score = coordinated_decision.get("fraud_score", 0.5)
        
        # Enhance transaction data with agent insights
        enhanced_transaction = {
            **transaction_data,
            "agent_fraud_score": fraud_score,
            "agent_confidence": coordinated_decision.get("confidence", 0.5),
            "agent_decision": coordinated_decision.get("decision", "review")
        }
        
        # Score transaction through decision engine
        scoring_result = await self.decision_server.server.call_tool("score_transaction", enhanced_transaction)
        
        # Perform risk assessment if needed
        if fraud_score > 0.6:
            risk_data = {
                "transaction": enhanced_transaction,
                "context": {
                    "agent_analysis": agent_result,
                    "fraud_score": fraud_score
                }
            }
            risk_result = await self.decision_server.server.call_tool("assess_risk", risk_data)
            scoring_result["risk_assessment"] = risk_result
        
        # Generate rules if new patterns detected
        if agent_result.get("agent_results", {}).get("pattern_recognition_001", {}).get("new_patterns_discovered", 0) > 0:
            pattern_data = {
                "patterns": agent_result["agent_results"]["pattern_recognition_001"].get("new_patterns", []),
                "confidence_threshold": 0.8
            }
            rules_result = await self.decision_server.server.call_tool("generate_rules", pattern_data)
            scoring_result["rule_generation"] = rules_result
        
        # Escalate if high risk
        if fraud_score > 0.8:
            escalation_data = {
                "transaction_id": transaction_data.get("transaction_id"),
                "reason": f"High fraud score: {fraud_score:.3f}",
                "priority": "high" if fraud_score > 0.9 else "medium"
            }
            escalation_result = await self.decision_server.server.call_tool("escalate_case", escalation_data)
            scoring_result["escalation"] = escalation_result
        
        return scoring_result
    
    async def _update_monitoring_metrics(self, transaction_data: Dict[str, Any], decision_result: Dict[str, Any], start_time: datetime) -> None:
        """Update system monitoring metrics"""
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Extract decision information
        fraud_decision = decision_result.get("decision", "approve")
        fraud_score = decision_result.get("fraud_score", 0.0)
        
        # Prepare monitoring data
        metrics_data = {
            "processing_times": [processing_time],
            "accuracy_data": {
                "fraud_detection_accuracy": 0.85,  # Would be calculated from feedback
                "precision": 0.82,
                "recall": 0.78,
                "f1_score": 0.80,
                "false_positive_rate": 0.05
            },
            "system_metrics": {
                "cpu_usage": 0.45,  # Would be actual system metrics
                "memory_usage": 0.62,
                "requests_per_second": self.transaction_count / max(1, (datetime.utcnow() - self.start_time).total_seconds()),
                "error_rate": 0.02,
                "availability": 1.0
            },
            "business_metrics": {
                "transactions_processed": 1,
                "fraud_detected": 1 if fraud_decision == "decline" else 0,
                "fraud_prevented_amount": transaction_data.get("amount", 0) if fraud_decision == "decline" else 0,
                "false_positives": 0  # Would be determined from feedback
            }
        }
        
        # Send to monitoring server
        await self.monitoring_server.server.call_tool("monitor_performance", metrics_data)
        
        # Update local metrics
        if fraud_decision == "decline":
            self.system_metrics["fraud_detected"] += 1
    
    async def _get_historical_data(self, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get historical transaction data for user (simulated)"""
        if not user_id:
            return None
        
        # In production, this would query a database
        # For demo, return simulated data
        import random
        
        historical = []
        for i in range(random.randint(5, 20)):
            historical.append({
                "transaction_id": f"hist_{user_id}_{i}",
                "user_id": user_id,
                "amount": random.uniform(10, 1000),
                "timestamp": (datetime.utcnow() - timedelta(days=random.randint(1, 30))).isoformat(),
                "merchant_category": random.choice(["grocery", "gas", "restaurant", "online"]),
                "location": {"country": "US"},
                "status": "approved"
            })
        
        return historical
    
    async def _get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile data (simulated)"""
        if not user_id:
            return None
        
        # In production, this would query user profile service
        # For demo, return simulated profile
        import random
        
        return {
            "user_id": user_id,
            "avg_transaction_amount": random.uniform(50, 500),
            "frequent_categories": random.sample(["grocery", "gas", "restaurant", "online", "retail"], 3),
            "typical_hours": list(range(8, 20)),  # Business hours
            "home_country": "US",
            "account_age_days": random.randint(30, 1000),
            "typically_has_device_info": True
        }
    
    async def _initialize_default_models(self) -> None:
        """Initialize default ML models"""
        try:
            # Train a default fraud detection model
            model_config = {
                "model_id": "default_fraud_model",
                "model_type": "random_forest",
                "parameters": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42
                }
            }
            
            await self.model_server.server.call_tool("train_model", model_config)
            
            # Deploy the model
            deployment_config = {
                "model_id": "default_fraud_model",
                "strategy": "replace"
            }
            
            await self.model_server.server.call_tool("deploy_model", deployment_config)
            
            logger.info("Default models initialized")
            
        except Exception as e:
            logger.warning(f"Default model initialization failed: {str(e)}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        
        status = {
            "system_status": "running" if self.is_running else "stopped",
            "uptime_seconds": uptime_seconds,
            "transactions_processed": self.transaction_count,
            "metrics": self.system_metrics,
            "servers": {
                "data_intelligence": self.data_server.get_metrics(),
                "model_orchestration": self.model_server.get_metrics(),
                "decision_engine": self.decision_server.get_metrics(),
                "monitoring": self.monitoring_server.get_metrics()
            },
            "agents": {
                "pattern_recognition": await self.pattern_agent.get_performance_metrics(),
                "risk_assessment": await self.risk_agent.get_performance_metrics(),
                "coordination": await self.coordination_agent.get_performance_metrics()
            }
        }
        
        return status
    
    async def process_feedback(self, feedback: Dict[str, Any]) -> None:
        """Process feedback to improve system performance"""
        try:
            transaction_id = feedback.get("transaction_id")
            actual_fraud = feedback.get("actual_fraud", False)
            
            logger.info(f"Processing feedback for transaction {transaction_id}: fraud={actual_fraud}")
            
            # Send feedback to all components
            await self.coordination_agent.learn(feedback)
            await self.pattern_agent.learn(feedback)
            await self.risk_agent.learn(feedback)
            
            # Update system metrics
            predicted_fraud = feedback.get("predicted_decision") == "decline"
            if predicted_fraud and not actual_fraud:
                self.system_metrics["false_positives"] += 1
            
        except Exception as e:
            logger.error(f"Feedback processing failed: {str(e)}")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the system"""
        logger.info("Shutting down Fraud Detection System...")
        self.is_running = False
        
        # Save any persistent state if needed
        # Close connections, cleanup resources
        
        logger.info("Fraud Detection System shutdown complete")

# Global system instance
fraud_detection_system = None

async def get_fraud_detection_system() -> FraudDetectionSystem:
    """Get or create the global fraud detection system instance"""
    global fraud_detection_system
    
    if fraud_detection_system is None:
        fraud_detection_system = FraudDetectionSystem()
        await fraud_detection_system.initialize()
    
    return fraud_detection_system