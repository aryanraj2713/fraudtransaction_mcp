import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime
from openai import AsyncOpenAI

from ..utils.config import config

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all fraud detection agents"""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.knowledge_base: Dict[str, Any] = {}
        self.decision_history: List[Dict[str, Any]] = []
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.performance_metrics = {
            "total_decisions": 0,
            "correct_decisions": 0,
            "processing_time_ms": [],
            "confidence_scores": []
        }
        
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return agent's analysis/decision"""
        pass
    
    @abstractmethod
    async def learn(self, feedback: Dict[str, Any]) -> None:
        """Learn from feedback to improve future decisions"""
        pass
    
    @abstractmethod
    async def get_reasoning(self, input_data: Dict[str, Any]) -> str:
        """Get human-readable reasoning for the agent's decision"""
        pass
    
    async def update_knowledge(self, new_knowledge: Dict[str, Any]) -> None:
        """Update the agent's knowledge base"""
        self.knowledge_base.update(new_knowledge)
        self.last_activity = datetime.utcnow()
        logger.info(f"Agent {self.agent_id} knowledge updated")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        accuracy = (self.performance_metrics["correct_decisions"] / 
                   max(1, self.performance_metrics["total_decisions"]))
        
        avg_processing_time = (sum(self.performance_metrics["processing_time_ms"]) / 
                              max(1, len(self.performance_metrics["processing_time_ms"])))
        
        avg_confidence = (sum(self.performance_metrics["confidence_scores"]) / 
                         max(1, len(self.performance_metrics["confidence_scores"])))
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "accuracy": accuracy,
            "total_decisions": self.performance_metrics["total_decisions"],
            "avg_processing_time_ms": avg_processing_time,
            "avg_confidence": avg_confidence,
            "uptime_hours": (datetime.utcnow() - self.created_at).total_seconds() / 3600,
            "last_activity": self.last_activity.isoformat()
        }
    
    async def record_decision(self, input_data: Dict[str, Any], output: Dict[str, Any], 
                            processing_time_ms: float, confidence: float) -> None:
        """Record a decision for performance tracking"""
        decision_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "input_hash": hash(str(input_data)),
            "output": output,
            "processing_time_ms": processing_time_ms,
            "confidence": confidence
        }
        
        self.decision_history.append(decision_record)
        self.performance_metrics["total_decisions"] += 1
        self.performance_metrics["processing_time_ms"].append(processing_time_ms)
        self.performance_metrics["confidence_scores"].append(confidence)
        self.last_activity = datetime.utcnow()
        
        # Keep only last 1000 decisions
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    async def record_feedback(self, decision_id: str, was_correct: bool) -> None:
        """Record feedback on a previous decision"""
        if was_correct:
            self.performance_metrics["correct_decisions"] += 1
        
        # Find and update the decision record
        for record in reversed(self.decision_history):
            if hash(str(record)) == hash(decision_id):  # Simple matching
                record["feedback"] = {
                    "was_correct": was_correct,
                    "feedback_timestamp": datetime.utcnow().isoformat()
                }
                break
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": "active",
            "knowledge_base_size": len(self.knowledge_base),
            "decision_history_size": len(self.decision_history),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat()
        }
    
    async def reset_agent(self) -> None:
        """Reset agent state (for testing/debugging)"""
        self.knowledge_base.clear()
        self.decision_history.clear()
        self.performance_metrics = {
            "total_decisions": 0,
            "correct_decisions": 0,
            "processing_time_ms": [],
            "confidence_scores": []
        }
        self.last_activity = datetime.utcnow()
        logger.info(f"Agent {self.agent_id} reset")
    
    async def _generate_ai_response(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Generate AI response using OpenAI (if available)"""
        if not self.client:
            return None
            
        try:
            response = await self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"AI response generation failed: {str(e)}")
            return None
    
    def _calculate_confidence(self, factors: Dict[str, float]) -> float:
        """Calculate confidence score based on various factors"""
        if not factors:
            return 0.5
        
        # Weighted average of confidence factors
        weights = {
            "data_quality": 0.3,
            "pattern_match": 0.25,
            "historical_accuracy": 0.2,
            "consensus": 0.15,
            "novelty": 0.1
        }
        
        weighted_sum = sum(weights.get(factor, 0.1) * score 
                          for factor, score in factors.items())
        total_weight = sum(weights.get(factor, 0.1) 
                          for factor in factors.keys())
        
        return min(1.0, max(0.0, weighted_sum / total_weight))
    
    def __str__(self) -> str:
        return f"{self.agent_type}Agent({self.agent_id})"
    
    def __repr__(self) -> str:
        return self.__str__()