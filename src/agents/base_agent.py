import asyncio
import logging
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from core.mcp_client import MCPClient
from schemas.transaction_schema import Transaction, FraudScore

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class DecisionConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class AgentDecision:
    agent_id: str
    decision_id: str
    transaction_id: str
    decision_type: str
    confidence: float
    risk_score: float
    reasoning: List[str]
    evidence: Dict[str, Any]
    processing_time_ms: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class LearningFeedback:
    feedback_id: str
    transaction_id: str
    agent_decision: AgentDecision
    actual_outcome: str  # fraud, legitimate
    accuracy: float
    learning_points: Dict[str, Any]
    timestamp: datetime


@dataclass
class CollaborationMessage:
    message_id: str
    sender_agent_id: str
    recipient_agent_id: str
    message_type: str  # request_opinion, share_insight, consensus_vote
    content: Dict[str, Any]
    priority: str
    timestamp: datetime
    response_deadline: Optional[datetime] = None


class KnowledgeBase:
    """Agent's knowledge storage and management system."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.learned_insights: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.collaboration_history: List[CollaborationMessage] = []
    
    def add_pattern(self, pattern_name: str, pattern_data: Dict[str, Any]):
        """Add a new fraud pattern to knowledge base."""
        self.patterns[pattern_name] = {
            "data": pattern_data,
            "confidence": pattern_data.get("confidence", 0.5),
            "discovered_at": datetime.now().isoformat(),
            "usage_count": 0,
            "accuracy_rate": 0.0
        }
        logger.debug(f"Agent {self.agent_id} learned new pattern: {pattern_name}")
    
    def update_pattern_performance(self, pattern_name: str, accuracy: float):
        """Update pattern performance based on feedback."""
        if pattern_name in self.patterns:
            pattern = self.patterns[pattern_name]
            old_accuracy = pattern["accuracy_rate"]
            usage_count = pattern["usage_count"]
            
            # Running average
            new_accuracy = (old_accuracy * usage_count + accuracy) / (usage_count + 1)
            
            pattern["accuracy_rate"] = new_accuracy
            pattern["usage_count"] += 1
            
            logger.debug(f"Updated pattern {pattern_name} accuracy: {new_accuracy:.3f}")
    
    def get_relevant_patterns(self, transaction: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Get patterns relevant to a transaction."""
        relevant = []
        
        for pattern_name, pattern_info in self.patterns.items():
            relevance_score = self._calculate_pattern_relevance(transaction, pattern_info["data"])
            if relevance_score > 0.3:  # Threshold for relevance
                relevant.append((pattern_name, {**pattern_info, "relevance": relevance_score}))
        
        # Sort by relevance and accuracy
        relevant.sort(key=lambda x: x[1]["relevance"] * x[1]["accuracy_rate"], reverse=True)
        return relevant
    
    def _calculate_pattern_relevance(self, transaction: Dict[str, Any], pattern: Dict[str, Any]) -> float:
        """Calculate how relevant a pattern is to a transaction."""
        # Simplified relevance calculation
        relevance_factors = []
        
        # Amount similarity
        if "amount_range" in pattern and "amount" in transaction:
            amount = float(transaction["amount"])
            min_amount, max_amount = pattern["amount_range"]
            if min_amount <= amount <= max_amount:
                relevance_factors.append(1.0)
            else:
                distance = min(abs(amount - min_amount), abs(amount - max_amount))
                normalized_distance = distance / max(amount, max_amount)
                relevance_factors.append(max(0, 1 - normalized_distance))
        
        # Geographic match
        if "countries" in pattern and "country" in transaction:
            if transaction["country"] in pattern["countries"]:
                relevance_factors.append(1.0)
            else:
                relevance_factors.append(0.2)
        
        # Time pattern match
        if "time_patterns" in pattern and "timestamp" in transaction:
            hour = datetime.fromisoformat(transaction["timestamp"]).hour
            if hour in pattern.get("suspicious_hours", []):
                relevance_factors.append(0.8)
            else:
                relevance_factors.append(0.3)
        
        return np.mean(relevance_factors) if relevance_factors else 0.0
    
    def add_learning_insight(self, insight: Dict[str, Any]):
        """Add a new learning insight."""
        insight["timestamp"] = datetime.now().isoformat()
        insight["agent_id"] = self.agent_id
        self.learned_insights.append(insight)
        
        # Keep only recent insights (last 1000)
        if len(self.learned_insights) > 1000:
            self.learned_insights = self.learned_insights[-1000:]


class ReasoningEngine:
    """Advanced reasoning engine for autonomous decision making."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.reasoning_chains: List[Dict[str, Any]] = []
        self.decision_templates: Dict[str, Dict[str, Any]] = {}
        self._setup_default_templates()
    
    def _setup_default_templates(self):
        """Setup default reasoning templates."""
        self.decision_templates = {
            "high_risk_decline": {
                "conditions": ["risk_score > 0.8", "confidence > 0.7"],
                "decision": "decline",
                "reasoning_template": "High fraud risk detected: {risk_factors}. Confidence: {confidence:.2f}"
            },
            "moderate_risk_review": {
                "conditions": ["risk_score > 0.4", "risk_score <= 0.8"],
                "decision": "review",
                "reasoning_template": "Moderate risk indicators present: {risk_factors}. Requires manual review."
            },
            "low_risk_approve": {
                "conditions": ["risk_score <= 0.4", "confidence > 0.6"],
                "decision": "approve",
                "reasoning_template": "Low fraud risk. Protective factors: {protective_factors}"
            }
        }
    
    def reason_about_transaction(
        self, 
        transaction: Dict[str, Any], 
        analysis_results: Dict[str, Any],
        knowledge_base: KnowledgeBase
    ) -> AgentDecision:
        """Perform autonomous reasoning about a transaction."""
        
        start_time = datetime.now()
        transaction_id = transaction.get("transaction_id", "unknown")
        decision_id = f"decision_{uuid.uuid4().hex[:8]}"
        
        # Extract key information
        risk_score = analysis_results.get("risk_score", 0.5)
        confidence = analysis_results.get("confidence", 0.5)
        risk_factors = analysis_results.get("risk_factors", [])
        protective_factors = analysis_results.get("protective_factors", [])
        
        # Build reasoning chain
        reasoning_chain = self._build_reasoning_chain(
            transaction, analysis_results, knowledge_base
        )
        
        # Make decision using templates
        decision_type, reasoning = self._apply_decision_templates(
            risk_score, confidence, risk_factors, protective_factors
        )
        
        # Enhance reasoning with knowledge base insights
        kb_insights = self._get_knowledge_base_insights(transaction, knowledge_base)
        if kb_insights:
            reasoning.extend(kb_insights)
        
        # Calculate final confidence
        final_confidence = self._calculate_decision_confidence(
            risk_score, confidence, reasoning_chain, knowledge_base
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        decision = AgentDecision(
            agent_id=self.agent_id,
            decision_id=decision_id,
            transaction_id=transaction_id,
            decision_type=decision_type,
            confidence=final_confidence,
            risk_score=risk_score,
            reasoning=reasoning,
            evidence={
                "risk_factors": risk_factors,
                "protective_factors": protective_factors,
                "reasoning_chain": reasoning_chain,
                "knowledge_base_patterns_used": len(kb_insights),
                # Include any evidence from analysis results (like coordination workflow results)
                **analysis_results.get("evidence", {})
            },
            processing_time_ms=processing_time,
            timestamp=datetime.now(),
            metadata={
                "analysis_results": analysis_results,
                "decision_template_used": self._get_matching_template_name(risk_score, confidence)
            }
        )
        
        # Store reasoning chain
        self.reasoning_chains.append({
            "decision_id": decision_id,
            "chain": reasoning_chain,
            "timestamp": datetime.now().isoformat()
        })
        
        return decision
    
    def _build_reasoning_chain(
        self, 
        transaction: Dict[str, Any], 
        analysis_results: Dict[str, Any],
        knowledge_base: KnowledgeBase
    ) -> List[Dict[str, Any]]:
        """Build a detailed reasoning chain."""
        
        chain = []
        
        # Step 1: Transaction analysis
        chain.append({
            "step": "transaction_analysis",
            "description": "Analyzed transaction basic properties",
            "findings": {
                "amount": transaction.get("amount"),
                "transaction_type": transaction.get("transaction_type"),
                "country": transaction.get("country"),
                "is_first_transaction": transaction.get("is_first_transaction", False)
            }
        })
        
        # Step 2: Risk factor identification
        risk_factors = analysis_results.get("risk_factors", [])
        if risk_factors:
            chain.append({
                "step": "risk_factor_identification",
                "description": f"Identified {len(risk_factors)} risk factors",
                "findings": {"risk_factors": risk_factors}
            })
        
        # Step 3: Pattern matching
        relevant_patterns = knowledge_base.get_relevant_patterns(transaction)
        if relevant_patterns:
            chain.append({
                "step": "pattern_matching",
                "description": f"Matched {len(relevant_patterns)} known fraud patterns",
                "findings": {
                    "patterns": [p[0] for p in relevant_patterns[:3]],  # Top 3
                    "highest_relevance": relevant_patterns[0][1]["relevance"] if relevant_patterns else 0
                }
            })
        
        # Step 4: Confidence assessment
        chain.append({
            "step": "confidence_assessment",
            "description": "Assessed confidence in analysis",
            "findings": {
                "base_confidence": analysis_results.get("confidence", 0.5),
                "data_completeness": self._assess_data_completeness(transaction),
                "pattern_support": len(relevant_patterns) > 0
            }
        })
        
        return chain
    
    def _apply_decision_templates(
        self, 
        risk_score: float, 
        confidence: float, 
        risk_factors: List[str], 
        protective_factors: List[str]
    ) -> Tuple[str, List[str]]:
        """Apply decision templates to make a decision."""
        
        # Check each template
        for template_name, template in self.decision_templates.items():
            if self._evaluate_conditions(template["conditions"], risk_score, confidence):
                reasoning = [template["reasoning_template"].format(
                    risk_factors=", ".join(risk_factors[:3]) if risk_factors else "None identified",
                    protective_factors=", ".join(protective_factors[:3]) if protective_factors else "None identified",
                    confidence=confidence,
                    risk_score=risk_score
                )]
                return template["decision"], reasoning
        
        # Default fallback
        return "review", [f"Risk score {risk_score:.3f} requires manual assessment"]
    
    def _evaluate_conditions(self, conditions: List[str], risk_score: float, confidence: float) -> bool:
        """Evaluate template conditions."""
        for condition in conditions:
            # Simple condition evaluation (would be more sophisticated in practice)
            condition = condition.replace("risk_score", str(risk_score))
            condition = condition.replace("confidence", str(confidence))
            
            try:
                if not eval(condition):
                    return False
            except:
                return False
        
        return True
    
    def _get_knowledge_base_insights(self, transaction: Dict[str, Any], knowledge_base: KnowledgeBase) -> List[str]:
        """Get insights from knowledge base."""
        insights = []
        
        relevant_patterns = knowledge_base.get_relevant_patterns(transaction)
        for pattern_name, pattern_info in relevant_patterns[:2]:  # Top 2 patterns
            insights.append(
                f"Matches known pattern '{pattern_name}' with {pattern_info['relevance']:.2f} relevance "
                f"and {pattern_info['accuracy_rate']:.2f} historical accuracy"
            )
        
        return insights
    
    def _calculate_decision_confidence(
        self, 
        risk_score: float, 
        base_confidence: float, 
        reasoning_chain: List[Dict[str, Any]],
        knowledge_base: KnowledgeBase
    ) -> float:
        """Calculate final decision confidence."""
        
        confidence_factors = [base_confidence]
        
        # Data completeness factor
        data_completeness = self._assess_data_completeness_from_chain(reasoning_chain)
        confidence_factors.append(data_completeness)
        
        # Pattern support factor
        pattern_support = any(step["step"] == "pattern_matching" for step in reasoning_chain)
        if pattern_support:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # Risk score certainty (mid-range scores have lower confidence)
        if risk_score < 0.2 or risk_score > 0.8:
            confidence_factors.append(0.9)  # High certainty
        else:
            confidence_factors.append(0.6)  # Medium certainty
        
        return np.mean(confidence_factors)
    
    def _assess_data_completeness(self, transaction: Dict[str, Any]) -> float:
        """Assess completeness of transaction data."""
        required_fields = ["amount", "timestamp", "user_id", "country", "payment_method"]
        present_fields = sum(1 for field in required_fields if transaction.get(field))
        return present_fields / len(required_fields)
    
    def _assess_data_completeness_from_chain(self, reasoning_chain: List[Dict[str, Any]]) -> float:
        """Extract data completeness from reasoning chain."""
        for step in reasoning_chain:
            if step["step"] == "confidence_assessment":
                return step["findings"].get("data_completeness", 0.5)
        return 0.5
    
    def _get_matching_template_name(self, risk_score: float, confidence: float) -> str:
        """Get the name of the matching decision template."""
        for template_name, template in self.decision_templates.items():
            if self._evaluate_conditions(template["conditions"], risk_score, confidence):
                return template_name
        return "default_fallback"


class BaseFraudDetectionAgent(ABC):
    """Base class for all fraud detection agents with autonomous capabilities."""
    
    def __init__(self, agent_id: str, mcp_client: MCPClient, specialization: str):
        self.agent_id = agent_id
        self.mcp_client = mcp_client
        self.specialization = specialization
        self.status = AgentStatus.INITIALIZING
        
        # Core components
        self.knowledge_base = KnowledgeBase(agent_id)
        self.reasoning_engine = ReasoningEngine(agent_id)
        
        # Agent state
        self.decision_history: List[AgentDecision] = []
        self.learning_memory: List[LearningFeedback] = []
        self.collaboration_partners: Dict[str, str] = {}  # agent_id -> specialization
        
        # Performance metrics
        self.performance_metrics = {
            "decisions_made": 0,
            "accuracy_rate": 0.0,
            "avg_processing_time_ms": 0.0,
            "collaboration_requests": 0,
            "patterns_learned": 0
        }
        
        # Configuration
        self.config = {
            "max_decision_history": 10000,
            "learning_rate": 0.01,
            "collaboration_threshold": 0.3,  # When to ask for help
            "pattern_discovery_threshold": 0.7
        }
        
        logger.info(f"Initialized {specialization} agent: {agent_id}")
    
    async def initialize(self):
        """Initialize agent-specific components."""
        self.status = AgentStatus.ACTIVE
        await self._load_specialized_knowledge()
        logger.info(f"Agent {self.agent_id} ({self.specialization}) is now active")
    
    @abstractmethod
    async def _load_specialized_knowledge(self):
        """Load specialization-specific knowledge and patterns."""
        pass
    
    @abstractmethod
    async def analyze_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Perform specialized analysis on a transaction."""
        pass
    
    async def process_transaction(self, transaction: Dict[str, Any]) -> AgentDecision:
        """Main transaction processing workflow."""
        
        if self.status != AgentStatus.ACTIVE:
            raise RuntimeError(f"Agent {self.agent_id} is not active (status: {self.status.value})")
        
        self.status = AgentStatus.BUSY
        start_time = datetime.now()
        
        try:
            # Step 1: Perform specialized analysis
            analysis_results = await self.analyze_transaction(transaction)
            
            # Step 2: Check if collaboration is needed
            if self._should_collaborate(analysis_results):
                collaboration_insights = await self._request_collaboration(transaction, analysis_results)
                analysis_results = self._integrate_collaboration_insights(analysis_results, collaboration_insights)
            
            # Step 3: Make autonomous decision
            decision = self.reasoning_engine.reason_about_transaction(
                transaction, analysis_results, self.knowledge_base
            )
            
            # Step 4: Store decision and update metrics
            self.decision_history.append(decision)
            self._update_performance_metrics(decision, start_time)
            
            # Step 5: Check for pattern discovery
            await self._discover_patterns(transaction, analysis_results, decision)
            
            logger.info(
                f"Agent {self.agent_id} processed transaction {transaction.get('transaction_id', 'unknown')}: "
                f"{decision.decision_type} (confidence: {decision.confidence:.3f})"
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} error processing transaction: {e}")
            self.status = AgentStatus.ERROR
            raise
        
        finally:
            if self.status == AgentStatus.BUSY:
                self.status = AgentStatus.ACTIVE
    
    def _should_collaborate(self, analysis_results: Dict[str, Any]) -> bool:
        """Determine if collaboration with other agents is needed."""
        confidence = analysis_results.get("confidence", 1.0)
        risk_score = analysis_results.get("risk_score", 0.0)
        
        # Collaborate if confidence is low or risk is in uncertain range
        if confidence < self.config["collaboration_threshold"]:
            return True
        
        if 0.3 <= risk_score <= 0.7:  # Uncertain risk range
            return True
        
        return False
    
    async def _request_collaboration(
        self, 
        transaction: Dict[str, Any], 
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Request insights from other agents."""
        
        collaboration_results = {}
        
        for partner_id, partner_specialization in self.collaboration_partners.items():
            try:
                # Create collaboration message
                message = CollaborationMessage(
                    message_id=f"collab_{uuid.uuid4().hex[:8]}",
                    sender_agent_id=self.agent_id,
                    recipient_agent_id=partner_id,
                    message_type="request_opinion",
                    content={
                        "transaction": transaction,
                        "sender_analysis": analysis_results,
                        "question": f"What is your {partner_specialization} perspective on this transaction?"
                    },
                    priority="normal",
                    timestamp=datetime.now(),
                    response_deadline=datetime.now() + timedelta(seconds=5)
                )
                
                # This would be implemented with proper agent-to-agent communication
                # For now, we'll simulate a response
                simulated_response = await self._simulate_collaboration_response(
                    partner_specialization, transaction, analysis_results
                )
                
                collaboration_results[partner_id] = simulated_response
                self.performance_metrics["collaboration_requests"] += 1
                
            except Exception as e:
                logger.error(f"Collaboration with {partner_id} failed: {e}")
        
        return collaboration_results
    
    async def _simulate_collaboration_response(
        self, 
        partner_specialization: str, 
        transaction: Dict[str, Any], 
        analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate collaboration response (would be real agent communication)."""
        
        # Simulate different perspectives based on specialization
        if partner_specialization == "pattern_recognition":
            return {
                "opinion": "moderate_risk",
                "confidence": 0.7,
                "insights": ["Similar pattern seen in historical data", "Geographic location flags attention"],
                "risk_adjustment": 0.1
            }
        elif partner_specialization == "risk_assessment":
            return {
                "opinion": "high_risk",
                "confidence": 0.8,
                "insights": ["Velocity patterns concerning", "Amount outside normal range"],
                "risk_adjustment": 0.2
            }
        else:
            return {
                "opinion": "low_risk",
                "confidence": 0.6,
                "insights": ["No significant behavioral anomalies"],
                "risk_adjustment": -0.1
            }
    
    def _integrate_collaboration_insights(
        self, 
        original_analysis: Dict[str, Any], 
        collaboration_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Integrate insights from collaborating agents."""
        
        if not collaboration_insights:
            return original_analysis
        
        enhanced_analysis = original_analysis.copy()
        
        # Aggregate risk adjustments
        risk_adjustments = []
        all_insights = []
        confidence_scores = []
        
        for partner_id, insight in collaboration_insights.items():
            risk_adjustments.append(insight.get("risk_adjustment", 0))
            all_insights.extend(insight.get("insights", []))
            confidence_scores.append(insight.get("confidence", 0.5))
        
        # Apply weighted average of risk adjustments
        if risk_adjustments:
            avg_adjustment = np.mean(risk_adjustments)
            enhanced_analysis["risk_score"] = min(1.0, max(0.0, 
                enhanced_analysis.get("risk_score", 0.5) + avg_adjustment
            ))
        
        # Boost confidence if collaborators agree
        if confidence_scores:
            collaboration_confidence = np.mean(confidence_scores)
            original_confidence = enhanced_analysis.get("confidence", 0.5)
            enhanced_analysis["confidence"] = min(1.0, 
                (original_confidence + collaboration_confidence) / 2
            )
        
        # Add collaboration insights
        enhanced_analysis["collaboration_insights"] = all_insights
        enhanced_analysis["collaboration_partners"] = list(collaboration_insights.keys())
        
        return enhanced_analysis
    
    async def _discover_patterns(
        self, 
        transaction: Dict[str, Any], 
        analysis_results: Dict[str, Any], 
        decision: AgentDecision
    ):
        """Discover and learn new fraud patterns."""
        
        # Only discover patterns for high-confidence decisions
        if decision.confidence < self.config["pattern_discovery_threshold"]:
            return
        
        # Extract potential pattern
        if decision.risk_score > 0.7:  # High-risk transaction
            pattern = {
                "pattern_type": "high_risk",
                "amount_range": (float(transaction.get("amount", 0)) * 0.8, 
                               float(transaction.get("amount", 0)) * 1.2),
                "countries": [transaction.get("country", "")],
                "time_patterns": {"suspicious_hours": [datetime.fromisoformat(
                    transaction.get("timestamp", datetime.now().isoformat())
                ).hour]},
                "confidence": decision.confidence,
                "discovered_from_decision": decision.decision_id
            }
            
            pattern_name = f"auto_discovered_{uuid.uuid4().hex[:8]}"
            self.knowledge_base.add_pattern(pattern_name, pattern)
            self.performance_metrics["patterns_learned"] += 1
            
            logger.info(f"Agent {self.agent_id} discovered new pattern: {pattern_name}")
    
    def _update_performance_metrics(self, decision: AgentDecision, start_time: datetime):
        """Update agent performance metrics."""
        processing_time = decision.processing_time_ms
        
        # Update counters
        self.performance_metrics["decisions_made"] += 1
        
        # Update average processing time
        current_avg = self.performance_metrics["avg_processing_time_ms"]
        total_decisions = self.performance_metrics["decisions_made"]
        self.performance_metrics["avg_processing_time_ms"] = (
            (current_avg * (total_decisions - 1) + processing_time) / total_decisions
        )
        
        # Keep decision history manageable
        if len(self.decision_history) > self.config["max_decision_history"]:
            self.decision_history = self.decision_history[-self.config["max_decision_history"]:]
    
    async def learn_from_feedback(self, feedback: LearningFeedback):
        """Learn from feedback about previous decisions."""
        
        try:
            self.learning_memory.append(feedback)
            
            # Update accuracy rate
            correct_decisions = sum(1 for f in self.learning_memory if f.accuracy > 0.5)
            self.performance_metrics["accuracy_rate"] = correct_decisions / len(self.learning_memory)
            
            # Update pattern performance
            decision = feedback.agent_decision
            if "patterns_used" in decision.metadata:
                for pattern_name in decision.metadata["patterns_used"]:
                    self.knowledge_base.update_pattern_performance(pattern_name, feedback.accuracy)
            
            # Extract learning insights
            if feedback.accuracy < 0.5:  # Incorrect decision
                insight = {
                    "type": "error_analysis",
                    "transaction_id": feedback.transaction_id,
                    "error_reason": feedback.learning_points.get("error_reason", "Unknown"),
                    "correction": feedback.learning_points.get("correction", "Review analysis approach"),
                    "impact": "adjust_decision_thresholds"
                }
                self.knowledge_base.add_learning_insight(insight)
                
                logger.warning(f"Agent {self.agent_id} learned from error in transaction {feedback.transaction_id}")
            
            # Adjust decision thresholds based on feedback (simple adaptation)
            await self._adapt_decision_strategy(feedback)
            
        except Exception as e:
            logger.error(f"Learning from feedback error: {e}")
    
    async def _adapt_decision_strategy(self, feedback: LearningFeedback):
        """Adapt decision-making strategy based on feedback."""
        
        # Simple threshold adjustment based on feedback
        if feedback.accuracy < 0.5:  # Incorrect decision
            decision = feedback.agent_decision
            
            if decision.decision_type == "approve" and feedback.actual_outcome == "fraud":
                # False positive - was too lenient, increase sensitivity
                for template in self.reasoning_engine.decision_templates.values():
                    if "risk_score >" in str(template.get("conditions", [])):
                        # This is a simplified adjustment - would be more sophisticated
                        pass
            
            elif decision.decision_type == "decline" and feedback.actual_outcome == "legitimate":
                # False negative - was too strict, decrease sensitivity
                pass
    
    async def collaborate_with_agents(self, message: CollaborationMessage) -> Dict[str, Any]:
        """Handle collaboration requests from other agents."""
        
        try:
            if message.message_type == "request_opinion":
                # Provide opinion based on specialization
                transaction = message.content.get("transaction", {})
                sender_analysis = message.content.get("sender_analysis", {})
                
                # Perform our own analysis
                our_analysis = await self.analyze_transaction(transaction)
                
                # Provide opinion
                response = {
                    "responder_agent_id": self.agent_id,
                    "specialization": self.specialization,
                    "opinion": self._generate_opinion(our_analysis, sender_analysis),
                    "confidence": our_analysis.get("confidence", 0.5),
                    "insights": our_analysis.get("risk_factors", [])[:3],  # Top 3 insights
                    "risk_adjustment": self._calculate_risk_adjustment(our_analysis, sender_analysis)
                }
                
                return response
                
            elif message.message_type == "share_insight":
                # Learn from shared insight
                insight = message.content.get("insight", {})
                self.knowledge_base.add_learning_insight(insight)
                
                return {"status": "insight_received", "agent_id": self.agent_id}
            
            else:
                return {"status": "unsupported_message_type", "agent_id": self.agent_id}
                
        except Exception as e:
            logger.error(f"Collaboration handling error: {e}")
            return {"status": "error", "error": str(e), "agent_id": self.agent_id}
    
    def _generate_opinion(self, our_analysis: Dict[str, Any], sender_analysis: Dict[str, Any]) -> str:
        """Generate opinion based on our analysis vs sender's analysis."""
        
        our_risk = our_analysis.get("risk_score", 0.5)
        sender_risk = sender_analysis.get("risk_score", 0.5)
        
        if abs(our_risk - sender_risk) < 0.1:
            return "agree"
        elif our_risk > sender_risk:
            return "higher_risk"
        else:
            return "lower_risk"
    
    def _calculate_risk_adjustment(self, our_analysis: Dict[str, Any], sender_analysis: Dict[str, Any]) -> float:
        """Calculate suggested risk adjustment."""
        
        our_risk = our_analysis.get("risk_score", 0.5)
        sender_risk = sender_analysis.get("risk_score", 0.5)
        our_confidence = our_analysis.get("confidence", 0.5)
        
        # Weight our adjustment by our confidence
        raw_adjustment = (our_risk - sender_risk) * our_confidence
        
        # Limit adjustment magnitude
        return max(-0.3, min(0.3, raw_adjustment))
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status."""
        
        return {
            "agent_id": self.agent_id,
            "specialization": self.specialization,
            "status": self.status.value,
            "performance_metrics": self.performance_metrics.copy(),
            "knowledge_base_stats": {
                "patterns_known": len(self.knowledge_base.patterns),
                "insights_learned": len(self.knowledge_base.learned_insights),
                "collaboration_history": len(self.knowledge_base.collaboration_history)
            },
            "recent_decisions": len([d for d in self.decision_history 
                                  if d.timestamp > datetime.now() - timedelta(hours=1)]),
            "collaboration_partners": len(self.collaboration_partners),
            "last_active": datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Gracefully shutdown the agent."""
        self.status = AgentStatus.SHUTDOWN
        logger.info(f"Agent {self.agent_id} ({self.specialization}) has been shutdown")