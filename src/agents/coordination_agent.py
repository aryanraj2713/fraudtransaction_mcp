import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from .base_agent import BaseFraudDetectionAgent, AgentDecision, CollaborationMessage
from .pattern_recognition_agent import PatternRecognitionAgent
from .risk_assessment_agent import RiskAssessmentAgent
from core.mcp_client import MCPClient

logger = logging.getLogger(__name__)


class ConsensusMethod(str, Enum):
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    EXPERT_OVERRIDE = "expert_override"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


@dataclass
class AgentConsensus:
    consensus_id: str
    transaction_id: str
    participating_agents: List[str]
    individual_decisions: List[AgentDecision]
    consensus_decision: str
    consensus_confidence: float
    consensus_risk_score: float
    method_used: ConsensusMethod
    reasoning: List[str]
    disagreement_level: float
    timestamp: datetime


@dataclass
class CollaborationRequest:
    request_id: str
    requesting_agent: str
    transaction_id: str
    request_type: str
    urgency: str
    context: Dict[str, Any]
    deadline: datetime
    status: str = "pending"


class ConflictResolver:
    """Resolves conflicts between agent decisions."""
    
    def __init__(self):
        self.resolution_strategies = {
            "confidence_based": self._resolve_by_confidence,
            "specialization_based": self._resolve_by_specialization,
            "consensus_building": self._resolve_by_consensus,
            "evidence_based": self._resolve_by_evidence
        }
    
    def resolve_conflict(
        self, 
        decisions: List[AgentDecision], 
        strategy: str = "confidence_based"
    ) -> AgentConsensus:
        """Resolve conflicts between multiple agent decisions."""
        
        if not decisions:
            raise ValueError("No decisions provided for conflict resolution")
        
        if len(decisions) == 1:
            # No conflict to resolve
            decision = decisions[0]
            return AgentConsensus(
                consensus_id=f"consensus_{uuid.uuid4().hex[:8]}",
                transaction_id=decision.transaction_id,
                participating_agents=[decision.agent_id],
                individual_decisions=decisions,
                consensus_decision=decision.decision_type,
                consensus_confidence=decision.confidence,
                consensus_risk_score=decision.risk_score,
                method_used=ConsensusMethod.WEIGHTED_AVERAGE,
                reasoning=["Single agent decision - no conflict to resolve"],
                disagreement_level=0.0,
                timestamp=datetime.now()
            )
        
        # Calculate disagreement level
        disagreement = self._calculate_disagreement(decisions)
        
        # Apply resolution strategy
        resolver = self.resolution_strategies.get(strategy, self._resolve_by_confidence)
        consensus_decision, consensus_confidence, consensus_risk, reasoning = resolver(decisions)
        
        return AgentConsensus(
            consensus_id=f"consensus_{uuid.uuid4().hex[:8]}",
            transaction_id=decisions[0].transaction_id,
            participating_agents=[d.agent_id for d in decisions],
            individual_decisions=decisions,
            consensus_decision=consensus_decision,
            consensus_confidence=consensus_confidence,
            consensus_risk_score=consensus_risk,
            method_used=ConsensusMethod.CONFIDENCE_WEIGHTED,
            reasoning=reasoning,
            disagreement_level=disagreement,
            timestamp=datetime.now()
        )
    
    def _calculate_disagreement(self, decisions: List[AgentDecision]) -> float:
        """Calculate level of disagreement between decisions."""
        
        # Risk score variance
        risk_scores = [d.risk_score for d in decisions]
        risk_variance = np.var(risk_scores) if len(risk_scores) > 1 else 0.0
        
        # Decision type disagreement
        decision_types = [d.decision_type for d in decisions]
        unique_decisions = set(decision_types)
        decision_disagreement = (len(unique_decisions) - 1) / max(len(decisions) - 1, 1)
        
        # Combine metrics
        disagreement = 0.6 * min(1.0, risk_variance * 4) + 0.4 * decision_disagreement
        
        return disagreement
    
    def _resolve_by_confidence(
        self, 
        decisions: List[AgentDecision]
    ) -> Tuple[str, float, float, List[str]]:
        """Resolve conflict by highest confidence."""
        
        # Find decision with highest confidence
        best_decision = max(decisions, key=lambda d: d.confidence)
        
        reasoning = [
            f"Selected decision from agent {best_decision.agent_id} with highest confidence ({best_decision.confidence:.3f})",
            f"Alternative decisions from {len(decisions) - 1} other agents considered"
        ]
        
        return best_decision.decision_type, best_decision.confidence, best_decision.risk_score, reasoning
    
    def _resolve_by_specialization(
        self, 
        decisions: List[AgentDecision]
    ) -> Tuple[str, float, float, List[str]]:
        """Resolve conflict by agent specialization relevance."""
        
        # Simplified specialization weighting
        specialization_weights = {
            "pattern_recognition": 1.2,
            "risk_assessment": 1.1,
            "behavioral_analysis": 1.0,
            "coordination": 0.8
        }
        
        weighted_decisions = []
        for decision in decisions:
            # Extract specialization from agent_id (simplified)
            specialization = "risk_assessment"  # Default
            if "pattern" in decision.agent_id.lower():
                specialization = "pattern_recognition"
            elif "risk" in decision.agent_id.lower():
                specialization = "risk_assessment"
            elif "behavioral" in decision.agent_id.lower():
                specialization = "behavioral_analysis"
            
            weight = specialization_weights.get(specialization, 1.0)
            weighted_score = decision.confidence * weight
            weighted_decisions.append((decision, weighted_score, specialization))
        
        # Select best weighted decision
        best_decision, best_score, best_spec = max(weighted_decisions, key=lambda x: x[1])
        
        reasoning = [
            f"Selected decision from {best_spec} specialist agent {best_decision.agent_id}",
            f"Weighted confidence score: {best_score:.3f}"
        ]
        
        return best_decision.decision_type, best_decision.confidence, best_decision.risk_score, reasoning
    
    def _resolve_by_consensus(
        self, 
        decisions: List[AgentDecision]
    ) -> Tuple[str, float, float, List[str]]:
        """Resolve conflict by building consensus."""
        
        # Weighted average of risk scores
        total_weight = sum(d.confidence for d in decisions)
        if total_weight > 0:
            consensus_risk = sum(d.risk_score * d.confidence for d in decisions) / total_weight
            consensus_confidence = np.mean([d.confidence for d in decisions])
        else:
            consensus_risk = np.mean([d.risk_score for d in decisions])
            consensus_confidence = 0.5
        
        # Determine consensus decision based on risk score
        if consensus_risk > 0.7:
            consensus_decision = "decline"
        elif consensus_risk > 0.3:
            consensus_decision = "review"
        else:
            consensus_decision = "approve"
        
        reasoning = [
            f"Consensus built from {len(decisions)} agent decisions",
            f"Weighted average risk score: {consensus_risk:.3f}",
            f"Average confidence: {consensus_confidence:.3f}"
        ]
        
        return consensus_decision, consensus_confidence, consensus_risk, reasoning
    
    def _resolve_by_evidence(
        self, 
        decisions: List[AgentDecision]
    ) -> Tuple[str, float, float, List[str]]:
        """Resolve conflict by strength of evidence."""
        
        evidence_scores = []
        for decision in decisions:
            # Calculate evidence strength
            reasoning_count = len(decision.reasoning)
            evidence_items = len(decision.evidence.get("risk_factors", []))
            evidence_strength = reasoning_count + evidence_items
            evidence_scores.append((decision, evidence_strength))
        
        # Select decision with strongest evidence
        best_decision, best_evidence = max(evidence_scores, key=lambda x: x[1])
        
        reasoning = [
            f"Selected decision with strongest evidence from agent {best_decision.agent_id}",
            f"Evidence strength score: {best_evidence}"
        ]
        
        return best_decision.decision_type, best_decision.confidence, best_decision.risk_score, reasoning


class WorkflowOrchestrator:
    """Orchestrates multi-agent workflows for fraud detection."""
    
    def __init__(self):
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_templates = {
            "standard_assessment": {
                "stages": ["pattern_recognition", "risk_assessment", "decision_coordination"],
                "parallel_execution": True,
                "consensus_required": True
            },
            "high_risk_escalation": {
                "stages": ["pattern_recognition", "risk_assessment", "behavioral_analysis", "expert_review"],
                "parallel_execution": False,
                "consensus_required": True
            },
            "fast_track": {
                "stages": ["risk_assessment", "decision_coordination"],
                "parallel_execution": True,
                "consensus_required": False
            }
        }
    
    async def execute_workflow(
        self, 
        workflow_type: str, 
        transaction: Dict[str, Any], 
        agents: Dict[str, BaseFraudDetectionAgent]
    ) -> Dict[str, Any]:
        """Execute a multi-agent workflow."""
        
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        transaction_id = transaction.get("transaction_id", "unknown")
        
        if workflow_type not in self.workflow_templates:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        template = self.workflow_templates[workflow_type]
        start_time = datetime.now()
        
        # Initialize workflow tracking
        self.active_workflows[workflow_id] = {
            "workflow_id": workflow_id,
            "transaction_id": transaction_id,
            "workflow_type": workflow_type,
            "start_time": start_time,
            "status": "running",
            "stages_completed": [],
            "agent_results": {}
        }
        
        try:
            agent_decisions = []
            
            if template["parallel_execution"]:
                # Execute agents in parallel
                tasks = []
                for stage in template["stages"]:
                    if stage in agents:
                        agent = agents[stage]
                        task = asyncio.create_task(agent.process_transaction(transaction))
                        tasks.append((stage, task))
                
                # Wait for all tasks to complete
                for stage, task in tasks:
                    try:
                        decision = await task
                        agent_decisions.append(decision)
                        self.active_workflows[workflow_id]["agent_results"][stage] = asdict(decision)
                    except Exception as e:
                        logger.error(f"Agent {stage} failed in workflow {workflow_id}: {e}")
                        continue
            
            else:
                # Execute agents sequentially
                for stage in template["stages"]:
                    if stage in agents:
                        try:
                            agent = agents[stage]
                            decision = await agent.process_transaction(transaction)
                            agent_decisions.append(decision)
                            self.active_workflows[workflow_id]["agent_results"][stage] = asdict(decision)
                        except Exception as e:
                            logger.error(f"Agent {stage} failed in workflow {workflow_id}: {e}")
                            continue
            
            # Build consensus if required
            if template["consensus_required"] and len(agent_decisions) > 1:
                resolver = ConflictResolver()
                consensus = resolver.resolve_conflict(agent_decisions, "confidence_based")
                final_result = {
                    "workflow_result": "consensus",
                    "consensus": asdict(consensus),
                    "individual_decisions": [asdict(d) for d in agent_decisions]
                }
            else:
                # Use single best decision
                if agent_decisions:
                    best_decision = max(agent_decisions, key=lambda d: d.confidence)
                    final_result = {
                        "workflow_result": "single_decision",
                        "best_decision": asdict(best_decision),
                        "all_decisions": [asdict(d) for d in agent_decisions]
                    }
                else:
                    final_result = {
                        "workflow_result": "error",
                        "error": "No agent decisions received"
                    }
            
            # Update workflow status
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.active_workflows[workflow_id].update({
                "status": "completed",
                "end_time": datetime.now(),
                "processing_time_ms": processing_time,
                "agents_participated": len(agent_decisions),
                "result": final_result
            })
            
            logger.info(
                f"Workflow {workflow_id} completed: {len(agent_decisions)} agents, "
                f"{processing_time:.1f}ms"
            )
            
            return {
                "workflow_id": workflow_id,
                "status": "success",
                "processing_time_ms": processing_time,
                "agents_participated": len(agent_decisions),
                **final_result
            }
            
        except Exception as e:
            self.active_workflows[workflow_id]["status"] = "error"
            self.active_workflows[workflow_id]["error"] = str(e)
            logger.error(f"Workflow {workflow_id} failed: {e}")
            
            return {
                "workflow_id": workflow_id,
                "status": "error",
                "error": str(e)
            }


class CoordinationAgent(BaseFraudDetectionAgent):
    """Coordination agent that orchestrates collaboration between specialized agents."""
    
    def __init__(self, agent_id: str, mcp_client: MCPClient):
        super().__init__(agent_id, mcp_client, "coordination")
        
        # Specialized components
        self.conflict_resolver = ConflictResolver()
        self.workflow_orchestrator = WorkflowOrchestrator()
        
        # Agent registry and management
        self.registered_agents: Dict[str, BaseFraudDetectionAgent] = {}
        self.agent_performance: Dict[str, Dict[str, float]] = {}
        self.collaboration_history: List[Dict[str, Any]] = []
        
        # Active collaborations
        self.active_requests: Dict[str, CollaborationRequest] = {}
        self.consensus_cache: Dict[str, AgentConsensus] = {}
    
    async def _load_specialized_knowledge(self):
        """Load coordination specific knowledge."""
        
        coordination_knowledge = {
            "agent_specializations": {
                "pattern_recognition": {
                    "strengths": ["anomaly_detection", "pattern_discovery", "sequential_analysis"],
                    "confidence_domains": ["unusual_patterns", "historical_analysis"]
                },
                "risk_assessment": {
                    "strengths": ["probabilistic_analysis", "uncertainty_quantification", "risk_scoring"],
                    "confidence_domains": ["statistical_analysis", "risk_modeling"]
                },
                "behavioral_analysis": {
                    "strengths": ["user_behavior", "device_analysis", "session_patterns"],
                    "confidence_domains": ["behavioral_anomalies", "user_profiling"]
                }
            },
            "collaboration_triggers": {
                "low_confidence": 0.3,
                "conflicting_decisions": 0.4,
                "high_risk_threshold": 0.8,
                "consensus_required_threshold": 0.6
            }
        }
        
        self.knowledge_base.add_pattern("coordination_rules", coordination_knowledge)
        
        logger.info(f"Coordination Agent {self.agent_id} loaded specialized knowledge")
    
    def register_agent(self, agent: BaseFraudDetectionAgent):
        """Register a specialized agent for coordination."""
        
        self.registered_agents[agent.specialization] = agent
        self.agent_performance[agent.agent_id] = {
            "accuracy_rate": 0.0,
            "avg_confidence": 0.0,
            "processing_time_ms": 0.0,
            "decisions_made": 0
        }
        
        # Establish bidirectional collaboration partnership
        agent.collaboration_partners[self.agent_id] = self.specialization
        self.collaboration_partners[agent.agent_id] = agent.specialization
        
        logger.info(f"Registered agent {agent.agent_id} ({agent.specialization}) for coordination")
    
    async def analyze_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multi-agent analysis of transaction."""
        
        start_time = datetime.now()
        transaction_id = transaction.get("transaction_id", "unknown")
        
        try:
            # Determine workflow type based on transaction characteristics
            workflow_type = self._determine_workflow_type(transaction)
            
            # Execute coordinated workflow
            workflow_result = await self.workflow_orchestrator.execute_workflow(
                workflow_type, transaction, self.registered_agents
            )
            
            # Extract coordination insights
            coordination_insights = self._analyze_coordination_results(workflow_result)
            
            # Calculate coordination metrics
            risk_score = coordination_insights.get("final_risk_score", 0.5)
            confidence = coordination_insights.get("coordination_confidence", 0.5)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            analysis_results = {
                "agent_type": "coordination",
                "risk_score": risk_score,
                "confidence": confidence,
                "risk_factors": coordination_insights.get("risk_factors", []),
                "protective_factors": coordination_insights.get("protective_factors", []),
                "workflow_type": workflow_type,
                "workflow_result": workflow_result,
                "coordination_insights": coordination_insights,
                "processing_time_ms": processing_time,
                "agents_coordinated": len(self.registered_agents)
            }
            
            # Update collaboration history
            self.collaboration_history.append({
                "transaction_id": transaction_id,
                "workflow_type": workflow_type,
                "agents_participated": workflow_result.get("agents_participated", 0),
                "consensus_achieved": "consensus" in workflow_result.get("workflow_result", ""),
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(
                f"Coordination completed for {transaction_id}: "
                f"workflow={workflow_type}, agents={workflow_result.get('agents_participated', 0)}, "
                f"risk={risk_score:.3f}"
            )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Coordination analysis error: {e}")
            return {
                "agent_type": "coordination",
                "risk_score": 0.5,
                "confidence": 0.1,
                "risk_factors": [f"Coordination error: {str(e)}"],
                "protective_factors": [],
                "error": str(e)
            }
    
    def _determine_workflow_type(self, transaction: Dict[str, Any]) -> str:
        """Determine appropriate workflow type for transaction."""
        
        amount = float(transaction.get("amount", 0))
        velocity_1h = transaction.get("velocity_1h", 0)
        is_first_transaction = transaction.get("is_first_transaction", False)
        
        # High-risk indicators suggest escalation workflow
        if (amount > 5000 or 
            velocity_1h > 10 or 
            is_first_transaction and amount > 1000):
            return "high_risk_escalation"
        
        # Low-risk indicators suggest fast track
        elif (amount < 100 and 
              velocity_1h <= 2 and 
              not is_first_transaction):
            return "fast_track"
        
        # Default standard assessment
        else:
            return "standard_assessment"
    
    def _analyze_coordination_results(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and extract insights from coordination workflow results."""
        
        insights = {
            "risk_factors": [],
            "protective_factors": [],
            "agent_agreement": 0.0,
            "final_risk_score": 0.5,
            "coordination_confidence": 0.5
        }
        
        if workflow_result.get("workflow_result") == "consensus":
            # Extract consensus information
            consensus_data = workflow_result.get("consensus", {})
            insights["final_risk_score"] = consensus_data.get("consensus_risk_score", 0.5)
            insights["coordination_confidence"] = consensus_data.get("consensus_confidence", 0.5)
            insights["agent_agreement"] = 1.0 - consensus_data.get("disagreement_level", 0.5)
            
            # Aggregate risk factors from individual decisions
            individual_decisions = consensus_data.get("individual_decisions", [])
            all_risk_factors = []
            all_protective_factors = []
            
            for decision in individual_decisions:
                evidence = decision.get("evidence", {})
                all_risk_factors.extend(evidence.get("risk_factors", []))
                all_protective_factors.extend(evidence.get("protective_factors", []))
            
            # Remove duplicates and take top factors
            insights["risk_factors"] = list(set(all_risk_factors))[:5]
            insights["protective_factors"] = list(set(all_protective_factors))[:3]
            
        elif workflow_result.get("workflow_result") == "single_decision":
            # Use single best decision
            best_decision = workflow_result.get("best_decision", {})
            insights["final_risk_score"] = best_decision.get("risk_score", 0.5)
            insights["coordination_confidence"] = best_decision.get("confidence", 0.5)
            insights["agent_agreement"] = 1.0  # Single decision = perfect agreement
            
            evidence = best_decision.get("evidence", {})
            insights["risk_factors"] = evidence.get("risk_factors", [])[:5]
            insights["protective_factors"] = evidence.get("protective_factors", [])[:3]
        
        # Add coordination-specific insights
        agents_participated = workflow_result.get("agents_participated", 0)
        if agents_participated > 1:
            insights["protective_factors"].append(f"Multi-agent analysis ({agents_participated} agents)")
        
        processing_time = workflow_result.get("processing_time_ms", 0)
        if processing_time < 100:
            insights["protective_factors"].append("Fast processing time indicates routine transaction")
        
        return insights
    
    async def request_agent_collaboration(
        self, 
        transaction: Dict[str, Any], 
        requesting_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Request collaboration from multiple agents."""
        
        request_id = f"collab_req_{uuid.uuid4().hex[:8]}"
        transaction_id = transaction.get("transaction_id", "unknown")
        
        # Create collaboration request
        request = CollaborationRequest(
            request_id=request_id,
            requesting_agent=self.agent_id,
            transaction_id=transaction_id,
            request_type="multi_agent_analysis",
            urgency="normal",
            context=requesting_context,
            deadline=datetime.now() + timedelta(seconds=10)
        )
        
        self.active_requests[request_id] = request
        
        # Send collaboration messages to all registered agents
        collaboration_results = {}
        
        for agent_id, agent in self.registered_agents.items():
            try:
                message = CollaborationMessage(
                    message_id=f"msg_{uuid.uuid4().hex[:8]}",
                    sender_agent_id=self.agent_id,
                    recipient_agent_id=agent.agent_id,
                    message_type="request_opinion",
                    content={
                        "transaction": transaction,
                        "context": requesting_context,
                        "request_id": request_id
                    },
                    priority="normal",
                    timestamp=datetime.now()
                )
                
                response = await agent.collaborate_with_agents(message)
                collaboration_results[agent_id] = response
                
            except Exception as e:
                logger.error(f"Collaboration with agent {agent_id} failed: {e}")
                collaboration_results[agent_id] = {"status": "error", "error": str(e)}
        
        # Update request status
        request.status = "completed"
        
        return {
            "request_id": request_id,
            "status": "completed",
            "collaboration_results": collaboration_results,
            "agents_responded": len([r for r in collaboration_results.values() 
                                   if r.get("status") != "error"]),
            "total_agents": len(self.registered_agents)
        }
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status."""
        
        return {
            "agent_id": self.agent_id,
            "specialization": self.specialization,
            "registered_agents": list(self.registered_agents.keys()),
            "active_workflows": len(self.workflow_orchestrator.active_workflows),
            "collaboration_history": len(self.collaboration_history),
            "recent_collaborations": len([
                c for c in self.collaboration_history 
                if datetime.fromisoformat(c["timestamp"]) > datetime.now() - timedelta(hours=1)
            ]),
            "agent_performance_summary": {
                agent_id: {
                    "decisions_made": perf["decisions_made"],
                    "accuracy_rate": perf["accuracy_rate"],
                    "avg_confidence": perf["avg_confidence"]
                }
                for agent_id, perf in self.agent_performance.items()
            }
        }