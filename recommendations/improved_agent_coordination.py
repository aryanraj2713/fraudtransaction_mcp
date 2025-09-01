"""
Improved Agent Coordination Strategy for MCP Fraud Detection System

This module demonstrates advanced multi-agent coordination techniques
that can improve overall system accuracy through better collaboration.
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low" 
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class ConsensusStrategy(Enum):
    WEIGHTED_VOTING = "weighted_voting"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    EXPERTISE_WEIGHTED = "expertise_weighted"
    DYNAMIC_WEIGHTED = "dynamic_weighted"
    HIERARCHICAL = "hierarchical"

@dataclass
class AgentDecisionEnhanced:
    agent_id: str
    decision_type: str
    confidence: float
    risk_score: float
    evidence_strength: float
    specialization_match: float
    reasoning: List[str]
    uncertainty_metrics: Dict[str, float]
    processing_time_ms: float

class ImprovedCoordinationAgent:
    """Enhanced coordination agent with advanced consensus mechanisms."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.registered_agents = {}
        self.agent_performance_history = {}
        self.specialization_expertise = {}
        self.consensus_cache = {}
        
        # Initialize agent specializations and expertise levels
        self._initialize_agent_expertise()
    
    def _initialize_agent_expertise(self):
        """Initialize agent expertise mappings."""
        
        self.specialization_expertise = {
            'pattern_recognition': {
                'velocity_patterns': 0.9,
                'sequential_patterns': 0.85,
                'anomaly_detection': 0.8,
                'behavioral_patterns': 0.7,
                'geographic_patterns': 0.6
            },
            'risk_assessment': {
                'probabilistic_analysis': 0.9,
                'uncertainty_quantification': 0.85,
                'statistical_modeling': 0.8,
                'risk_scoring': 0.9,
                'cost_benefit_analysis': 0.7
            },
            'behavioral_analysis': {
                'user_profiling': 0.9,
                'device_analysis': 0.85,
                'session_patterns': 0.8,
                'behavioral_anomalies': 0.9,
                'biometric_patterns': 0.7
            }
        }
    
    async def coordinate_enhanced_decision(
        self,
        transaction: Dict[str, Any],
        agent_decisions: List[AgentDecisionEnhanced],
        consensus_strategy: ConsensusStrategy = ConsensusStrategy.DYNAMIC_WEIGHTED
    ) -> Dict[str, Any]:
        """Coordinate agent decisions using enhanced consensus mechanisms."""
        
        if not agent_decisions:
            return self._default_decision(transaction)
        
        # Analyze transaction to determine required expertise
        required_expertise = self._analyze_transaction_expertise_needs(transaction)
        
        # Apply consensus strategy
        if consensus_strategy == ConsensusStrategy.WEIGHTED_VOTING:
            consensus = await self._weighted_voting_consensus(agent_decisions)
        elif consensus_strategy == ConsensusStrategy.CONFIDENCE_WEIGHTED:
            consensus = await self._confidence_weighted_consensus(agent_decisions)
        elif consensus_strategy == ConsensusStrategy.EXPERTISE_WEIGHTED:
            consensus = await self._expertise_weighted_consensus(agent_decisions, required_expertise)
        elif consensus_strategy == ConsensusStrategy.DYNAMIC_WEIGHTED:
            consensus = await self._dynamic_weighted_consensus(agent_decisions, required_expertise)
        elif consensus_strategy == ConsensusStrategy.HIERARCHICAL:
            consensus = await self._hierarchical_consensus(agent_decisions, required_expertise)
        else:
            consensus = await self._weighted_voting_consensus(agent_decisions)
        
        # Add coordination metadata
        consensus['coordination_metadata'] = {
            'num_agents_participated': len(agent_decisions),
            'consensus_strategy': consensus_strategy.value,
            'required_expertise': required_expertise,
            'coordination_confidence': consensus.get('coordination_confidence', 0.5)
        }
        
        return consensus
    
    def _analyze_transaction_expertise_needs(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """Analyze transaction to determine what expertise is most needed."""
        
        expertise_needs = {
            'velocity_patterns': 0.0,
            'sequential_patterns': 0.0,
            'anomaly_detection': 0.0,
            'behavioral_patterns': 0.0,
            'geographic_patterns': 0.0,
            'probabilistic_analysis': 0.0,
            'uncertainty_quantification': 0.0,
            'risk_scoring': 0.0,
            'user_profiling': 0.0,
            'device_analysis': 0.0
        }
        
        # Velocity-based needs
        velocity_1h = transaction.get('velocity_1h', 0)
        velocity_24h = transaction.get('velocity_24h', 0)
        
        if velocity_1h > 5 or velocity_24h > 20:
            expertise_needs['velocity_patterns'] = 0.9
            expertise_needs['sequential_patterns'] = 0.7
        
        # Amount-based needs
        amount = transaction.get('amount', 0)
        if amount > 1000 or amount < 5:
            expertise_needs['anomaly_detection'] = 0.8
            expertise_needs['probabilistic_analysis'] = 0.7
        
        # Geographic needs
        country = transaction.get('country', 'US')
        if country in ['XX', 'YY', 'ZZ']:  # High-risk countries
            expertise_needs['geographic_patterns'] = 0.9
            expertise_needs['risk_scoring'] = 0.8
        
        # Device-based needs
        if transaction.get('is_first_transaction', False):
            expertise_needs['user_profiling'] = 0.8
            expertise_needs['device_analysis'] = 0.7
        
        # New device or suspicious behavior
        device_data = transaction.get('device_data', {})
        if device_data.get('is_new_device', False):
            expertise_needs['device_analysis'] = 0.9
            expertise_needs['behavioral_patterns'] = 0.7
        
        return expertise_needs
    
    async def _weighted_voting_consensus(
        self, 
        agent_decisions: List[AgentDecisionEnhanced]
    ) -> Dict[str, Any]:
        """Simple weighted voting based on historical performance."""
        
        total_weight = 0
        weighted_risk_score = 0
        weighted_confidence = 0
        decision_votes = {'approve': 0, 'decline': 0, 'review': 0}
        
        for decision in agent_decisions:
            # Get agent performance weight
            agent_performance = self.agent_performance_history.get(
                decision.agent_id, {'accuracy': 0.7, 'precision': 0.7}
            )
            weight = (agent_performance['accuracy'] + agent_performance['precision']) / 2
            
            total_weight += weight
            weighted_risk_score += decision.risk_score * weight
            weighted_confidence += decision.confidence * weight
            
            # Vote for decision
            decision_votes[decision.decision_type] += weight
        
        if total_weight > 0:
            final_risk_score = weighted_risk_score / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_risk_score = np.mean([d.risk_score for d in agent_decisions])
            final_confidence = np.mean([d.confidence for d in agent_decisions])
        
        # Determine final decision
        final_decision = max(decision_votes.items(), key=lambda x: x[1])[0]
        
        return {
            'decision': final_decision,
            'risk_score': final_risk_score,
            'confidence': final_confidence,
            'coordination_confidence': min(final_confidence * 1.1, 1.0),
            'reasoning': self._combine_reasoning(agent_decisions)
        }
    
    async def _confidence_weighted_consensus(
        self, 
        agent_decisions: List[AgentDecisionEnhanced]
    ) -> Dict[str, Any]:
        """Consensus based on agent confidence levels."""
        
        # Weight decisions by confidence
        total_confidence = sum(d.confidence for d in agent_decisions)
        
        if total_confidence == 0:
            return await self._weighted_voting_consensus(agent_decisions)
        
        weighted_risk_score = sum(
            d.risk_score * d.confidence for d in agent_decisions
        ) / total_confidence
        
        weighted_confidence = sum(
            d.confidence * d.confidence for d in agent_decisions
        ) / total_confidence
        
        # Decision voting weighted by confidence
        decision_votes = {'approve': 0, 'decline': 0, 'review': 0}
        for decision in agent_decisions:
            decision_votes[decision.decision_type] += decision.confidence
        
        final_decision = max(decision_votes.items(), key=lambda x: x[1])[0]
        
        return {
            'decision': final_decision,
            'risk_score': weighted_risk_score,
            'confidence': weighted_confidence,
            'coordination_confidence': weighted_confidence,
            'reasoning': self._combine_reasoning(agent_decisions, weight_by_confidence=True)
        }
    
    async def _expertise_weighted_consensus(
        self,
        agent_decisions: List[AgentDecisionEnhanced],
        required_expertise: Dict[str, float]
    ) -> Dict[str, Any]:
        """Consensus based on agent expertise relevance to the transaction."""
        
        expertise_weights = {}
        
        for decision in agent_decisions:
            # Get agent specialization
            agent_type = self._get_agent_type(decision.agent_id)
            
            # Calculate expertise relevance weight
            expertise_weight = 0
            if agent_type in self.specialization_expertise:
                agent_expertise = self.specialization_expertise[agent_type]
                
                for expertise_area, need_level in required_expertise.items():
                    if expertise_area in agent_expertise:
                        expertise_weight += need_level * agent_expertise[expertise_area]
            
            expertise_weights[decision.agent_id] = max(expertise_weight, 0.1)  # Minimum weight
        
        # Normalize weights
        total_weight = sum(expertise_weights.values())
        if total_weight > 0:
            expertise_weights = {k: v / total_weight for k, v in expertise_weights.items()}
        
        # Calculate weighted consensus
        weighted_risk_score = sum(
            d.risk_score * expertise_weights.get(d.agent_id, 0.1) 
            for d in agent_decisions
        )
        
        weighted_confidence = sum(
            d.confidence * expertise_weights.get(d.agent_id, 0.1)
            for d in agent_decisions
        )
        
        # Decision voting weighted by expertise
        decision_votes = {'approve': 0, 'decline': 0, 'review': 0}
        for decision in agent_decisions:
            weight = expertise_weights.get(decision.agent_id, 0.1)
            decision_votes[decision.decision_type] += weight
        
        final_decision = max(decision_votes.items(), key=lambda x: x[1])[0]
        
        return {
            'decision': final_decision,
            'risk_score': weighted_risk_score,
            'confidence': weighted_confidence,
            'coordination_confidence': weighted_confidence,
            'expertise_weights': expertise_weights,
            'reasoning': self._combine_reasoning(agent_decisions, custom_weights=expertise_weights)
        }
    
    async def _dynamic_weighted_consensus(
        self,
        agent_decisions: List[AgentDecisionEnhanced],
        required_expertise: Dict[str, float]
    ) -> Dict[str, Any]:
        """Dynamic consensus that adapts weights based on multiple factors."""
        
        dynamic_weights = {}
        
        for decision in agent_decisions:
            agent_id = decision.agent_id
            
            # Factor 1: Historical performance
            performance = self.agent_performance_history.get(
                agent_id, {'accuracy': 0.7, 'precision': 0.7, 'recall': 0.7}
            )
            performance_weight = (
                performance['accuracy'] + performance['precision'] + performance['recall']
            ) / 3
            
            # Factor 2: Confidence level
            confidence_weight = decision.confidence
            
            # Factor 3: Expertise relevance
            agent_type = self._get_agent_type(agent_id)
            expertise_weight = 0
            if agent_type in self.specialization_expertise:
                agent_expertise = self.specialization_expertise[agent_type]
                for expertise_area, need_level in required_expertise.items():
                    if expertise_area in agent_expertise:
                        expertise_weight += need_level * agent_expertise[expertise_area]
            expertise_weight = min(expertise_weight, 1.0)
            
            # Factor 4: Evidence strength
            evidence_weight = decision.evidence_strength
            
            # Factor 5: Processing time (faster = slightly better for real-time)
            time_weight = max(0.5, 1.0 - (decision.processing_time_ms / 1000))
            
            # Combine all factors
            combined_weight = (
                performance_weight * 0.3 +
                confidence_weight * 0.25 +
                expertise_weight * 0.25 +
                evidence_weight * 0.15 +
                time_weight * 0.05
            )
            
            dynamic_weights[agent_id] = combined_weight
        
        # Normalize weights
        total_weight = sum(dynamic_weights.values())
        if total_weight > 0:
            dynamic_weights = {k: v / total_weight for k, v in dynamic_weights.items()}
        
        # Calculate weighted consensus
        weighted_risk_score = sum(
            d.risk_score * dynamic_weights.get(d.agent_id, 0.1)
            for d in agent_decisions
        )
        
        weighted_confidence = sum(
            d.confidence * dynamic_weights.get(d.agent_id, 0.1)
            for d in agent_decisions
        )
        
        # Decision voting with dynamic weights
        decision_votes = {'approve': 0, 'decline': 0, 'review': 0}
        for decision in agent_decisions:
            weight = dynamic_weights.get(decision.agent_id, 0.1)
            decision_votes[decision.decision_type] += weight
        
        final_decision = max(decision_votes.items(), key=lambda x: x[1])[0]
        
        # Calculate coordination confidence based on agreement
        decision_agreement = self._calculate_decision_agreement(agent_decisions)
        coordination_confidence = min(weighted_confidence * (1 + decision_agreement), 1.0)
        
        return {
            'decision': final_decision,
            'risk_score': weighted_risk_score,
            'confidence': weighted_confidence,
            'coordination_confidence': coordination_confidence,
            'dynamic_weights': dynamic_weights,
            'decision_agreement': decision_agreement,
            'reasoning': self._combine_reasoning(agent_decisions, custom_weights=dynamic_weights)
        }
    
    async def _hierarchical_consensus(
        self,
        agent_decisions: List[AgentDecisionEnhanced],
        required_expertise: Dict[str, float]
    ) -> Dict[str, Any]:
        """Hierarchical consensus with escalation for disagreements."""
        
        # Step 1: Check for high-confidence unanimous decisions
        high_confidence_decisions = [d for d in agent_decisions if d.confidence > 0.8]
        
        if len(high_confidence_decisions) == len(agent_decisions):
            # All agents are highly confident
            decision_types = [d.decision_type for d in high_confidence_decisions]
            if len(set(decision_types)) == 1:
                # Unanimous high-confidence decision
                avg_risk = np.mean([d.risk_score for d in high_confidence_decisions])
                avg_confidence = np.mean([d.confidence for d in high_confidence_decisions])
                
                return {
                    'decision': decision_types[0],
                    'risk_score': avg_risk,
                    'confidence': avg_confidence,
                    'coordination_confidence': min(avg_confidence * 1.2, 1.0),
                    'consensus_type': 'unanimous_high_confidence',
                    'reasoning': self._combine_reasoning(high_confidence_decisions)
                }
        
        # Step 2: Check for expert consensus on specialized transactions
        most_needed_expertise = max(required_expertise.items(), key=lambda x: x[1])
        expert_agents = []
        
        for decision in agent_decisions:
            agent_type = self._get_agent_type(decision.agent_id)
            if (agent_type in self.specialization_expertise and 
                most_needed_expertise[0] in self.specialization_expertise[agent_type]):
                expertise_level = self.specialization_expertise[agent_type][most_needed_expertise[0]]
                if expertise_level > 0.8:  # High expertise
                    expert_agents.append(decision)
        
        if expert_agents:
            # Use expert consensus
            expert_decision_types = [d.decision_type for d in expert_agents]
            if len(set(expert_decision_types)) == 1:
                # Expert consensus
                avg_risk = np.mean([d.risk_score for d in expert_agents])
                avg_confidence = np.mean([d.confidence for d in expert_agents])
                
                return {
                    'decision': expert_decision_types[0],
                    'risk_score': avg_risk,
                    'confidence': avg_confidence,
                    'coordination_confidence': min(avg_confidence * 1.1, 1.0),
                    'consensus_type': 'expert_consensus',
                    'expert_agents': [d.agent_id for d in expert_agents],
                    'reasoning': self._combine_reasoning(expert_agents)
                }
        
        # Step 3: Fallback to dynamic weighted consensus
        return await self._dynamic_weighted_consensus(agent_decisions, required_expertise)
    
    def _calculate_decision_agreement(self, agent_decisions: List[AgentDecisionEnhanced]) -> float:
        """Calculate the level of agreement between agent decisions."""
        
        if len(agent_decisions) <= 1:
            return 1.0
        
        decision_types = [d.decision_type for d in agent_decisions]
        risk_scores = [d.risk_score for d in agent_decisions]
        
        # Decision type agreement
        most_common_decision = max(set(decision_types), key=decision_types.count)
        decision_agreement = decision_types.count(most_common_decision) / len(decision_types)
        
        # Risk score agreement (based on standard deviation)
        risk_std = np.std(risk_scores)
        risk_agreement = max(0, 1 - (risk_std / 0.5))  # Normalize by expected max std
        
        # Combined agreement
        return (decision_agreement + risk_agreement) / 2
    
    def _get_agent_type(self, agent_id: str) -> str:
        """Extract agent type from agent ID."""
        if 'pattern_recognition' in agent_id:
            return 'pattern_recognition'
        elif 'risk_assessment' in agent_id:
            return 'risk_assessment'
        elif 'behavioral' in agent_id:
            return 'behavioral_analysis'
        else:
            return 'unknown'
    
    def _combine_reasoning(
        self, 
        agent_decisions: List[AgentDecisionEnhanced],
        weight_by_confidence: bool = False,
        custom_weights: Dict[str, float] = None
    ) -> List[str]:
        """Combine reasoning from multiple agents."""
        
        combined_reasoning = []
        
        for decision in agent_decisions:
            weight = 1.0
            
            if custom_weights and decision.agent_id in custom_weights:
                weight = custom_weights[decision.agent_id]
            elif weight_by_confidence:
                weight = decision.confidence
            
            # Add reasoning with weight indication
            if weight > 0.7:
                prefix = "[HIGH WEIGHT]"
            elif weight > 0.4:
                prefix = "[MEDIUM WEIGHT]"
            else:
                prefix = "[LOW WEIGHT]"
            
            for reason in decision.reasoning:
                combined_reasoning.append(f"{prefix} {decision.agent_id}: {reason}")
        
        return combined_reasoning
    
    def _default_decision(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Default decision when no agent decisions are available."""
        
        # Simple rule-based fallback
        amount = transaction.get('amount', 0)
        velocity_1h = transaction.get('velocity_1h', 0)
        country = transaction.get('country', 'US')
        
        risk_score = 0.0
        
        if amount > 2000:
            risk_score += 0.3
        if velocity_1h > 10:
            risk_score += 0.4
        if country in ['XX', 'YY', 'ZZ']:
            risk_score += 0.5
        
        decision = 'decline' if risk_score > 0.6 else 'approve'
        
        return {
            'decision': decision,
            'risk_score': min(risk_score, 1.0),
            'confidence': 0.3,  # Low confidence for fallback
            'coordination_confidence': 0.2,
            'reasoning': ['Fallback decision - no agent decisions available'],
            'consensus_type': 'fallback'
        }
    
    def update_agent_performance(
        self, 
        agent_id: str, 
        performance_metrics: Dict[str, float]
    ):
        """Update agent performance history for future weighting."""
        
        if agent_id not in self.agent_performance_history:
            self.agent_performance_history[agent_id] = {
                'accuracy': 0.7,
                'precision': 0.7,
                'recall': 0.7,
                'f1_score': 0.7,
                'update_count': 0
            }
        
        # Exponential moving average for performance updates
        alpha = 0.3  # Learning rate
        current = self.agent_performance_history[agent_id]
        
        for metric, value in performance_metrics.items():
            if metric in current:
                current[metric] = alpha * value + (1 - alpha) * current[metric]
        
        current['update_count'] += 1
