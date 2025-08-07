import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import uuid

from .base_agent import BaseAgent
from .pattern_recognition_agent import PatternRecognitionAgent
from .risk_assessment_agent import RiskAssessmentAgent

logger = logging.getLogger(__name__)

class CoordinationAgent(BaseAgent):
    """Agent that coordinates and orchestrates multi-agent fraud detection"""
    
    def __init__(self, agent_id: str = "coordination_001"):
        super().__init__(agent_id, "coordination")
        self.managed_agents: Dict[str, BaseAgent] = {}
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        self.consensus_history: List[Dict[str, Any]] = []
        self.task_queue: List[Dict[str, Any]] = []
        self.coordination_strategies = {
            "consensus": self._consensus_strategy,
            "weighted_voting": self._weighted_voting_strategy,
            "expert_selection": self._expert_selection_strategy,
            "cascade": self._cascade_strategy
        }
        self.current_strategy = "weighted_voting"
        
    async def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent for coordination"""
        self.managed_agents[agent.agent_id] = agent
        self.agent_performance[agent.agent_id] = {
            "accuracy": 0.5,
            "response_time": 100.0,
            "confidence": 0.5,
            "specialization_score": 1.0,
            "last_updated": datetime.utcnow().isoformat()
        }
        logger.info(f"Registered agent {agent.agent_id} ({agent.agent_type})")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate multiple agents to process fraud detection task"""
        start_time = datetime.utcnow()
        task_id = str(uuid.uuid4())
        
        try:
            # Determine which agents should participate
            participating_agents = await self._select_agents_for_task(input_data)
            
            if not participating_agents:
                return {
                    "agent_id": self.agent_id,
                    "error": "No suitable agents available",
                    "confidence": 0.0
                }
            
            # Execute agents in parallel
            agent_results = await self._execute_agents_parallel(participating_agents, input_data)
            
            # Apply coordination strategy to combine results
            coordinated_result = await self._apply_coordination_strategy(agent_results, input_data)
            
            # Generate explanation of coordination decision
            coordination_explanation = await self._generate_coordination_explanation(
                agent_results, coordinated_result
            )
            
            # Update agent performance metrics
            await self._update_agent_performance(agent_results, coordinated_result)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            confidence = coordinated_result.get("confidence", 0.5)
            
            # Log coordination results with structured data
            transaction_id = input_data.get("transaction", {}).get("transaction_id", "unknown")
            logger.info(
                "🤝 Agent coordination completed",
                extra={
                    "agent_id": self.agent_id,
                    "transaction_id": transaction_id,
                    "task_id": task_id,
                    "coordination_strategy": self.current_strategy,
                    "participating_agents": [agent.agent_id for agent in participating_agents],
                    "fraud_score": coordinated_result.get("fraud_score", 0.0),
                    "decision": coordinated_result.get("decision", "unknown"),
                    "confidence": confidence,
                    "processing_time_ms": processing_time,
                    "event": "agent_coordination_complete"
                }
            )
            
            result = {
                "task_id": task_id,
                "agent_id": self.agent_id,
                "coordination_strategy": self.current_strategy,
                "participating_agents": [agent.agent_id for agent in participating_agents],
                "agent_results": {agent.agent_id: result for agent, result in agent_results.items()},
                "coordinated_decision": coordinated_result,
                "coordination_explanation": coordination_explanation,
                "consensus_score": coordinated_result.get("consensus_score", 0.5),
                "confidence": confidence,
                "processing_time_ms": processing_time
            }
            
            # Store coordination decision
            self.consensus_history.append({
                "task_id": task_id,
                "timestamp": start_time.isoformat(),
                "strategy": self.current_strategy,
                "agents": [agent.agent_id for agent in participating_agents],
                "result": coordinated_result,
                "confidence": confidence
            })
            
            await self.record_decision(input_data, result, processing_time, confidence)
            return result
            
        except Exception as e:
            logger.error(f"Coordination processing failed: {str(e)}")
            return {
                "task_id": task_id,
                "agent_id": self.agent_id,
                "error": str(e),
                "confidence": 0.0
            }
    
    async def learn(self, feedback: Dict[str, Any]) -> None:
        """Learn from feedback to improve coordination strategies"""
        try:
            task_id = feedback.get("task_id")
            was_accurate = feedback.get("was_accurate", False)
            actual_fraud = feedback.get("actual_fraud", False)
            
            # Find the corresponding coordination decision
            coordination_record = None
            for record in reversed(self.consensus_history):
                if record.get("task_id") == task_id:
                    coordination_record = record
                    break
            
            if not coordination_record:
                logger.warning(f"No coordination record found for task {task_id}")
                return
            
            # Update strategy performance
            strategy_used = coordination_record["strategy"]
            if strategy_used not in self.knowledge_base:
                self.knowledge_base[strategy_used] = {"correct": 0, "total": 0, "accuracy": 0.5}
            
            self.knowledge_base[strategy_used]["total"] += 1
            if was_accurate:
                self.knowledge_base[strategy_used]["correct"] += 1
            
            self.knowledge_base[strategy_used]["accuracy"] = (
                self.knowledge_base[strategy_used]["correct"] / 
                self.knowledge_base[strategy_used]["total"]
            )
            
            # Propagate feedback to participating agents
            participating_agents = coordination_record.get("agents", [])
            for agent_id in participating_agents:
                if agent_id in self.managed_agents:
                    agent_feedback = {
                        **feedback,
                        "coordination_context": True,
                        "strategy_used": strategy_used
                    }
                    await self.managed_agents[agent_id].learn(agent_feedback)
            
            # Adapt coordination strategy if needed
            await self._adapt_coordination_strategy()
            
            logger.info(f"Coordination learning completed for task {task_id}")
            
        except Exception as e:
            logger.error(f"Coordination learning failed: {str(e)}")
    
    async def get_reasoning(self, input_data: Dict[str, Any]) -> str:
        """Get human-readable reasoning for coordination decisions"""
        reasoning_prompt = f"""
        Explain how the coordination agent orchestrated multiple AI agents for fraud detection:
        
        Available Agents: {len(self.managed_agents)}
        Current Strategy: {self.current_strategy}
        Recent Coordination History: {len(self.consensus_history)} decisions
        
        Describe:
        1. How agents were selected for this task
        2. How their individual results were combined
        3. Why this coordination approach was chosen
        4. The confidence in the final decision
        
        Provide a clear explanation of the multi-agent coordination process.
        """
        
        ai_reasoning = await self._generate_ai_response(reasoning_prompt, max_tokens=350)
        
        if ai_reasoning:
            return ai_reasoning
        
        # Fallback reasoning
        return f"Coordination agent orchestrated {len(self.managed_agents)} specialized agents using {self.current_strategy} strategy to reach consensus on fraud detection."
    
    async def _select_agents_for_task(self, input_data: Dict[str, Any]) -> List[BaseAgent]:
        """Select which agents should participate in the task"""
        task_type = input_data.get("task_type", "fraud_detection")
        transaction_data = input_data.get("transaction", {})
        
        selected_agents = []
        
        # Always include pattern recognition for fraud detection
        pattern_agents = [agent for agent in self.managed_agents.values() 
                         if agent.agent_type == "pattern_recognition"]
        selected_agents.extend(pattern_agents)
        
        # Always include risk assessment
        risk_agents = [agent for agent in self.managed_agents.values() 
                      if agent.agent_type == "risk_assessment"]
        selected_agents.extend(risk_agents)
        
        # Add other agents based on task characteristics
        if transaction_data.get("amount", 0) > 5000:
            # High-value transactions - include all available agents
            selected_agents = list(self.managed_agents.values())
        
        # Remove coordination agent itself
        selected_agents = [agent for agent in selected_agents if agent.agent_id != self.agent_id]
        
        # Limit to top-performing agents if too many
        if len(selected_agents) > 5:
            # Sort by performance and take top 5
            agent_scores = []
            for agent in selected_agents:
                performance = self.agent_performance.get(agent.agent_id, {})
                score = (performance.get("accuracy", 0.5) * 0.4 + 
                        performance.get("confidence", 0.5) * 0.3 +
                        performance.get("specialization_score", 0.5) * 0.3)
                agent_scores.append((agent, score))
            
            agent_scores.sort(key=lambda x: x[1], reverse=True)
            selected_agents = [agent for agent, _ in agent_scores[:5]]
        
        return selected_agents
    
    async def _execute_agents_parallel(self, agents: List[BaseAgent], input_data: Dict[str, Any]) -> Dict[BaseAgent, Dict[str, Any]]:
        """Execute multiple agents in parallel"""
        tasks = []
        for agent in agents:
            task = asyncio.create_task(agent.process(input_data))
            tasks.append((agent, task))
        
        results = {}
        for agent, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=5.0)  # 5 second timeout
                results[agent] = result
            except asyncio.TimeoutError:
                logger.warning(f"Agent {agent.agent_id} timed out")
                results[agent] = {"error": "timeout", "confidence": 0.0}
            except Exception as e:
                logger.error(f"Agent {agent.agent_id} failed: {str(e)}")
                results[agent] = {"error": str(e), "confidence": 0.0}
        
        return results
    
    async def _apply_coordination_strategy(self, agent_results: Dict[BaseAgent, Dict[str, Any]], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the current coordination strategy to combine agent results"""
        strategy_func = self.coordination_strategies.get(self.current_strategy)
        if not strategy_func:
            logger.error(f"Unknown coordination strategy: {self.current_strategy}")
            return {"error": "Unknown strategy", "confidence": 0.0}
        
        return await strategy_func(agent_results, input_data)
    
    async def _consensus_strategy(self, agent_results: Dict[BaseAgent, Dict[str, Any]], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Consensus-based coordination strategy"""
        valid_results = [(agent, result) for agent, result in agent_results.items() 
                        if "error" not in result]
        
        if not valid_results:
            return {"error": "No valid agent results", "confidence": 0.0}
        
        # Extract fraud scores/decisions
        fraud_scores = []
        decisions = []
        
        for agent, result in valid_results:
            # Try to extract fraud score
            score = result.get("fraud_score", result.get("composite_risk_score", result.get("pattern_risk_score", 0.5)))
            fraud_scores.append(score)
            
            # Determine decision based on score (more sensitive thresholds)
            if score > 0.6:
                decisions.append("decline")
            elif score > 0.3:
                decisions.append("review")
            else:
                decisions.append("approve")
        
        # Calculate consensus
        from collections import Counter
        decision_counts = Counter(decisions)
        consensus_decision = decision_counts.most_common(1)[0][0]
        consensus_strength = decision_counts[consensus_decision] / len(decisions)
        
        # Average fraud score
        avg_fraud_score = sum(fraud_scores) / len(fraud_scores)
        
        return {
            "fraud_score": avg_fraud_score,
            "decision": consensus_decision,
            "consensus_score": consensus_strength,
            "confidence": consensus_strength * 0.8 + 0.2,  # Minimum 0.2 confidence
            "participating_agents": len(valid_results),
            "strategy": "consensus"
        }
    
    async def _weighted_voting_strategy(self, agent_results: Dict[BaseAgent, Dict[str, Any]], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Weighted voting coordination strategy based on agent performance"""
        valid_results = [(agent, result) for agent, result in agent_results.items() 
                        if "error" not in result]
        
        if not valid_results:
            return {"error": "No valid agent results", "confidence": 0.0}
        
        weighted_score = 0.0
        total_weight = 0.0
        agent_contributions = {}
        
        for agent, result in valid_results:
            # Get agent weight based on performance
            performance = self.agent_performance.get(agent.agent_id, {})
            weight = (performance.get("accuracy", 0.5) * 0.5 + 
                     performance.get("confidence", 0.5) * 0.3 +
                     performance.get("specialization_score", 0.5) * 0.2)
            
            # Extract fraud score
            score = result.get("fraud_score", result.get("composite_risk_score", result.get("pattern_risk_score", 0.5)))
            agent_confidence = result.get("confidence", 0.5)
            
            # Weight by both performance and confidence
            final_weight = weight * agent_confidence
            
            weighted_score += score * final_weight
            total_weight += final_weight
            
            agent_contributions[agent.agent_id] = {
                "score": score,
                "weight": weight,
                "confidence": agent_confidence,
                "contribution": score * final_weight
            }
        
        if total_weight == 0:
            return {"error": "Zero total weight", "confidence": 0.0}
        
        final_score = weighted_score / total_weight
        
        # Determine decision
        if final_score > 0.7:
            decision = "decline"
        elif final_score > 0.5:
            decision = "review"
        else:
            decision = "approve"
        
        # Calculate confidence based on weight distribution
        confidence = min(1.0, total_weight / len(valid_results))
        
        return {
            "fraud_score": final_score,
            "decision": decision,
            "confidence": confidence,
            "agent_contributions": agent_contributions,
            "total_weight": total_weight,
            "participating_agents": len(valid_results),
            "strategy": "weighted_voting"
        }
    
    async def _expert_selection_strategy(self, agent_results: Dict[BaseAgent, Dict[str, Any]], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select the most expert agent for this specific task"""
        valid_results = [(agent, result) for agent, result in agent_results.items() 
                        if "error" not in result]
        
        if not valid_results:
            return {"error": "No valid agent results", "confidence": 0.0}
        
        # Determine task characteristics to select expert
        transaction_data = input_data.get("transaction", {})
        amount = transaction_data.get("amount", 0)
        
        expert_agent = None
        expert_result = None
        expert_score = 0.0
        
        for agent, result in valid_results:
            performance = self.agent_performance.get(agent.agent_id, {})
            
            # Calculate expertise score based on agent type and task
            if agent.agent_type == "pattern_recognition" and amount > 5000:
                # Pattern recognition is expert for high-value transactions
                expertise = performance.get("accuracy", 0.5) * 1.2
            elif agent.agent_type == "risk_assessment":
                # Risk assessment is generally expert for fraud detection
                expertise = performance.get("accuracy", 0.5) * 1.1
            else:
                expertise = performance.get("accuracy", 0.5)
            
            if expertise > expert_score:
                expert_score = expertise
                expert_agent = agent
                expert_result = result
        
        if not expert_agent:
            return {"error": "No expert selected", "confidence": 0.0}
        
        # Use expert's result with some confidence adjustment
        fraud_score = expert_result.get("fraud_score", expert_result.get("composite_risk_score", expert_result.get("pattern_risk_score", 0.5)))
        
        if fraud_score > 0.7:
            decision = "decline"
        elif fraud_score > 0.5:
            decision = "review"
        else:
            decision = "approve"
        
        return {
            "fraud_score": fraud_score,
            "decision": decision,
            "confidence": expert_result.get("confidence", 0.5) * expert_score,
            "expert_agent": expert_agent.agent_id,
            "expert_type": expert_agent.agent_type,
            "expertise_score": expert_score,
            "strategy": "expert_selection"
        }
    
    async def _cascade_strategy(self, agent_results: Dict[BaseAgent, Dict[str, Any]], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cascade strategy - agents vote in order of confidence"""
        valid_results = [(agent, result) for agent, result in agent_results.items() 
                        if "error" not in result]
        
        if not valid_results:
            return {"error": "No valid agent results", "confidence": 0.0}
        
        # Sort by confidence
        sorted_results = sorted(valid_results, 
                              key=lambda x: x[1].get("confidence", 0.5), 
                              reverse=True)
        
        # Use the most confident agent's decision
        most_confident_agent, most_confident_result = sorted_results[0]
        
        fraud_score = most_confident_result.get("fraud_score", most_confident_result.get("composite_risk_score", most_confident_result.get("pattern_risk_score", 0.5)))
        
        if fraud_score > 0.7:
            decision = "decline"
        elif fraud_score > 0.5:
            decision = "review"
        else:
            decision = "approve"
        
        return {
            "fraud_score": fraud_score,
            "decision": decision,
            "confidence": most_confident_result.get("confidence", 0.5),
            "leading_agent": most_confident_agent.agent_id,
            "agent_order": [agent.agent_id for agent, _ in sorted_results],
            "strategy": "cascade"
        }
    
    async def _generate_coordination_explanation(self, agent_results: Dict[BaseAgent, Dict[str, Any]], coordinated_result: Dict[str, Any]) -> str:
        """Generate explanation of how coordination was performed"""
        strategy = coordinated_result.get("strategy", self.current_strategy)
        participating_count = len([r for r in agent_results.values() if "error" not in r])
        
        explanations = [
            f"Coordinated {participating_count} agents using {strategy} strategy"
        ]
        
        if strategy == "weighted_voting":
            contributions = coordinated_result.get("agent_contributions", {})
            if contributions:
                top_contributor = max(contributions.items(), key=lambda x: x[1]["contribution"])
                explanations.append(f"Top contributor: {top_contributor[0]}")
        
        elif strategy == "expert_selection":
            expert = coordinated_result.get("expert_agent")
            if expert:
                explanations.append(f"Selected expert: {expert}")
        
        elif strategy == "consensus":
            consensus_score = coordinated_result.get("consensus_score", 0)
            explanations.append(f"Consensus strength: {consensus_score:.2f}")
        
        final_decision = coordinated_result.get("decision", "unknown")
        confidence = coordinated_result.get("confidence", 0)
        explanations.append(f"Final decision: {final_decision} (confidence: {confidence:.2f})")
        
        return " | ".join(explanations)
    
    async def _update_agent_performance(self, agent_results: Dict[BaseAgent, Dict[str, Any]], coordinated_result: Dict[str, Any]) -> None:
        """Update performance metrics for participating agents"""
        for agent, result in agent_results.items():
            if "error" in result:
                continue
                
            agent_id = agent.agent_id
            if agent_id not in self.agent_performance:
                continue
            
            # Update response time
            processing_time = result.get("processing_time_ms", 100)
            current_time = self.agent_performance[agent_id].get("response_time", 100)
            # Exponential moving average
            self.agent_performance[agent_id]["response_time"] = 0.8 * current_time + 0.2 * processing_time
            
            # Update confidence
            agent_confidence = result.get("confidence", 0.5)
            current_confidence = self.agent_performance[agent_id].get("confidence", 0.5)
            self.agent_performance[agent_id]["confidence"] = 0.8 * current_confidence + 0.2 * agent_confidence
            
            self.agent_performance[agent_id]["last_updated"] = datetime.utcnow().isoformat()
    
    async def _adapt_coordination_strategy(self) -> None:
        """Adapt coordination strategy based on performance"""
        if len(self.knowledge_base) < 4:  # Need some history
            return
        
        # Find best performing strategy
        best_strategy = None
        best_accuracy = 0.0
        
        for strategy, performance in self.knowledge_base.items():
            if isinstance(performance, dict) and "accuracy" in performance:
                if performance["accuracy"] > best_accuracy and performance.get("total", 0) > 5:
                    best_accuracy = performance["accuracy"]
                    best_strategy = strategy
        
        # Switch if significantly better
        current_accuracy = self.knowledge_base.get(self.current_strategy, {}).get("accuracy", 0.5)
        if best_strategy and best_accuracy > current_accuracy + 0.1:
            logger.info(f"Switching coordination strategy from {self.current_strategy} to {best_strategy}")
            self.current_strategy = best_strategy
    
    async def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status"""
        return {
            "managed_agents": len(self.managed_agents),
            "agent_types": [agent.agent_type for agent in self.managed_agents.values()],
            "current_strategy": self.current_strategy,
            "strategy_performance": {k: v for k, v in self.knowledge_base.items() 
                                   if isinstance(v, dict) and "accuracy" in v},
            "recent_coordinations": len([h for h in self.consensus_history 
                                       if (datetime.utcnow() - datetime.fromisoformat(h["timestamp"])).seconds < 3600]),
            "task_queue_size": len(self.task_queue)
        }