import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import uuid
from mcp.server import Server
from mcp.types import Tool, TextContent
from openai import AsyncOpenAI

from ..schemas.transaction_schema import TransactionSchema, FraudScoreSchema
from ..utils.config import config

logger = logging.getLogger(__name__)

class DecisionEngineServer:
    def __init__(self):
        self.server = Server("decision-engine-server")
        self.decision_rules: List[Dict[str, Any]] = []
        self.escalation_queue: List[Dict[str, Any]] = []
        self.decision_history: List[Dict[str, Any]] = []
        self.risk_thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.95
        }
        self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
        self.setup_tools()
        self._initialize_default_rules()
        
    def setup_tools(self):
        """Register MCP tools for decision engine operations"""
        
        @self.server.tool()
        async def score_transaction(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
            """Score a transaction for fraud risk"""
            try:
                start_time = datetime.utcnow()
                
                # Validate transaction
                transaction = TransactionSchema(**transaction_data)
                
                # Calculate fraud score
                fraud_score = await self._calculate_fraud_score(transaction)
                
                # Make decision
                decision = await self._make_fraud_decision(fraud_score, transaction)
                
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Create fraud score response
                fraud_response = FraudScoreSchema(
                    transaction_id=transaction.transaction_id,
                    fraud_score=fraud_score["score"],
                    confidence=fraud_score["confidence"],
                    risk_factors=fraud_score["risk_factors"],
                    decision=decision["action"],
                    explanation=decision["explanation"],
                    processing_time_ms=processing_time
                )
                
                # Store decision in history
                self.decision_history.append({
                    "transaction_id": transaction.transaction_id,
                    "fraud_score": fraud_score["score"],
                    "decision": decision["action"],
                    "timestamp": start_time.isoformat(),
                    "processing_time_ms": processing_time
                })
                
                logger.info(f"Scored transaction {transaction.transaction_id}: {fraud_score['score']:.3f} -> {decision['action']}")
                
                return fraud_response.dict()
                
            except Exception as e:
                logger.error(f"Transaction scoring failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e),
                    "transaction_id": transaction_data.get("transaction_id", "unknown")
                }
        
        @self.server.tool()
        async def assess_risk(risk_assessment_data: Dict[str, Any]) -> Dict[str, Any]:
            """Perform comprehensive risk assessment with confidence intervals"""
            try:
                transaction_data = risk_assessment_data.get("transaction", {})
                context_data = risk_assessment_data.get("context", {})
                
                risk_assessment = await self._assess_comprehensive_risk(transaction_data, context_data)
                
                return {
                    "status": "success",
                    "risk_assessment": risk_assessment,
                    "confidence_interval": risk_assessment["confidence_interval"],
                    "risk_level": risk_assessment["risk_level"],
                    "contributing_factors": risk_assessment["contributing_factors"]
                }
                
            except Exception as e:
                logger.error(f"Risk assessment failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def generate_rules(pattern_data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate new fraud detection rules based on emerging patterns"""
            try:
                patterns = pattern_data.get("patterns", [])
                confidence_threshold = pattern_data.get("confidence_threshold", 0.8)
                
                new_rules = await self._generate_fraud_rules(patterns, confidence_threshold)
                
                # Add rules to active rule set
                for rule in new_rules:
                    rule["id"] = str(uuid.uuid4())
                    rule["created_at"] = datetime.utcnow().isoformat()
                    rule["status"] = "active"
                    self.decision_rules.append(rule)
                
                return {
                    "status": "success",
                    "generated_rules": new_rules,
                    "total_active_rules": len(self.decision_rules)
                }
                
            except Exception as e:
                logger.error(f"Rule generation failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def escalate_case(escalation_data: Dict[str, Any]) -> Dict[str, Any]:
            """Escalate a case for human review"""
            try:
                transaction_id = escalation_data["transaction_id"]
                reason = escalation_data.get("reason", "High risk score")
                priority = escalation_data.get("priority", "medium")
                
                escalation_case = await self._create_escalation_case(transaction_id, reason, priority)
                
                self.escalation_queue.append(escalation_case)
                
                return {
                    "status": "success",
                    "case_id": escalation_case["case_id"],
                    "priority": escalation_case["priority"],
                    "estimated_review_time": escalation_case["estimated_review_time"],
                    "queue_position": len(self.escalation_queue)
                }
                
            except Exception as e:
                logger.error(f"Case escalation failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
    
    def _initialize_default_rules(self):
        """Initialize default fraud detection rules"""
        default_rules = [
            {
                "id": "high_amount_rule",
                "name": "High Amount Transaction",
                "condition": "amount > 5000",
                "action": "flag",
                "weight": 0.3,
                "description": "Flag transactions over $5000"
            },
            {
                "id": "velocity_rule",
                "name": "High Velocity",
                "condition": "tx_count_1h > 5",
                "action": "flag",
                "weight": 0.4,
                "description": "Flag users with >5 transactions in 1 hour"
            },
            {
                "id": "unusual_time_rule",
                "name": "Unusual Time",
                "condition": "hour_of_day < 6 OR hour_of_day > 23",
                "action": "review",
                "weight": 0.2,
                "description": "Review transactions outside business hours"
            },
            {
                "id": "round_amount_rule",
                "name": "Round Amount Pattern",
                "condition": "amount % 100 == 0 AND amount > 1000",
                "action": "flag",
                "weight": 0.15,
                "description": "Flag large round amount transactions"
            }
        ]
        
        for rule in default_rules:
            rule["created_at"] = datetime.utcnow().isoformat()
            rule["status"] = "active"
            
        self.decision_rules = default_rules
    
    async def _calculate_fraud_score(self, transaction: TransactionSchema) -> Dict[str, Any]:
        """Calculate comprehensive fraud score for a transaction"""
        score_components = {}
        risk_factors = []
        
        # Amount-based scoring
        amount_score = min(1.0, transaction.amount / 10000)  # Normalize by $10k
        if transaction.amount > 5000:
            risk_factors.append("high_amount")
            score_components["amount"] = amount_score * 0.3
        else:
            score_components["amount"] = amount_score * 0.1
        
        # Time-based scoring
        hour = transaction.timestamp.hour
        if hour < 6 or hour > 23:
            risk_factors.append("unusual_time")
            score_components["time"] = 0.25
        elif hour < 9 or hour > 18:
            score_components["time"] = 0.1
        else:
            score_components["time"] = 0.0
        
        # Location-based scoring
        if not transaction.location:
            risk_factors.append("no_location")
            score_components["location"] = 0.2
        else:
            score_components["location"] = 0.0
        
        # Device-based scoring
        if not transaction.device_info:
            risk_factors.append("no_device_info")
            score_components["device"] = 0.15
        else:
            score_components["device"] = 0.0
        
        # Merchant-based scoring
        if transaction.merchant_category in ["gambling", "adult", "cryptocurrency"]:
            risk_factors.append("high_risk_merchant")
            score_components["merchant"] = 0.3
        else:
            score_components["merchant"] = 0.0
        
        # Apply business rules
        rule_score = await self._apply_business_rules(transaction)
        score_components["rules"] = rule_score["score"]
        risk_factors.extend(rule_score["triggered_rules"])
        
        # Calculate final score
        base_score = sum(score_components.values())
        
        # Apply ML model if available (simulated)
        ml_score = await self._get_ml_model_score(transaction)
        final_score = min(1.0, 0.6 * base_score + 0.4 * ml_score)
        
        # Calculate confidence based on available data
        data_completeness = sum([
            1 if transaction.location else 0,
            1 if transaction.device_info else 0,
            1 if transaction.merchant_id else 0,
            1 if transaction.ip_address else 0
        ]) / 4
        
        confidence = 0.5 + 0.5 * data_completeness
        
        return {
            "score": float(final_score),
            "confidence": float(confidence),
            "risk_factors": risk_factors,
            "score_components": score_components,
            "ml_score": ml_score
        }
    
    async def _apply_business_rules(self, transaction: TransactionSchema) -> Dict[str, Any]:
        """Apply business rules to transaction"""
        triggered_rules = []
        total_weight = 0
        
        for rule in self.decision_rules:
            if rule["status"] != "active":
                continue
                
            # Simple rule evaluation (in production, use a proper rule engine)
            rule_triggered = await self._evaluate_rule(rule, transaction)
            
            if rule_triggered:
                triggered_rules.append(rule["name"])
                total_weight += rule.get("weight", 0.1)
        
        # Normalize score
        rule_score = min(1.0, total_weight)
        
        return {
            "score": rule_score,
            "triggered_rules": triggered_rules
        }
    
    async def _evaluate_rule(self, rule: Dict[str, Any], transaction: TransactionSchema) -> bool:
        """Evaluate a single business rule against transaction"""
        condition = rule.get("condition", "")
        
        # Simple condition evaluation (expand for production)
        if "amount > 5000" in condition:
            return transaction.amount > 5000
        elif "hour_of_day < 6 OR hour_of_day > 23" in condition:
            hour = transaction.timestamp.hour
            return hour < 6 or hour > 23
        elif "amount % 100 == 0 AND amount > 1000" in condition:
            return transaction.amount % 100 == 0 and transaction.amount > 1000
        
        return False
    
    async def _get_ml_model_score(self, transaction: TransactionSchema) -> float:
        """Get ML model fraud score (simulated)"""
        # In production, this would call the Model Orchestration Server
        # For now, simulate based on transaction characteristics
        
        risk_indicators = 0
        
        if transaction.amount > 1000:
            risk_indicators += 1
        if transaction.timestamp.hour < 6 or transaction.timestamp.hour > 23:
            risk_indicators += 1
        if not transaction.location:
            risk_indicators += 1
        if not transaction.device_info:
            risk_indicators += 1
        
        # Simulate ML score
        ml_score = min(1.0, risk_indicators * 0.2 + np.random.normal(0, 0.1))
        return max(0.0, ml_score)
    
    async def _make_fraud_decision(self, fraud_score: Dict[str, Any], transaction: TransactionSchema) -> Dict[str, Any]:
        """Make final fraud decision based on score"""
        score = fraud_score["score"]
        confidence = fraud_score["confidence"]
        
        # Decision thresholds
        if score >= self.risk_thresholds["critical"]:
            action = "decline"
            explanation = f"High fraud risk (score: {score:.3f}). Transaction declined."
        elif score >= self.risk_thresholds["high"]:
            if confidence > 0.8:
                action = "decline"
                explanation = f"High fraud risk with high confidence (score: {score:.3f}). Transaction declined."
            else:
                action = "review"
                explanation = f"High fraud risk but low confidence (score: {score:.3f}). Manual review required."
        elif score >= self.risk_thresholds["medium"]:
            action = "review"
            explanation = f"Medium fraud risk (score: {score:.3f}). Manual review recommended."
        elif score >= self.risk_thresholds["low"]:
            action = "monitor"
            explanation = f"Low fraud risk (score: {score:.3f}). Transaction approved with monitoring."
        else:
            action = "approve"
            explanation = f"Very low fraud risk (score: {score:.3f}). Transaction approved."
        
        # Add risk factors to explanation
        if fraud_score["risk_factors"]:
            explanation += f" Risk factors: {', '.join(fraud_score['risk_factors'])}"
        
        return {
            "action": action,
            "explanation": explanation,
            "confidence": confidence
        }
    
    async def _assess_comprehensive_risk(self, transaction_data: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment with confidence intervals"""
        
        # Base risk assessment
        base_risk = 0.0
        contributing_factors = []
        
        # Transaction amount risk
        amount = transaction_data.get("amount", 0)
        if amount > 10000:
            base_risk += 0.4
            contributing_factors.append({"factor": "high_amount", "contribution": 0.4})
        elif amount > 5000:
            base_risk += 0.2
            contributing_factors.append({"factor": "elevated_amount", "contribution": 0.2})
        
        # Velocity risk from context
        velocity = context_data.get("velocity_score", 0)
        velocity_risk = min(0.3, velocity * 0.3)
        base_risk += velocity_risk
        if velocity_risk > 0:
            contributing_factors.append({"factor": "velocity", "contribution": velocity_risk})
        
        # Geographic risk
        location = transaction_data.get("location", {})
        if location.get("country") in ["high_risk_country_1", "high_risk_country_2"]:
            base_risk += 0.25
            contributing_factors.append({"factor": "high_risk_geography", "contribution": 0.25})
        
        # Device risk
        device_info = transaction_data.get("device_info", {})
        if not device_info:
            base_risk += 0.15
            contributing_factors.append({"factor": "unknown_device", "contribution": 0.15})
        
        # Normalize risk score
        risk_score = min(1.0, base_risk)
        
        # Calculate confidence interval
        data_quality = len([x for x in [amount, location, device_info] if x]) / 3
        confidence_width = 0.2 * (1 - data_quality)  # Lower quality = wider interval
        
        confidence_interval = {
            "lower": max(0.0, risk_score - confidence_width),
            "upper": min(1.0, risk_score + confidence_width),
            "width": confidence_width * 2
        }
        
        # Determine risk level
        if risk_score >= 0.8:
            risk_level = "critical"
        elif risk_score >= 0.6:
            risk_level = "high"
        elif risk_score >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return {
            "risk_score": float(risk_score),
            "risk_level": risk_level,
            "confidence_interval": confidence_interval,
            "contributing_factors": contributing_factors,
            "data_quality_score": float(data_quality)
        }
    
    async def _generate_fraud_rules(self, patterns: List[Dict[str, Any]], confidence_threshold: float) -> List[Dict[str, Any]]:
        """Generate new fraud detection rules using AI"""
        if not self.client:
            # Fallback to simple rule generation
            return await self._generate_simple_rules(patterns)
        
        try:
            # Use OpenAI to generate sophisticated rules
            prompt = f"""
            Based on the following fraud patterns, generate specific fraud detection rules:
            
            Patterns: {json.dumps(patterns, indent=2)}
            
            Generate rules that:
            1. Have clear conditions that can be evaluated programmatically
            2. Include appropriate weights (0.0 to 1.0)
            3. Have confidence scores above {confidence_threshold}
            4. Are specific enough to avoid false positives
            
            Return as JSON array with format:
            [{{
                "name": "Rule Name",
                "condition": "programmable condition",
                "action": "approve|review|decline",
                "weight": 0.0-1.0,
                "confidence": 0.0-1.0,
                "description": "human readable description"
            }}]
            """
            
            response = await self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            rules_text = response.choices[0].message.content
            rules = json.loads(rules_text)
            
            # Filter by confidence threshold
            filtered_rules = [rule for rule in rules if rule.get("confidence", 0) >= confidence_threshold]
            
            return filtered_rules
            
        except Exception as e:
            logger.error(f"AI rule generation failed: {str(e)}")
            return await self._generate_simple_rules(patterns)
    
    async def _generate_simple_rules(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate simple rules as fallback"""
        rules = []
        
        for pattern in patterns:
            pattern_type = pattern.get("pattern_type", "unknown")
            confidence = pattern.get("confidence", 0.5)
            
            if pattern_type == "high_amount_pattern" and confidence > 0.7:
                rules.append({
                    "name": f"High Amount Pattern {pattern.get('pattern_id', '')}",
                    "condition": "amount > 3000",
                    "action": "review",
                    "weight": 0.3,
                    "confidence": confidence,
                    "description": f"Pattern-based rule for high amounts"
                })
            elif pattern_type == "velocity_pattern" and confidence > 0.8:
                rules.append({
                    "name": f"Velocity Pattern {pattern.get('pattern_id', '')}",
                    "condition": "tx_count_1h > 3",
                    "action": "flag",
                    "weight": 0.4,
                    "confidence": confidence,
                    "description": f"Pattern-based rule for transaction velocity"
                })
        
        return rules
    
    async def _create_escalation_case(self, transaction_id: str, reason: str, priority: str) -> Dict[str, Any]:
        """Create an escalation case for human review"""
        case_id = str(uuid.uuid4())
        
        # Estimate review time based on priority and queue
        priority_multipliers = {"low": 4, "medium": 2, "high": 1, "critical": 0.5}
        base_time = 30  # 30 minutes base
        estimated_time = int(base_time * priority_multipliers.get(priority, 2))
        
        escalation_case = {
            "case_id": case_id,
            "transaction_id": transaction_id,
            "reason": reason,
            "priority": priority,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "estimated_review_time": estimated_time,
            "assigned_reviewer": None
        }
        
        return escalation_case
    
    def get_server(self) -> Server:
        """Get the MCP server instance"""
        return self.server
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current server metrics"""
        recent_decisions = [d for d in self.decision_history 
                          if (datetime.utcnow() - datetime.fromisoformat(d["timestamp"])).seconds < 3600]
        
        avg_processing_time = np.mean([d["processing_time_ms"] for d in recent_decisions]) if recent_decisions else 0
        
        decision_breakdown = {}
        for decision in recent_decisions:
            action = decision["decision"]
            decision_breakdown[action] = decision_breakdown.get(action, 0) + 1
        
        return {
            "total_decisions": len(self.decision_history),
            "recent_decisions_1h": len(recent_decisions),
            "avg_processing_time_ms": float(avg_processing_time),
            "decision_breakdown": decision_breakdown,
            "active_rules": len([r for r in self.decision_rules if r["status"] == "active"]),
            "escalation_queue_size": len(self.escalation_queue),
            "risk_thresholds": self.risk_thresholds
        }
    
    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """Update risk thresholds"""
        self.risk_thresholds.update(new_thresholds)
        logger.info(f"Updated risk thresholds: {self.risk_thresholds}")
    
    def get_escalation_queue(self) -> List[Dict[str, Any]]:
        """Get current escalation queue"""
        return self.escalation_queue.copy()