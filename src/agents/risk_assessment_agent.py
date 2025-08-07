import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import json
from scipy import stats

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class RiskAssessmentAgent(BaseAgent):
    """Agent specialized in comprehensive risk assessment with reasoning chains"""
    
    def __init__(self, agent_id: str = "risk_assessment_001"):
        super().__init__(agent_id, "risk_assessment")
        self.risk_factors: Dict[str, Dict[str, Any]] = {}
        self.risk_models: Dict[str, Any] = {}
        self.confidence_intervals: Dict[str, Tuple[float, float]] = {}
        self.reasoning_chains: List[Dict[str, Any]] = []
        self._initialize_risk_models()
        
    def _initialize_risk_models(self):
        """Initialize risk assessment models"""
        self.risk_models = {
            "amount_risk": {
                "thresholds": {"low": 50, "medium": 500, "high": 2000, "critical": 5000},
                "weights": {"low": 0.1, "medium": 0.4, "high": 0.7, "critical": 0.9}
            },
            "velocity_risk": {
                "thresholds": {"tx_1h": 3, "tx_24h": 10, "amount_1h": 2000},
                "weights": {"tx_1h": 0.4, "tx_24h": 0.3, "amount_1h": 0.3}
            },
            "geographic_risk": {
                "high_risk_countries": ["NG", "PK", "RU", "CN", "IR", "KP", "AF", "SY"],
                "medium_risk_countries": ["IN", "BR", "MX", "TR", "EG", "ID"],
                "risk_scores": {"unknown": 0.6, "domestic": 0.1, "medium_risk": 0.5, "high_risk": 0.8}
            },
            "temporal_risk": {
                "high_risk_hours": list(range(0, 6)) + list(range(22, 24)),
                "weekend_multiplier": 1.3
            },
            "merchant_risk": {
                "high_risk_categories": ["cryptocurrency", "gambling", "money_transfer", "adult", "cash_advance"],
                "medium_risk_categories": ["online", "travel", "electronics"],
                "risk_scores": {"high_risk": 0.7, "medium_risk": 0.4, "low_risk": 0.1}
            }
        }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment with reasoning chains"""
        start_time = datetime.utcnow()
        
        try:
            transaction_data = input_data.get("transaction", {})
            context_data = input_data.get("context", {})
            user_profile = input_data.get("user_profile", {})
            
            # Build reasoning chain
            reasoning_chain = []
            
            # Assess individual risk components
            amount_risk = await self._assess_amount_risk(transaction_data, reasoning_chain)
            velocity_risk = await self._assess_velocity_risk(context_data, reasoning_chain)
            geographic_risk = await self._assess_geographic_risk(transaction_data, reasoning_chain)
            temporal_risk = await self._assess_temporal_risk(transaction_data, reasoning_chain)
            merchant_risk = await self._assess_merchant_risk(transaction_data, reasoning_chain)
            behavioral_risk = await self._assess_behavioral_risk(transaction_data, user_profile, reasoning_chain)
            
            # Calculate composite risk score
            composite_risk = await self._calculate_composite_risk({
                "amount": amount_risk,
                "velocity": velocity_risk,
                "geographic": geographic_risk,
                "temporal": temporal_risk,
                "merchant": merchant_risk,
                "behavioral": behavioral_risk
            }, reasoning_chain)
            
            # Calculate confidence intervals
            confidence_interval = await self._calculate_confidence_interval(
                composite_risk, transaction_data, context_data
            )
            
            # Generate risk explanation
            risk_explanation = await self._generate_risk_explanation(reasoning_chain)
            
            # Determine risk level
            risk_level = self._determine_risk_level(composite_risk["final_score"])
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                composite_risk, risk_level, reasoning_chain
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            confidence = self._calculate_confidence({
                "data_quality": len([x for x in [transaction_data, context_data, user_profile] if x]) / 3,
                "pattern_match": min(1.0, len(reasoning_chain) / 5),
                "historical_accuracy": self.performance_metrics["correct_decisions"] / max(1, self.performance_metrics["total_decisions"])
            })
            
            result = {
                "agent_id": self.agent_id,
                "composite_risk_score": composite_risk["final_score"],
                "risk_level": risk_level,
                "risk_components": {
                    "amount_risk": amount_risk,
                    "velocity_risk": velocity_risk,
                    "geographic_risk": geographic_risk,
                    "temporal_risk": temporal_risk,
                    "merchant_risk": merchant_risk,
                    "behavioral_risk": behavioral_risk
                },
                "confidence_interval": confidence_interval,
                "reasoning_chain": reasoning_chain,
                "risk_explanation": risk_explanation,
                "recommendations": recommendations,
                "confidence": confidence,
                "processing_time_ms": processing_time
            }
            
            # Store reasoning chain for learning
            self.reasoning_chains.append({
                "timestamp": start_time.isoformat(),
                "reasoning_chain": reasoning_chain,
                "final_score": composite_risk["final_score"],
                "confidence": confidence
            })
            
            await self.record_decision(input_data, result, processing_time, confidence)
            return result
            
        except Exception as e:
            logger.error(f"Risk assessment processing failed: {str(e)}")
            return {
                "agent_id": self.agent_id,
                "error": str(e),
                "confidence": 0.0
            }
    
    async def learn(self, feedback: Dict[str, Any]) -> None:
        """Learn from feedback to improve risk assessment accuracy"""
        try:
            transaction_id = feedback.get("transaction_id")
            actual_fraud = feedback.get("actual_fraud", False)
            predicted_risk = feedback.get("predicted_risk", 0.5)
            
            # Calculate prediction error
            error = abs((1.0 if actual_fraud else 0.0) - predicted_risk)
            
            # Update risk factor weights based on feedback
            risk_components = feedback.get("risk_components", {})
            
            for component, risk_data in risk_components.items():
                if component in self.risk_factors:
                    # Adjust component weight based on accuracy
                    current_weight = self.risk_factors[component].get("weight", 1.0)
                    
                    if error < 0.2:  # Good prediction
                        new_weight = min(2.0, current_weight * 1.05)
                    else:  # Poor prediction
                        new_weight = max(0.5, current_weight * 0.95)
                    
                    self.risk_factors[component]["weight"] = new_weight
                    self.risk_factors[component]["last_updated"] = datetime.utcnow().isoformat()
            
            # Update confidence interval models
            await self._update_confidence_models(feedback)
            
            logger.info(f"Risk assessment learning updated for transaction {transaction_id}")
            
        except Exception as e:
            logger.error(f"Risk assessment learning failed: {str(e)}")
    
    async def get_reasoning(self, input_data: Dict[str, Any]) -> str:
        """Get human-readable reasoning for risk assessment"""
        transaction = input_data.get("transaction", {})
        
        reasoning_prompt = f"""
        Provide a clear explanation of the risk assessment for this transaction:
        
        Transaction Details:
        - Amount: ${transaction.get('amount', 0):,.2f}
        - Time: {transaction.get('timestamp', 'unknown')}
        - Merchant: {transaction.get('merchant_category', 'unknown')}
        - Location: {transaction.get('location', {}).get('country', 'unknown')}
        
        Consider the following risk factors:
        1. Transaction amount relative to user's typical spending
        2. Time of transaction and user's typical activity patterns
        3. Geographic location and travel patterns
        4. Transaction velocity and frequency
        5. Merchant category and risk level
        
        Explain step-by-step how each factor contributes to the overall risk assessment.
        """
        
        ai_reasoning = await self._generate_ai_response(reasoning_prompt, max_tokens=400)
        
        if ai_reasoning:
            return ai_reasoning
        
        # Fallback reasoning
        return f"Risk assessment analyzed multiple factors including amount, timing, location, and user behavior patterns to determine fraud probability."
    
    async def _assess_amount_risk(self, transaction_data: Dict[str, Any], reasoning_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess risk based on transaction amount"""
        amount = transaction_data.get("amount", 0)
        model = self.risk_models["amount_risk"]
        
        # Determine amount risk level
        risk_level = "low"
        risk_score = 0.1
        
        for level, threshold in model["thresholds"].items():
            if amount >= threshold:
                risk_level = level
                risk_score = model["weights"][level]
        
        reasoning_step = {
            "factor": "amount_risk",
            "reasoning": f"Transaction amount ${amount:,.2f} classified as {risk_level} risk",
            "risk_score": risk_score,
            "evidence": {"amount": amount, "risk_level": risk_level}
        }
        reasoning_chain.append(reasoning_step)
        
        return {
            "score": risk_score,
            "level": risk_level,
            "amount": amount,
            "reasoning": reasoning_step["reasoning"]
        }
    
    async def _assess_velocity_risk(self, context_data: Dict[str, Any], reasoning_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess risk based on transaction velocity"""
        tx_count_1h = context_data.get("tx_count_1h", 0)
        tx_count_24h = context_data.get("tx_count_24h", 0)
        amount_1h = context_data.get("amount_1h", 0)
        
        model = self.risk_models["velocity_risk"]
        risk_factors = []
        total_risk = 0.0
        
        # Check 1-hour transaction count
        if tx_count_1h >= model["thresholds"]["tx_1h"]:
            factor_risk = min(1.0, tx_count_1h / 10) * model["weights"]["tx_1h"]
            total_risk += factor_risk
            risk_factors.append(f"{tx_count_1h} transactions in 1 hour")
        
        # Check 24-hour transaction count
        if tx_count_24h >= model["thresholds"]["tx_24h"]:
            factor_risk = min(1.0, tx_count_24h / 50) * model["weights"]["tx_24h"]
            total_risk += factor_risk
            risk_factors.append(f"{tx_count_24h} transactions in 24 hours")
        
        # Check 1-hour amount
        if amount_1h >= model["thresholds"]["amount_1h"]:
            factor_risk = min(1.0, amount_1h / 20000) * model["weights"]["amount_1h"]
            total_risk += factor_risk
            risk_factors.append(f"${amount_1h:,.2f} in 1 hour")
        
        risk_score = min(1.0, total_risk)
        reasoning_step = {
            "factor": "velocity_risk",
            "reasoning": f"Velocity analysis: {', '.join(risk_factors) if risk_factors else 'Normal transaction velocity'}",
            "risk_score": risk_score,
            "evidence": {"tx_1h": tx_count_1h, "tx_24h": tx_count_24h, "amount_1h": amount_1h}
        }
        reasoning_chain.append(reasoning_step)
        
        return {
            "score": risk_score,
            "factors": risk_factors,
            "tx_count_1h": tx_count_1h,
            "tx_count_24h": tx_count_24h,
            "amount_1h": amount_1h,
            "reasoning": reasoning_step["reasoning"]
        }
    
    async def _assess_geographic_risk(self, transaction_data: Dict[str, Any], reasoning_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess risk based on geographic location"""
        location = transaction_data.get("location", {})
        country = location.get("country", "unknown")
        
        model = self.risk_models["geographic_risk"]
        
        if country == "unknown":
            risk_score = model["risk_scores"]["unknown"]
            risk_level = "medium"
            reasoning = "Unknown location increases risk"
        elif country in model["high_risk_countries"]:
            risk_score = model["risk_scores"]["high_risk"]
            risk_level = "high"
            reasoning = f"Transaction from high-risk country: {country}"
        elif country in model["medium_risk_countries"]:
            risk_score = model["risk_scores"]["medium_risk"]
            risk_level = "medium"
            reasoning = f"Transaction from medium-risk country: {country}"
        else:
            risk_score = model["risk_scores"]["domestic"]
            risk_level = "low"
            reasoning = f"Transaction from low-risk country: {country}"
        
        reasoning_step = {
            "factor": "geographic_risk",
            "reasoning": reasoning,
            "risk_score": risk_score,
            "evidence": {"country": country, "risk_level": risk_level}
        }
        reasoning_chain.append(reasoning_step)
        
        return {
            "score": risk_score,
            "country": country,
            "risk_level": risk_level,
            "reasoning": reasoning
        }
    
    async def _assess_temporal_risk(self, transaction_data: Dict[str, Any], reasoning_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess risk based on transaction timing"""
        try:
            timestamp = transaction_data.get("timestamp", "")
            tx_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            hour = tx_time.hour
            is_weekend = tx_time.weekday() >= 5
        except:
            reasoning_step = {
                "factor": "temporal_risk",
                "reasoning": "Invalid timestamp, assuming medium risk",
                "risk_score": 0.5,
                "evidence": {"timestamp": "invalid"}
            }
            reasoning_chain.append(reasoning_step)
            return {"score": 0.5, "reasoning": "Invalid timestamp"}
        
        model = self.risk_models["temporal_risk"]
        risk_score = 0.1  # Base risk
        risk_factors = []
        
        # Check for high-risk hours
        if hour in model["high_risk_hours"]:
            risk_score += 0.4
            risk_factors.append(f"Transaction at {hour}:00 (high-risk hour)")
        
        # Check weekend multiplier
        if is_weekend:
            risk_score *= model["weekend_multiplier"]
            risk_factors.append("Weekend transaction")
        
        risk_score = min(1.0, risk_score)
        
        reasoning_step = {
            "factor": "temporal_risk",
            "reasoning": f"Time-based risk: {', '.join(risk_factors) if risk_factors else 'Normal business hours'}",
            "risk_score": risk_score,
            "evidence": {"hour": hour, "is_weekend": is_weekend, "risk_factors": risk_factors}
        }
        reasoning_chain.append(reasoning_step)
        
        return {
            "score": risk_score,
            "hour": hour,
            "is_weekend": is_weekend,
            "risk_factors": risk_factors,
            "reasoning": reasoning_step["reasoning"]
        }
    
    async def _assess_merchant_risk(self, transaction_data: Dict[str, Any], reasoning_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess risk based on merchant category"""
        merchant_category = transaction_data.get("merchant_category", "unknown")
        model = self.risk_models["merchant_risk"]
        
        if merchant_category in model["high_risk_categories"]:
            risk_score = model["risk_scores"]["high_risk"]
            risk_level = "high"
            reasoning = f"High-risk merchant category: {merchant_category}"
        elif merchant_category in model["medium_risk_categories"]:
            risk_score = model["risk_scores"]["medium_risk"]
            risk_level = "medium"
            reasoning = f"Medium-risk merchant category: {merchant_category}"
        else:
            risk_score = model["risk_scores"]["low_risk"]
            risk_level = "low"
            reasoning = f"Low-risk merchant category: {merchant_category}"
        
        reasoning_step = {
            "factor": "merchant_risk",
            "reasoning": reasoning,
            "risk_score": risk_score,
            "evidence": {"merchant_category": merchant_category, "risk_level": risk_level}
        }
        reasoning_chain.append(reasoning_step)
        
        return {
            "score": risk_score,
            "merchant_category": merchant_category,
            "risk_level": risk_level,
            "reasoning": reasoning
        }
    
    async def _assess_behavioral_risk(self, transaction_data: Dict[str, Any], user_profile: Dict[str, Any], reasoning_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess risk based on behavioral patterns"""
        if not user_profile:
            reasoning_step = {
                "factor": "behavioral_risk",
                "reasoning": "No user profile available, assuming medium risk",
                "risk_score": 0.5,
                "evidence": {"profile_available": False}
            }
            reasoning_chain.append(reasoning_step)
            return {"score": 0.5, "reasoning": "No user profile"}
        
        risk_score = 0.0
        risk_factors = []
        
        # Check amount deviation from typical spending
        typical_amount = user_profile.get("avg_transaction_amount", 100)
        current_amount = transaction_data.get("amount", 0)
        
        if current_amount > typical_amount * 5:
            risk_score += 0.4
            risk_factors.append(f"Amount {current_amount/typical_amount:.1f}x typical spending")
        elif current_amount > typical_amount * 2:
            risk_score += 0.2
            risk_factors.append(f"Amount {current_amount/typical_amount:.1f}x typical spending")
        
        # Check merchant category deviation
        typical_categories = user_profile.get("frequent_categories", [])
        current_category = transaction_data.get("merchant_category", "unknown")
        
        if current_category not in typical_categories and current_category != "unknown":
            risk_score += 0.3
            risk_factors.append(f"Unusual merchant category: {current_category}")
        
        # Check device/location consistency
        if not transaction_data.get("device_info") and user_profile.get("typically_has_device_info", True):
            risk_score += 0.2
            risk_factors.append("Missing device information")
        
        risk_score = min(1.0, risk_score)
        
        reasoning_step = {
            "factor": "behavioral_risk",
            "reasoning": f"Behavioral analysis: {', '.join(risk_factors) if risk_factors else 'Consistent with user behavior'}",
            "risk_score": risk_score,
            "evidence": {
                "typical_amount": typical_amount,
                "current_amount": current_amount,
                "typical_categories": typical_categories,
                "current_category": current_category,
                "risk_factors": risk_factors
            }
        }
        reasoning_chain.append(reasoning_step)
        
        return {
            "score": risk_score,
            "risk_factors": risk_factors,
            "reasoning": reasoning_step["reasoning"]
        }
    
    async def _calculate_composite_risk(self, risk_components: Dict[str, Dict[str, Any]], reasoning_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate composite risk score from individual components"""
        # Default weights for risk components
        default_weights = {
            "amount": 0.20,
            "velocity": 0.20,
            "geographic": 0.20,
            "temporal": 0.15,
            "merchant": 0.15,
            "behavioral": 0.10
        }
        
        # Use learned weights if available
        weights = {}
        for component in default_weights:
            if component in self.risk_factors and "weight" in self.risk_factors[component]:
                weights[component] = self.risk_factors[component]["weight"]
            else:
                weights[component] = default_weights[component]
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted score
        weighted_score = 0.0
        component_contributions = {}
        
        for component, risk_data in risk_components.items():
            if component in weights:
                component_score = risk_data.get("score", 0.0)
                contribution = weights[component] * component_score
                weighted_score += contribution
                component_contributions[component] = {
                    "score": component_score,
                    "weight": weights[component],
                    "contribution": contribution
                }
        
        # Apply non-linear scaling for extreme cases
        if weighted_score > 0.8:
            final_score = 0.8 + (weighted_score - 0.8) * 1.5  # Amplify high risk
        elif weighted_score < 0.2:
            final_score = weighted_score * 0.5  # Reduce very low risk
        else:
            final_score = weighted_score
        
        final_score = min(1.0, max(0.0, final_score))
        
        reasoning_step = {
            "factor": "composite_calculation",
            "reasoning": f"Weighted composite score: {final_score:.3f} from {len(risk_components)} components",
            "risk_score": final_score,
            "evidence": {
                "component_contributions": component_contributions,
                "weights": weights,
                "weighted_score": weighted_score,
                "final_score": final_score
            }
        }
        reasoning_chain.append(reasoning_step)
        
        return {
            "final_score": final_score,
            "weighted_score": weighted_score,
            "component_contributions": component_contributions,
            "weights_used": weights
        }
    
    async def _calculate_confidence_interval(self, composite_risk: Dict[str, Any], transaction_data: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence interval for risk assessment"""
        base_score = composite_risk["final_score"]
        
        # Factors affecting confidence
        data_completeness = 0.0
        data_completeness += 0.2 if transaction_data.get("amount") else 0
        data_completeness += 0.2 if transaction_data.get("timestamp") else 0
        data_completeness += 0.2 if transaction_data.get("location") else 0
        data_completeness += 0.2 if transaction_data.get("merchant_category") else 0
        data_completeness += 0.2 if context_data else 0
        
        # Historical accuracy
        historical_accuracy = (self.performance_metrics["correct_decisions"] / 
                             max(1, self.performance_metrics["total_decisions"]))
        
        # Confidence width (narrower = more confident)
        confidence_width = 0.1 + 0.2 * (1 - data_completeness) + 0.1 * (1 - historical_accuracy)
        
        lower_bound = max(0.0, base_score - confidence_width)
        upper_bound = min(1.0, base_score + confidence_width)
        
        return {
            "point_estimate": base_score,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "confidence_width": confidence_width * 2,
            "confidence_level": 0.95,
            "data_completeness": data_completeness,
            "historical_accuracy": historical_accuracy
        }
    
    async def _generate_risk_explanation(self, reasoning_chain: List[Dict[str, Any]]) -> str:
        """Generate human-readable risk explanation"""
        if not reasoning_chain:
            return "No risk factors analyzed."
        
        explanations = []
        for step in reasoning_chain:
            if step.get("risk_score", 0) > 0.1:  # Only include significant factors
                explanations.append(step["reasoning"])
        
        if not explanations:
            return "All risk factors are within normal ranges."
        
        return " | ".join(explanations)
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine categorical risk level from numeric score"""
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    async def _generate_recommendations(self, composite_risk: Dict[str, Any], risk_level: str, reasoning_chain: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on risk assessment"""
        recommendations = []
        risk_score = composite_risk["final_score"]
        
        if risk_level == "critical":
            recommendations.append("Immediately decline transaction and flag account for review")
            recommendations.append("Notify fraud team for urgent investigation")
        elif risk_level == "high":
            recommendations.append("Hold transaction for manual review")
            recommendations.append("Request additional authentication from user")
        elif risk_level == "medium":
            recommendations.append("Apply enhanced monitoring for this transaction")
            recommendations.append("Consider step-up authentication")
        else:
            recommendations.append("Process transaction with standard monitoring")
        
        # Add specific recommendations based on risk factors
        high_risk_factors = [step for step in reasoning_chain if step.get("risk_score", 0) > 0.5]
        
        for factor in high_risk_factors:
            factor_type = factor.get("factor", "")
            if factor_type == "velocity_risk":
                recommendations.append("Monitor user for unusual transaction patterns")
            elif factor_type == "geographic_risk":
                recommendations.append("Verify user location and travel status")
            elif factor_type == "behavioral_risk":
                recommendations.append("Review user's recent spending patterns")
        
        return recommendations
    
    async def _update_confidence_models(self, feedback: Dict[str, Any]) -> None:
        """Update confidence interval models based on feedback"""
        try:
            predicted_interval = feedback.get("confidence_interval", {})
            actual_fraud = feedback.get("actual_fraud", False)
            
            if predicted_interval:
                lower = predicted_interval.get("lower_bound", 0)
                upper = predicted_interval.get("upper_bound", 1)
                actual_value = 1.0 if actual_fraud else 0.0
                
                # Check if actual value fell within predicted interval
                within_interval = lower <= actual_value <= upper
                
                # Update confidence model accuracy
                confidence_key = "confidence_accuracy"
                if confidence_key not in self.knowledge_base:
                    self.knowledge_base[confidence_key] = {"correct": 0, "total": 0}
                
                self.knowledge_base[confidence_key]["total"] += 1
                if within_interval:
                    self.knowledge_base[confidence_key]["correct"] += 1
                
        except Exception as e:
            logger.error(f"Confidence model update failed: {str(e)}")
    
    async def get_risk_factor_importance(self) -> Dict[str, float]:
        """Get current importance weights for risk factors"""
        importance = {}
        
        for factor_name, factor_data in self.risk_factors.items():
            importance[factor_name] = factor_data.get("weight", 1.0)
        
        # Add default factors if not present
        default_factors = ["amount", "velocity", "geographic", "temporal", "behavioral"]
        for factor in default_factors:
            if factor not in importance:
                importance[factor] = 1.0
        
        return importance