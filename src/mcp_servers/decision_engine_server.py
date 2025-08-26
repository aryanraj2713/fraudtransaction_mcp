import asyncio
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import uuid
from enum import Enum
import math

from .base_server import BaseMCPServer
from schemas.mcp_tools_schema import DECISION_ENGINE_TOOLS
from schemas.transaction_schema import Transaction, FraudScore, FraudAlert

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionType(str, Enum):
    APPROVE = "approve"
    DECLINE = "decline"
    REVIEW = "review"
    ESCALATE = "escalate"


@dataclass
class RiskAssessment:
    transaction_id: str
    overall_risk_score: float
    confidence_score: float
    risk_level: RiskLevel
    risk_factors: List[str]
    protective_factors: List[str]
    assessment_timestamp: datetime
    individual_scores: Dict[str, float]
    uncertainty_metrics: Dict[str, float]


@dataclass
class DecisionContext:
    transaction: Dict[str, Any]
    risk_assessment: RiskAssessment
    business_rules: Dict[str, Any]
    cost_matrix: Dict[str, float]
    user_context: Dict[str, Any]


@dataclass
class FraudDecision:
    transaction_id: str
    decision: DecisionType
    confidence: float
    risk_score: float
    reasoning: List[str]
    recommendation: str
    review_priority: Optional[str]
    decision_timestamp: datetime
    processing_time_ms: float
    metadata: Dict[str, Any]


class RiskAssessor:
    """Advanced risk assessment system with uncertainty quantification."""
    
    def __init__(self):
        self.risk_thresholds = {
            RiskLevel.MINIMAL: 0.1,
            RiskLevel.LOW: 0.25,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRITICAL: 0.9
        }
        
        self.risk_factors_weights = {
            # Velocity-based factors
            "high_transaction_velocity": 0.15,
            "unusual_time_pattern": 0.10,
            "velocity_spike": 0.12,
            
            # Amount-based factors
            "unusual_amount": 0.18,
            "round_amount_pattern": 0.05,
            "amount_vs_history": 0.15,
            
            # Geographic factors
            "geographic_anomaly": 0.20,
            "high_risk_location": 0.25,
            "location_velocity": 0.10,
            
            # Device and behavioral factors
            "new_device": 0.08,
            "suspicious_behavior": 0.12,
            "device_fingerprint_anomaly": 0.15,
            
            # Account factors
            "new_account": 0.10,
            "account_history_flags": 0.20,
            "payment_method_risk": 0.12
        }
    
    async def assess_risk(
        self,
        transaction: Dict[str, Any],
        user_context: Dict[str, Any] = None,
        risk_tolerance: str = "moderate"
    ) -> RiskAssessment:
        """Perform comprehensive risk assessment with uncertainty quantification."""
        
        try:
            start_time = datetime.now()
            transaction_id = transaction.get("transaction_id", "unknown")
            
            if user_context is None:
                user_context = {}
            
            # Calculate individual risk scores
            individual_scores = {}
            risk_factors = []
            protective_factors = []
            
            # Velocity risk assessment
            velocity_score, velocity_factors = self._assess_velocity_risk(transaction, user_context)
            individual_scores["velocity"] = velocity_score
            risk_factors.extend(velocity_factors["risk"])
            protective_factors.extend(velocity_factors["protective"])
            
            # Amount risk assessment
            amount_score, amount_factors = self._assess_amount_risk(transaction, user_context)
            individual_scores["amount"] = amount_score
            risk_factors.extend(amount_factors["risk"])
            protective_factors.extend(amount_factors["protective"])
            
            # Geographic risk assessment
            geo_score, geo_factors = self._assess_geographic_risk(transaction, user_context)
            individual_scores["geographic"] = geo_score
            risk_factors.extend(geo_factors["risk"])
            protective_factors.extend(geo_factors["protective"])
            
            # Device and behavioral risk assessment
            device_score, device_factors = self._assess_device_risk(transaction, user_context)
            individual_scores["device_behavior"] = device_score
            risk_factors.extend(device_factors["risk"])
            protective_factors.extend(device_factors["protective"])
            
            # Account risk assessment
            account_score, account_factors = self._assess_account_risk(transaction, user_context)
            individual_scores["account"] = account_score
            risk_factors.extend(account_factors["risk"])
            protective_factors.extend(account_factors["protective"])
            
            # Calculate overall risk score with weights
            risk_weights = {
                "velocity": 0.20,
                "amount": 0.25,
                "geographic": 0.25,
                "device_behavior": 0.15,
                "account": 0.15
            }
            
            overall_risk_score = sum(
                individual_scores[component] * risk_weights[component]
                for component in individual_scores
            )
            
            # Apply risk tolerance adjustment
            overall_risk_score = self._apply_risk_tolerance(overall_risk_score, risk_tolerance)
            
            # Calculate confidence score based on data quality and consistency
            confidence_score = self._calculate_confidence(individual_scores, transaction, user_context)
            
            # Calculate uncertainty metrics
            uncertainty_metrics = self._calculate_uncertainty_metrics(individual_scores, confidence_score)
            
            # Determine risk level
            risk_level = self._determine_risk_level(overall_risk_score)
            
            assessment = RiskAssessment(
                transaction_id=transaction_id,
                overall_risk_score=overall_risk_score,
                confidence_score=confidence_score,
                risk_level=risk_level,
                risk_factors=list(set(risk_factors)),  # Remove duplicates
                protective_factors=list(set(protective_factors)),
                assessment_timestamp=datetime.now(),
                individual_scores=individual_scores,
                uncertainty_metrics=uncertainty_metrics
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(
                f"Risk assessment completed for {transaction_id}: "
                f"score={overall_risk_score:.3f}, level={risk_level.value}, "
                f"confidence={confidence_score:.3f}, time={processing_time:.1f}ms"
            )
            
            return assessment
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            raise
    
    def _assess_velocity_risk(self, transaction: Dict[str, Any], user_context: Dict[str, Any]) -> Tuple[float, Dict[str, List[str]]]:
        """Assess velocity-based risk factors."""
        score = 0.0
        factors = {"risk": [], "protective": []}
        
        velocity_1h = transaction.get("velocity_1h", 0)
        velocity_24h = transaction.get("velocity_24h", 0)
        hour = datetime.fromisoformat(transaction.get("timestamp", datetime.now().isoformat())).hour
        
        # High velocity risk
        if velocity_1h > 10:
            score += 0.3
            factors["risk"].append(f"High hourly velocity: {velocity_1h} transactions")
        elif velocity_1h > 5:
            score += 0.15
            factors["risk"].append(f"Elevated hourly velocity: {velocity_1h} transactions")
        
        if velocity_24h > 50:
            score += 0.4
            factors["risk"].append(f"High daily velocity: {velocity_24h} transactions")
        elif velocity_24h > 20:
            score += 0.2
            factors["risk"].append(f"Elevated daily velocity: {velocity_24h} transactions")
        
        # Unusual time patterns
        if hour < 6 or hour > 23:
            score += 0.1
            factors["risk"].append("Transaction during unusual hours")
        
        # Velocity spike detection
        avg_daily = user_context.get("avg_daily_transactions", 5)
        if velocity_24h > avg_daily * 3:
            score += 0.25
            factors["risk"].append("Significant velocity spike detected")
        
        # Protective factors
        if velocity_1h <= 1 and velocity_24h <= 5:
            factors["protective"].append("Normal transaction velocity")
        
        if 9 <= hour <= 17:  # Business hours
            factors["protective"].append("Transaction during business hours")
        
        return min(score, 1.0), factors
    
    def _assess_amount_risk(self, transaction: Dict[str, Any], user_context: Dict[str, Any]) -> Tuple[float, Dict[str, List[str]]]:
        """Assess amount-based risk factors."""
        score = 0.0
        factors = {"risk": [], "protective": []}
        
        amount = float(transaction.get("amount", 0))
        user_avg_amount = user_context.get("avg_amount", amount)
        
        # Unusual amount patterns
        if amount > user_avg_amount * 10:
            score += 0.4
            factors["risk"].append(f"Amount significantly above user average: ${amount:.2f} vs ${user_avg_amount:.2f}")
        elif amount > user_avg_amount * 5:
            score += 0.2
            factors["risk"].append(f"Amount above user average: ${amount:.2f} vs ${user_avg_amount:.2f}")
        
        # Round amount patterns (potential fraud indicator)
        if amount == round(amount) and amount >= 100:
            score += 0.05
            factors["risk"].append("Round amount pattern detected")
        
        # High absolute amounts
        if amount > 10000:
            score += 0.3
            factors["risk"].append(f"High transaction amount: ${amount:.2f}")
        elif amount > 5000:
            score += 0.15
            factors["risk"].append(f"Elevated transaction amount: ${amount:.2f}")
        
        # Very small amounts (testing pattern)
        if amount < 1:
            score += 0.1
            factors["risk"].append("Very small amount - potential testing pattern")
        
        # Protective factors
        if 0.5 * user_avg_amount <= amount <= 2 * user_avg_amount:
            factors["protective"].append("Amount within normal user range")
        
        if 10 <= amount <= 500:  # Typical retail range
            factors["protective"].append("Amount in typical transaction range")
        
        return min(score, 1.0), factors
    
    def _assess_geographic_risk(self, transaction: Dict[str, Any], user_context: Dict[str, Any]) -> Tuple[float, Dict[str, List[str]]]:
        """Assess geographic risk factors."""
        score = 0.0
        factors = {"risk": [], "protective": []}
        
        country = transaction.get("country", "")
        city = transaction.get("city", "")
        ip_address = transaction.get("ip_address", "")
        
        # High-risk countries (simplified list)
        high_risk_countries = ["XX", "YY", "ZZ"]  # Placeholder codes
        if country in high_risk_countries:
            score += 0.4
            factors["risk"].append(f"Transaction from high-risk country: {country}")
        
        # Geographic velocity (simplified)
        user_countries = user_context.get("recent_countries", [country])
        if len(set(user_countries)) > 3:  # Many countries recently
            score += 0.2
            factors["risk"].append("Multiple recent geographic locations")
        
        # IP address analysis (simplified)
        if self._is_suspicious_ip(ip_address):
            score += 0.15
            factors["risk"].append("Suspicious IP address detected")
        
        # Distance-based risk (simplified)
        user_location = user_context.get("primary_location", {})
        if user_location and country != user_location.get("country"):
            score += 0.1
            factors["risk"].append("Transaction from different country than usual")
        
        # Protective factors
        if country in ["US", "CA", "GB", "DE", "FR", "AU"]:  # Low-risk countries
            factors["protective"].append("Transaction from low-risk country")
        
        if country == user_location.get("country", country):
            factors["protective"].append("Transaction from user's primary country")
        
        return min(score, 1.0), factors
    
    def _assess_device_risk(self, transaction: Dict[str, Any], user_context: Dict[str, Any]) -> Tuple[float, Dict[str, List[str]]]:
        """Assess device and behavioral risk factors."""
        score = 0.0
        factors = {"risk": [], "protective": []}
        
        device_id = transaction.get("device_id", "")
        device_type = transaction.get("device_type", "")
        is_mobile = transaction.get("is_mobile", False)
        
        # New device risk
        known_devices = user_context.get("known_devices", [device_id])
        if device_id not in known_devices:
            score += 0.15
            factors["risk"].append("Transaction from new/unknown device")
        
        # Behavioral patterns
        session_duration = transaction.get("session_duration", 0)
        if session_duration < 30:  # Very short session
            score += 0.1
            factors["risk"].append("Very short session duration")
        
        pages_visited = transaction.get("pages_visited", 1)
        if pages_visited < 2:  # Minimal browsing
            score += 0.08
            factors["risk"].append("Minimal website interaction")
        
        # Device type patterns
        if device_type == "unknown":
            score += 0.05
            factors["risk"].append("Unknown device type")
        
        # Protective factors
        if device_id in known_devices:
            factors["protective"].append("Transaction from recognized device")
        
        if session_duration > 300:  # 5+ minutes
            factors["protective"].append("Normal session duration")
        
        if pages_visited >= 3:
            factors["protective"].append("Normal browsing behavior")
        
        return min(score, 1.0), factors
    
    def _assess_account_risk(self, transaction: Dict[str, Any], user_context: Dict[str, Any]) -> Tuple[float, Dict[str, List[str]]]:
        """Assess account-related risk factors."""
        score = 0.0
        factors = {"risk": [], "protective": []}
        
        account_age_days = transaction.get("account_age_days", 0)
        is_first_transaction = transaction.get("is_first_transaction", False)
        payment_method = transaction.get("payment_method", "")
        
        # New account risk
        if account_age_days < 1:
            score += 0.3
            factors["risk"].append("Very new account (< 1 day)")
        elif account_age_days < 7:
            score += 0.15
            factors["risk"].append("New account (< 1 week)")
        elif account_age_days < 30:
            score += 0.05
            factors["risk"].append("Recent account (< 1 month)")
        
        # First transaction risk
        if is_first_transaction:
            score += 0.1
            factors["risk"].append("First transaction on account")
        
        # Payment method risk
        high_risk_methods = ["prepaid_card", "cryptocurrency", "wire_transfer"]
        if payment_method in high_risk_methods:
            score += 0.2
            factors["risk"].append(f"High-risk payment method: {payment_method}")
        
        # Account history flags
        account_flags = user_context.get("account_flags", [])
        if account_flags:
            score += 0.15 * len(account_flags)
            factors["risk"].append(f"Account has {len(account_flags)} historical flags")
        
        # Protective factors
        if account_age_days > 365:
            factors["protective"].append("Established account (> 1 year)")
        elif account_age_days > 90:
            factors["protective"].append("Mature account (> 3 months)")
        
        if payment_method in ["credit_card", "debit_card", "bank_transfer"]:
            factors["protective"].append("Standard payment method")
        
        transaction_count = user_context.get("total_transactions", 1)
        if transaction_count > 50:
            factors["protective"].append("Active account with transaction history")
        
        return min(score, 1.0), factors
    
    def _apply_risk_tolerance(self, score: float, tolerance: str) -> float:
        """Apply risk tolerance adjustment to score."""
        if tolerance == "conservative":
            return min(score * 1.2, 1.0)  # Increase risk scores
        elif tolerance == "aggressive":
            return score * 0.8  # Decrease risk scores
        return score  # Moderate - no adjustment
    
    def _calculate_confidence(self, individual_scores: Dict[str, float], transaction: Dict[str, Any], user_context: Dict[str, Any]) -> float:
        """Calculate confidence in the risk assessment."""
        confidence_factors = []
        
        # Data completeness
        required_fields = ["amount", "timestamp", "user_id", "device_id"]
        completeness = sum(1 for field in required_fields if transaction.get(field)) / len(required_fields)
        confidence_factors.append(completeness)
        
        # User context richness
        context_richness = min(len(user_context) / 10, 1.0)  # Normalize to 0-1
        confidence_factors.append(context_richness)
        
        # Score consistency (low variance indicates higher confidence)
        scores = list(individual_scores.values())
        if len(scores) > 1:
            variance = np.var(scores)
            consistency = max(0, 1 - variance)  # Lower variance = higher confidence
            confidence_factors.append(consistency)
        
        return np.mean(confidence_factors)
    
    def _calculate_uncertainty_metrics(self, individual_scores: Dict[str, float], confidence: float) -> Dict[str, float]:
        """Calculate uncertainty quantification metrics."""
        scores = list(individual_scores.values())
        
        return {
            "epistemic_uncertainty": 1 - confidence,  # Model uncertainty
            "aleatoric_uncertainty": np.std(scores) if len(scores) > 1 else 0.0,  # Data uncertainty
            "total_uncertainty": (1 - confidence) + (np.std(scores) if len(scores) > 1 else 0.0),
            "prediction_interval_width": 2 * np.std(scores) if len(scores) > 1 else 0.1
        }
    
    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level from score."""
        if score >= self.risk_thresholds[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif score >= self.risk_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif score >= self.risk_thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        elif score >= self.risk_thresholds[RiskLevel.LOW]:
            return RiskLevel.LOW
        else:
            return RiskLevel.MINIMAL
    
    def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Simple IP address risk assessment."""
        # Placeholder implementation - would integrate with IP reputation services
        suspicious_patterns = ["10.0.0.", "192.168.", "0.0.0.0"]
        return any(pattern in ip_address for pattern in suspicious_patterns)


class DecisionMaker:
    """Intelligent decision-making system with business rules integration."""
    
    def __init__(self):
        self.default_cost_matrix = {
            "false_positive_cost": 10.0,  # Cost of declining good transaction
            "false_negative_cost": 100.0,  # Cost of approving fraud
            "review_cost": 5.0,  # Cost of human review
            "escalation_cost": 15.0  # Cost of escalation
        }
        
        self.default_business_rules = {
            "max_daily_amount": 10000,
            "max_transaction_count_daily": 100,
            "require_review_above": 5000,
            "auto_decline_above": 0.9,
            "auto_approve_below": 0.1,
            "high_risk_countries_decline": True,
            "new_device_review_threshold": 0.3
        }
    
    async def make_decision(
        self,
        risk_scores: Dict[str, Any],
        business_rules: Dict[str, Any] = None,
        cost_matrix: Dict[str, float] = None
    ) -> FraudDecision:
        """Make final fraud decision with confidence scoring."""
        
        try:
            start_time = datetime.now()
            
            if business_rules is None:
                business_rules = self.default_business_rules.copy()
            
            if cost_matrix is None:
                cost_matrix = self.default_cost_matrix.copy()
            
            # Extract risk information
            overall_risk = risk_scores.get("overall_risk_score", 0.5)
            confidence = risk_scores.get("confidence_score", 0.5)
            risk_level = risk_scores.get("risk_level", "medium")
            transaction_id = risk_scores.get("transaction_id", "unknown")
            risk_factors = risk_scores.get("risk_factors", [])
            
            # Apply business rules
            decision, reasoning = self._apply_business_rules(risk_scores, business_rules)
            
            # If business rules didn't make a decision, use risk-based decision
            if decision is None:
                decision, rule_reasoning = self._make_risk_based_decision(overall_risk, confidence, cost_matrix)
                reasoning.extend(rule_reasoning)
            
            # Determine review priority if needed
            review_priority = None
            if decision == DecisionType.REVIEW:
                review_priority = self._determine_review_priority(overall_risk, risk_level, risk_factors)
            
            # Generate recommendation text
            recommendation = self._generate_recommendation(decision, overall_risk, confidence, reasoning)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            fraud_decision = FraudDecision(
                transaction_id=transaction_id,
                decision=decision,
                confidence=confidence,
                risk_score=overall_risk,
                reasoning=reasoning,
                recommendation=recommendation,
                review_priority=review_priority,
                decision_timestamp=datetime.now(),
                processing_time_ms=processing_time,
                metadata={
                    "risk_level": risk_level,
                    "business_rules_applied": list(business_rules.keys()),
                    "cost_matrix_used": cost_matrix,
                    "decision_factors": risk_factors[:5]  # Top 5 factors
                }
            )
            
            logger.info(
                f"Decision made for {transaction_id}: {decision.value} "
                f"(risk={overall_risk:.3f}, confidence={confidence:.3f}, time={processing_time:.1f}ms)"
            )
            
            return fraud_decision
            
        except Exception as e:
            logger.error(f"Decision making error: {e}")
            raise
    
    def _apply_business_rules(self, risk_scores: Dict[str, Any], business_rules: Dict[str, Any]) -> Tuple[Optional[DecisionType], List[str]]:
        """Apply business rules to make decision."""
        reasoning = []
        
        overall_risk = risk_scores.get("overall_risk_score", 0.5)
        amount = risk_scores.get("amount", 0)
        country = risk_scores.get("country", "")
        is_new_device = "new/unknown device" in " ".join(risk_scores.get("risk_factors", []))
        
        # Auto-decline rules
        if overall_risk >= business_rules.get("auto_decline_above", 0.9):
            reasoning.append(f"Auto-decline: Risk score {overall_risk:.3f} exceeds threshold")
            return DecisionType.DECLINE, reasoning
        
        # High-risk country auto-decline
        high_risk_countries = ["XX", "YY", "ZZ"]  # Example
        if (business_rules.get("high_risk_countries_decline", False) and 
            country in high_risk_countries and overall_risk > 0.3):
            reasoning.append(f"Auto-decline: High-risk country {country} with elevated risk")
            return DecisionType.DECLINE, reasoning
        
        # Auto-approve rules
        if overall_risk <= business_rules.get("auto_approve_below", 0.1):
            reasoning.append(f"Auto-approve: Risk score {overall_risk:.3f} below threshold")
            return DecisionType.APPROVE, reasoning
        
        # Review rules
        if amount >= business_rules.get("require_review_above", 5000):
            reasoning.append(f"Require review: Amount ${amount} exceeds review threshold")
            return DecisionType.REVIEW, reasoning
        
        if is_new_device and overall_risk >= business_rules.get("new_device_review_threshold", 0.3):
            reasoning.append("Require review: New device with elevated risk")
            return DecisionType.REVIEW, reasoning
        
        # No business rule matched
        return None, reasoning
    
    def _make_risk_based_decision(self, risk_score: float, confidence: float, cost_matrix: Dict[str, float]) -> Tuple[DecisionType, List[str]]:
        """Make decision based on risk score and cost-benefit analysis."""
        reasoning = []
        
        # Calculate expected costs for each decision
        fp_cost = cost_matrix["false_positive_cost"]
        fn_cost = cost_matrix["false_negative_cost"]
        review_cost = cost_matrix["review_cost"]
        
        # Expected cost of approving
        approve_cost = risk_score * fn_cost
        
        # Expected cost of declining
        decline_cost = (1 - risk_score) * fp_cost
        
        # Expected cost of reviewing (simplified)
        review_expected_cost = review_cost + 0.5 * (approve_cost + decline_cost)
        
        # Adjust for confidence
        if confidence < 0.7:
            # Low confidence - prefer review
            review_expected_cost *= 0.8
            reasoning.append("Low confidence favors review option")
        
        # Make decision based on minimum expected cost
        costs = {
            DecisionType.APPROVE: approve_cost,
            DecisionType.DECLINE: decline_cost,
            DecisionType.REVIEW: review_expected_cost
        }
        
        optimal_decision = min(costs, key=costs.get)
        min_cost = costs[optimal_decision]
        
        reasoning.append(f"Cost-based decision: {optimal_decision.value} (expected cost: ${min_cost:.2f})")
        reasoning.append(f"Risk score: {risk_score:.3f}, Confidence: {confidence:.3f}")
        
        return optimal_decision, reasoning
    
    def _determine_review_priority(self, risk_score: float, risk_level: str, risk_factors: List[str]) -> str:
        """Determine priority for human review."""
        if risk_score > 0.8 or risk_level == "critical":
            return "urgent"
        elif risk_score > 0.6 or risk_level == "high":
            return "high"
        elif len(risk_factors) > 5:
            return "medium"
        else:
            return "normal"
    
    def _generate_recommendation(self, decision: DecisionType, risk_score: float, confidence: float, reasoning: List[str]) -> str:
        """Generate human-readable recommendation."""
        base_recommendation = {
            DecisionType.APPROVE: "Approve transaction - low fraud risk detected",
            DecisionType.DECLINE: "Decline transaction - high fraud risk detected", 
            DecisionType.REVIEW: "Send for manual review - moderate risk indicators present",
            DecisionType.ESCALATE: "Escalate to specialist - complex risk pattern detected"
        }[decision]
        
        context = []
        if confidence < 0.5:
            context.append("Note: Low confidence in assessment")
        if risk_score > 0.7:
            context.append("Warning: High risk indicators present")
        
        if context:
            return f"{base_recommendation}. {' '.join(context)}"
        return base_recommendation


class ExplanationGenerator:
    """Generate human-readable explanations for fraud decisions."""
    
    def __init__(self):
        self.explanation_templates = {
            "technical": {
                "approve": "Technical Analysis: Risk score {risk_score:.3f} below approval threshold. Primary factors: {factors}",
                "decline": "Technical Analysis: Risk score {risk_score:.3f} exceeds decline threshold. Risk factors: {factors}",
                "review": "Technical Analysis: Risk score {risk_score:.3f} requires manual review. Key indicators: {factors}"
            },
            "business": {
                "approve": "Business Assessment: Transaction approved based on low risk profile. {summary}",
                "decline": "Business Assessment: Transaction declined due to high fraud indicators. {summary}",
                "review": "Business Assessment: Transaction flagged for review due to moderate risk. {summary}"
            },
            "customer_facing": {
                "approve": "Your transaction has been approved.",
                "decline": "We're unable to process this transaction due to security concerns. Please contact customer service.",
                "review": "Your transaction is being reviewed for security purposes. You'll be notified of the outcome shortly."
            }
        }
    
    async def generate_explanation(
        self,
        decision: Dict[str, Any],
        explanation_type: str = "business",
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """Generate human-readable explanation for decisions."""
        
        try:
            decision_type = decision.get("decision", "review").lower()
            risk_score = decision.get("risk_score", 0.5)
            reasoning = decision.get("reasoning", [])
            risk_factors = decision.get("metadata", {}).get("decision_factors", [])
            
            # Generate base explanation
            template = self.explanation_templates.get(explanation_type, {}).get(decision_type, "")
            
            if explanation_type == "technical":
                explanation = template.format(
                    risk_score=risk_score,
                    factors=", ".join(risk_factors[:3]) if risk_factors else "Standard risk assessment"
                )
            elif explanation_type == "business":
                summary = self._generate_business_summary(decision)
                explanation = template.format(summary=summary)
            else:  # customer_facing
                explanation = template
            
            # Add detailed reasoning for technical/business types
            detailed_reasoning = []
            if explanation_type in ["technical", "business"]:
                detailed_reasoning = reasoning[:5]  # Top 5 reasons
            
            # Generate recommendations if requested
            recommendations = []
            if include_recommendations and explanation_type != "customer_facing":
                recommendations = self._generate_recommendations(decision)
            
            result = {
                "explanation": explanation,
                "detailed_reasoning": detailed_reasoning,
                "recommendations": recommendations,
                "explanation_type": explanation_type,
                "confidence": decision.get("confidence", 0.5),
                "risk_level": decision.get("metadata", {}).get("risk_level", "medium"),
                "generated_at": datetime.now().isoformat()
            }
            
            logger.info(f"Generated {explanation_type} explanation for decision {decision_type}")
            return result
            
        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
            return {
                "explanation": "Unable to generate explanation due to processing error",
                "error": str(e)
            }
    
    def _generate_business_summary(self, decision: Dict[str, Any]) -> str:
        """Generate business-friendly summary."""
        risk_score = decision.get("risk_score", 0.5)
        risk_level = decision.get("metadata", {}).get("risk_level", "medium")
        
        if risk_score > 0.7:
            return "Multiple high-risk indicators detected requiring immediate attention"
        elif risk_score > 0.4:
            return f"Moderate risk profile ({risk_level} level) identified through pattern analysis"
        else:
            return "Low risk profile with standard transaction patterns"
    
    def _generate_recommendations(self, decision: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        decision_type = decision.get("decision", "review")
        risk_score = decision.get("risk_score", 0.5)
        
        if decision_type == "decline":
            recommendations.extend([
                "Block transaction immediately",
                "Alert fraud team for investigation",
                "Monitor account for additional suspicious activity"
            ])
        elif decision_type == "review":
            recommendations.extend([
                "Conduct manual review within 2 hours",
                "Verify transaction with additional authentication if approved",
                "Document review findings for pattern analysis"
            ])
            
            if risk_score > 0.6:
                recommendations.append("Prioritize for urgent review")
                
        elif decision_type == "approve":
            recommendations.extend([
                "Process transaction normally",
                "Continue monitoring account activity"
            ])
        
        return recommendations


class EscalationManager:
    """Manages transaction escalation for human review."""
    
    def __init__(self):
        self.escalation_queues = {
            "urgent": [],
            "high": [],
            "medium": [],
            "normal": []
        }
        
        self.reviewer_expertise = {
            "fraud_specialist": ["high_risk_patterns", "account_takeover", "synthetic_identity"],
            "risk_analyst": ["velocity_patterns", "geographic_anomalies", "device_analysis"],
            "senior_analyst": ["complex_schemes", "business_logic", "cost_optimization"]
        }
    
    async def escalate_for_review(
        self,
        transaction_id: str,
        escalation_reason: str,
        priority: str = "medium",
        reviewer_expertise: str = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Escalate transactions for human review."""
        
        try:
            if context is None:
                context = {}
            
            escalation_id = f"esc_{uuid.uuid4().hex[:8]}"
            
            escalation_record = {
                "escalation_id": escalation_id,
                "transaction_id": transaction_id,
                "reason": escalation_reason,
                "priority": priority,
                "created_at": datetime.now().isoformat(),
                "status": "pending",
                "assigned_to": None,
                "context": context,
                "required_expertise": reviewer_expertise
            }
            
            # Add to appropriate queue
            if priority in self.escalation_queues:
                self.escalation_queues[priority].append(escalation_record)
            else:
                self.escalation_queues["medium"].append(escalation_record)
            
            # Auto-assign if possible
            assigned_reviewer = self._auto_assign_reviewer(escalation_record)
            if assigned_reviewer:
                escalation_record["assigned_to"] = assigned_reviewer
                escalation_record["status"] = "assigned"
            
            # Calculate estimated review time
            estimated_review_time = self._estimate_review_time(priority, len(self.escalation_queues[priority]))
            
            result = {
                "status": "success",
                "escalation_id": escalation_id,
                "transaction_id": transaction_id,
                "queue_priority": priority,
                "queue_position": len(self.escalation_queues[priority]),
                "assigned_reviewer": assigned_reviewer,
                "estimated_review_time_minutes": estimated_review_time,
                "escalation_timestamp": escalation_record["created_at"],
                "context_provided": bool(context)
            }
            
            logger.info(f"Escalated transaction {transaction_id} to {priority} priority queue (ID: {escalation_id})")
            return result
            
        except Exception as e:
            logger.error(f"Escalation error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "transaction_id": transaction_id
            }
    
    def _auto_assign_reviewer(self, escalation_record: Dict[str, Any]) -> Optional[str]:
        """Auto-assign reviewer based on expertise requirements."""
        required_expertise = escalation_record.get("required_expertise")
        
        if not required_expertise:
            return None
        
        # Simple assignment logic - would be more sophisticated in practice
        for reviewer, expertise_areas in self.reviewer_expertise.items():
            if required_expertise in expertise_areas:
                return reviewer
        
        return None
    
    def _estimate_review_time(self, priority: str, queue_position: int) -> int:
        """Estimate review time based on priority and queue position."""
        base_times = {
            "urgent": 15,    # 15 minutes
            "high": 60,      # 1 hour
            "medium": 240,   # 4 hours
            "normal": 1440   # 24 hours
        }
        
        base_time = base_times.get(priority, 240)
        queue_delay = queue_position * (base_time * 0.1)  # 10% additional time per position
        
        return int(base_time + queue_delay)


class DecisionEngineServer(BaseMCPServer):
    """Decision Engine MCP Server for fraud detection system."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8003):
        super().__init__("DecisionEngine", "1.0.0", host, port)
        
        # Initialize components
        self.risk_assessor = RiskAssessor()
        self.decision_maker = DecisionMaker()
        self.explanation_generator = ExplanationGenerator()
        self.escalation_manager = EscalationManager()
        
        # Decision tracking
        self.decision_history = []
        self.performance_metrics = {
            "decisions_made": 0,
            "auto_approvals": 0,
            "auto_declines": 0,
            "manual_reviews": 0,
            "escalations": 0,
            "avg_processing_time_ms": 0
        }
    
    async def initialize(self):
        """Initialize the Decision Engine server."""
        # Register tools
        for tool in DECISION_ENGINE_TOOLS:
            if tool.name == "assess_risk":
                self.register_tool(tool, self.assess_risk)
            elif tool.name == "make_decision":
                self.register_tool(tool, self.make_decision)
            elif tool.name == "generate_explanation":
                self.register_tool(tool, self.generate_explanation)
            elif tool.name == "escalate_for_review":
                self.register_tool(tool, self.escalate_for_review)
        
        # Add capabilities
        self.add_capability("risk_assessment")
        self.add_capability("decision_making")
        self.add_capability("explanation_generation")
        self.add_capability("escalation_management")
        
        logger.info("Decision Engine MCP Server initialized successfully")
    
    async def shutdown(self):
        """Clean up resources."""
        logger.info("Decision Engine MCP Server shutting down")
    
    async def assess_risk(
        self,
        transaction: Dict[str, Any],
        user_context: Dict[str, Any] = None,
        risk_tolerance: str = "moderate"
    ) -> Dict[str, Any]:
        """Perform comprehensive risk assessment."""
        try:
            if not transaction:
                raise ValueError("Transaction data is required")
            
            # Perform risk assessment
            assessment = await self.risk_assessor.assess_risk(transaction, user_context, risk_tolerance)
            
            result = {
                "status": "success",
                "transaction_id": assessment.transaction_id,
                "assessment_timestamp": assessment.assessment_timestamp.isoformat(),
                "risk_assessment": {
                    "overall_risk_score": assessment.overall_risk_score,
                    "confidence_score": assessment.confidence_score,
                    "risk_level": assessment.risk_level.value,
                    "risk_factors": assessment.risk_factors,
                    "protective_factors": assessment.protective_factors,
                    "individual_scores": assessment.individual_scores,
                    "uncertainty_metrics": assessment.uncertainty_metrics
                },
                "risk_tolerance": risk_tolerance,
                "user_context_provided": user_context is not None
            }
            
            logger.info(f"Risk assessment completed for {assessment.transaction_id}: {assessment.risk_level.value} risk")
            return result
            
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "transaction_id": transaction.get("transaction_id", "unknown")
            }
    
    async def make_decision(
        self,
        risk_scores: Dict[str, Any],
        business_rules: Dict[str, Any] = None,
        cost_matrix: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Make final fraud decision with confidence scoring."""
        try:
            if not risk_scores:
                raise ValueError("Risk scores are required")
            
            # Make decision
            decision = await self.decision_maker.make_decision(risk_scores, business_rules, cost_matrix)
            
            # Update metrics
            self.performance_metrics["decisions_made"] += 1
            if decision.decision == DecisionType.APPROVE:
                self.performance_metrics["auto_approvals"] += 1
            elif decision.decision == DecisionType.DECLINE:
                self.performance_metrics["auto_declines"] += 1
            elif decision.decision == DecisionType.REVIEW:
                self.performance_metrics["manual_reviews"] += 1
            elif decision.decision == DecisionType.ESCALATE:
                self.performance_metrics["escalations"] += 1
            
            # Update average processing time
            current_avg = self.performance_metrics["avg_processing_time_ms"]
            total_decisions = self.performance_metrics["decisions_made"]
            self.performance_metrics["avg_processing_time_ms"] = (
                (current_avg * (total_decisions - 1) + decision.processing_time_ms) / total_decisions
            )
            
            # Store decision history
            self.decision_history.append(asdict(decision))
            
            result = {
                "status": "success",
                "decision": {
                    "transaction_id": decision.transaction_id,
                    "decision": decision.decision.value,
                    "confidence": decision.confidence,
                    "risk_score": decision.risk_score,
                    "reasoning": decision.reasoning,
                    "recommendation": decision.recommendation,
                    "review_priority": decision.review_priority,
                    "decision_timestamp": decision.decision_timestamp.isoformat(),
                    "processing_time_ms": decision.processing_time_ms,
                    "metadata": decision.metadata
                },
                "performance_metrics": self.performance_metrics.copy()
            }
            
            logger.info(f"Decision made for {decision.transaction_id}: {decision.decision.value}")
            return result
            
        except Exception as e:
            logger.error(f"Decision making error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "risk_scores_provided": bool(risk_scores)
            }
    
    async def generate_explanation(
        self,
        decision: Dict[str, Any],
        explanation_type: str = "business",
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """Generate human-readable explanation for decisions."""
        try:
            if not decision:
                raise ValueError("Decision data is required")
            
            # Generate explanation
            explanation = await self.explanation_generator.generate_explanation(
                decision, explanation_type, include_recommendations
            )
            
            result = {
                "status": "success",
                "transaction_id": decision.get("transaction_id", "unknown"),
                "explanation": explanation,
                "explanation_type": explanation_type,
                "recommendations_included": include_recommendations
            }
            
            logger.info(f"Generated {explanation_type} explanation for transaction {decision.get('transaction_id', 'unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "explanation_type": explanation_type
            }
    
    async def escalate_for_review(
        self,
        transaction_id: str,
        escalation_reason: str,
        priority: str = "medium",
        reviewer_expertise: str = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Escalate transactions for human review."""
        try:
            # Escalate transaction
            result = await self.escalation_manager.escalate_for_review(
                transaction_id, escalation_reason, priority, reviewer_expertise, context
            )
            
            if result.get("status") == "success":
                self.performance_metrics["escalations"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Escalation error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "transaction_id": transaction_id
            }


async def main():
    """Main function to run the Decision Engine MCP Server."""
    server = DecisionEngineServer()
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await server.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())