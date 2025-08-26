import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import math

from .base_agent import BaseFraudDetectionAgent
from core.mcp_client import MCPClient

logger = logging.getLogger(__name__)


@dataclass
class RiskFactor:
    factor_name: str
    weight: float
    value: float
    contribution: float
    description: str


class ProbabilisticRiskModel:
    """Probabilistic risk assessment using Bayesian inference."""
    
    def __init__(self):
        self.prior_fraud_rate = 0.02  # 2% base fraud rate
        self.risk_factors = {
            "high_amount": {"sensitivity": 0.8, "specificity": 0.7},
            "new_device": {"sensitivity": 0.6, "specificity": 0.8},
            "unusual_location": {"sensitivity": 0.7, "specificity": 0.75},
            "high_velocity": {"sensitivity": 0.9, "specificity": 0.6},
            "new_account": {"sensitivity": 0.5, "specificity": 0.9}
        }
    
    def calculate_posterior_probability(self, evidence: Dict[str, bool]) -> Dict[str, float]:
        """Calculate posterior fraud probability given evidence."""
        
        # Start with prior
        fraud_prob = self.prior_fraud_rate
        legitimate_prob = 1 - self.prior_fraud_rate
        
        # Apply Bayes theorem for each piece of evidence
        for factor_name, is_present in evidence.items():
            if factor_name in self.risk_factors:
                factor = self.risk_factors[factor_name]
                
                if is_present:
                    # P(Evidence|Fraud) = sensitivity
                    # P(Evidence|Legitimate) = 1 - specificity
                    likelihood_fraud = factor["sensitivity"]
                    likelihood_legit = 1 - factor["specificity"]
                else:
                    # P(Not Evidence|Fraud) = 1 - sensitivity
                    # P(Not Evidence|Legitimate) = specificity
                    likelihood_fraud = 1 - factor["sensitivity"]
                    likelihood_legit = factor["specificity"]
                
                # Update probabilities
                fraud_prob *= likelihood_fraud
                legitimate_prob *= likelihood_legit
        
        # Normalize
        total_prob = fraud_prob + legitimate_prob
        if total_prob > 0:
            fraud_prob /= total_prob
            legitimate_prob /= total_prob
        
        return {
            "fraud_probability": fraud_prob,
            "legitimate_probability": legitimate_prob,
            "confidence": max(fraud_prob, legitimate_prob)
        }


class UncertaintyQuantifier:
    """Quantify uncertainty in risk assessments."""
    
    def __init__(self):
        self.confidence_factors = {
            "data_completeness": 0.3,
            "model_consensus": 0.3,
            "historical_similarity": 0.2,
            "feature_reliability": 0.2
        }
    
    def quantify_uncertainty(
        self, 
        transaction: Dict[str, Any], 
        risk_components: Dict[str, float],
        historical_data: List[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Quantify uncertainty in risk assessment."""
        
        uncertainty_metrics = {}
        
        # Data completeness uncertainty
        completeness = self._assess_data_completeness(transaction)
        uncertainty_metrics["epistemic_uncertainty"] = 1 - completeness
        
        # Model uncertainty (variability in risk components)
        if len(risk_components) > 1:
            risk_variance = np.var(list(risk_components.values()))
            uncertainty_metrics["aleatoric_uncertainty"] = min(1.0, risk_variance * 2)
        else:
            uncertainty_metrics["aleatoric_uncertainty"] = 0.5
        
        # Historical similarity uncertainty
        if historical_data:
            similarity = self._calculate_historical_similarity(transaction, historical_data)
            uncertainty_metrics["similarity_confidence"] = similarity
        else:
            uncertainty_metrics["similarity_confidence"] = 0.3
        
        # Overall uncertainty
        epistemic = uncertainty_metrics["epistemic_uncertainty"]
        aleatoric = uncertainty_metrics["aleatoric_uncertainty"]
        uncertainty_metrics["total_uncertainty"] = math.sqrt(epistemic**2 + aleatoric**2)
        
        # Confidence interval width (approximate)
        uncertainty_metrics["confidence_interval_width"] = (
            2 * uncertainty_metrics["total_uncertainty"]
        )
        
        return uncertainty_metrics
    
    def _assess_data_completeness(self, transaction: Dict[str, Any]) -> float:
        """Assess completeness of transaction data."""
        required_fields = [
            "amount", "timestamp", "user_id", "country", "payment_method",
            "device_id", "ip_address", "velocity_1h", "velocity_24h"
        ]
        
        present_fields = sum(1 for field in required_fields if transaction.get(field) is not None)
        return present_fields / len(required_fields)
    
    def _calculate_historical_similarity(
        self, 
        transaction: Dict[str, Any], 
        historical_data: List[Dict[str, Any]]
    ) -> float:
        """Calculate similarity to historical transactions."""
        
        if not historical_data:
            return 0.0
        
        similarities = []
        current_amount = float(transaction.get("amount", 0))
        current_country = transaction.get("country", "")
        
        for hist_tx in historical_data[-100:]:  # Last 100 transactions
            similarity_factors = []
            
            # Amount similarity
            hist_amount = float(hist_tx.get("amount", 0))
            if max(current_amount, hist_amount) > 0:
                amount_similarity = min(current_amount, hist_amount) / max(current_amount, hist_amount)
                similarity_factors.append(amount_similarity)
            
            # Country similarity
            if current_country == hist_tx.get("country", ""):
                similarity_factors.append(1.0)
            else:
                similarity_factors.append(0.0)
            
            if similarity_factors:
                similarities.append(np.mean(similarity_factors))
        
        return max(similarities) if similarities else 0.0


class RiskAssessmentAgent(BaseFraudDetectionAgent):
    """Specialized agent for comprehensive risk assessment with uncertainty quantification."""
    
    def __init__(self, agent_id: str, mcp_client: MCPClient):
        super().__init__(agent_id, mcp_client, "risk_assessment")
        
        # Specialized components
        self.probabilistic_model = ProbabilisticRiskModel()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        
        # Risk assessment configuration
        self.risk_weights = {
            "amount_risk": 0.25,
            "velocity_risk": 0.20,
            "geographic_risk": 0.20,
            "account_risk": 0.15,
            "behavioral_risk": 0.10,
            "temporal_risk": 0.10
        }
        
        # Historical data for context
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.global_statistics: Dict[str, float] = {}
    
    async def _load_specialized_knowledge(self):
        """Load risk assessment specific knowledge."""
        
        # Load risk thresholds and weights
        risk_knowledge = {
            "high_risk_countries": ["XX", "YY", "ZZ"],  # Placeholder
            "suspicious_amounts": {
                "small_test_amounts": (0.01, 5.0),
                "large_amounts": (5000, float('inf')),
                "round_amounts": [100, 200, 500, 1000]
            },
            "velocity_thresholds": {
                "high_velocity_1h": 10,
                "high_velocity_24h": 50,
                "suspicious_burst": 5  # transactions in 5 minutes
            },
            "account_risk_factors": {
                "new_account_days": 30,
                "dormant_reactivation_days": 180,
                "first_transaction_risk": 0.3
            }
        }
        
        self.knowledge_base.add_pattern("risk_thresholds", risk_knowledge)
        
        # Initialize global statistics (would be loaded from database)
        self.global_statistics = {
            "avg_transaction_amount": 150.0,
            "fraud_rate_by_country": {"US": 0.015, "GB": 0.018, "FR": 0.020},
            "fraud_rate_by_hour": {h: 0.02 + 0.01 * (abs(h - 12) / 12) for h in range(24)}
        }
        
        logger.info(f"Risk Assessment Agent {self.agent_id} loaded specialized knowledge")
    
    async def analyze_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment with uncertainty quantification."""
        
        start_time = datetime.now()
        transaction_id = transaction.get("transaction_id", "unknown")
        user_id = transaction.get("user_id", "unknown")
        
        try:
            # Get or create user profile
            user_profile = self._get_user_profile(user_id, transaction)
            
            # Step 1: Calculate individual risk components
            risk_components = await self._calculate_risk_components(transaction, user_profile)
            
            # Step 2: Apply probabilistic model
            evidence = self._extract_evidence(transaction, risk_components)
            probability_results = self.probabilistic_model.calculate_posterior_probability(evidence)
            
            # Step 3: Quantify uncertainty
            uncertainty_metrics = self.uncertainty_quantifier.quantify_uncertainty(
                transaction, risk_components, self._get_historical_data(user_id)
            )
            
            # Step 4: Calculate final risk score
            risk_score = self._calculate_final_risk_score(risk_components, probability_results)
            
            # Step 5: Determine confidence
            confidence = self._calculate_assessment_confidence(probability_results, uncertainty_metrics)
            
            # Step 6: Generate risk factors and explanations
            risk_factors = self._generate_detailed_risk_factors(risk_components, evidence)
            protective_factors = self._generate_protective_factors(risk_components, user_profile)
            
            # Step 7: Update user profile
            self._update_user_profile(user_id, transaction, risk_score)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            analysis_results = {
                "agent_type": "risk_assessment",
                "risk_score": risk_score,
                "confidence": confidence,
                "risk_factors": risk_factors,
                "protective_factors": protective_factors,
                "risk_components": risk_components,
                "probability_assessment": probability_results,
                "uncertainty_metrics": uncertainty_metrics,
                "processing_time_ms": processing_time,
                "user_profile_available": user_id in self.user_profiles,
                "evidence_strength": len([e for e in evidence.values() if e])
            }
            
            logger.debug(
                f"Risk assessment completed for {transaction_id}: "
                f"risk={risk_score:.3f}, confidence={confidence:.3f}, "
                f"fraud_prob={probability_results['fraud_probability']:.3f}"
            )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Risk assessment analysis error: {e}")
            return {
                "agent_type": "risk_assessment",
                "risk_score": 0.5,
                "confidence": 0.1,
                "risk_factors": [f"Risk assessment error: {str(e)}"],
                "protective_factors": [],
                "error": str(e)
            }
    
    async def _calculate_risk_components(self, transaction: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, float]:
        """Calculate individual risk components."""
        
        components = {}
        
        # Amount risk
        components["amount_risk"] = self._assess_amount_risk(transaction, user_profile)
        
        # Velocity risk
        components["velocity_risk"] = self._assess_velocity_risk(transaction, user_profile)
        
        # Geographic risk
        components["geographic_risk"] = self._assess_geographic_risk(transaction, user_profile)
        
        # Account risk
        components["account_risk"] = self._assess_account_risk(transaction, user_profile)
        
        # Behavioral risk
        components["behavioral_risk"] = self._assess_behavioral_risk(transaction, user_profile)
        
        # Temporal risk
        components["temporal_risk"] = self._assess_temporal_risk(transaction, user_profile)
        
        return components
    
    def _assess_amount_risk(self, transaction: Dict[str, Any], user_profile: Dict[str, Any]) -> float:
        """Assess risk based on transaction amount."""
        
        amount = float(transaction.get("amount", 0))
        user_avg_amount = user_profile.get("avg_amount", self.global_statistics["avg_transaction_amount"])
        
        risk_factors = []
        
        # Very small amounts (card testing)
        if amount < 5:
            risk_factors.append(0.6)
        
        # Very large amounts
        if amount > 5000:
            risk_factors.append(0.4)
        elif amount > 1000:
            risk_factors.append(0.2)
        
        # Deviation from user average
        if user_avg_amount > 0:
            ratio = amount / user_avg_amount
            if ratio > 10:  # More than 10x average
                risk_factors.append(0.8)
            elif ratio > 5:  # More than 5x average
                risk_factors.append(0.4)
            elif ratio < 0.1:  # Less than 10% of average
                risk_factors.append(0.3)
        
        # Round amounts
        if amount == round(amount) and amount >= 100:
            risk_factors.append(0.2)
        
        return min(1.0, max(risk_factors) if risk_factors else 0.1)
    
    def _assess_velocity_risk(self, transaction: Dict[str, Any], user_profile: Dict[str, Any]) -> float:
        """Assess risk based on transaction velocity."""
        
        velocity_1h = transaction.get("velocity_1h", 0)
        velocity_24h = transaction.get("velocity_24h", 0)
        user_avg_daily = user_profile.get("avg_daily_transactions", 5)
        
        risk_factors = []
        
        # High absolute velocity
        if velocity_1h > 10:
            risk_factors.append(0.9)
        elif velocity_1h > 5:
            risk_factors.append(0.6)
        
        if velocity_24h > 50:
            risk_factors.append(0.8)
        elif velocity_24h > 20:
            risk_factors.append(0.4)
        
        # Velocity relative to user pattern
        if velocity_24h > user_avg_daily * 5:
            risk_factors.append(0.7)
        elif velocity_24h > user_avg_daily * 3:
            risk_factors.append(0.4)
        
        return min(1.0, max(risk_factors) if risk_factors else 0.1)
    
    def _assess_geographic_risk(self, transaction: Dict[str, Any], user_profile: Dict[str, Any]) -> float:
        """Assess risk based on geographic factors."""
        
        country = transaction.get("country", "")
        user_countries = user_profile.get("countries", [country])
        
        risk_factors = []
        
        # High-risk country
        high_risk_countries = self.knowledge_base.patterns.get("risk_thresholds", {}).get("data", {}).get("high_risk_countries", [])
        if country in high_risk_countries:
            risk_factors.append(0.6)
        
        # New country for user
        if country not in user_countries:
            risk_factors.append(0.3)
        
        # Country-specific fraud rate
        country_fraud_rate = self.global_statistics.get("fraud_rate_by_country", {}).get(country, 0.02)
        normalized_country_risk = min(1.0, country_fraud_rate * 25)  # Scale to 0-1
        risk_factors.append(normalized_country_risk)
        
        return min(1.0, max(risk_factors) if risk_factors else 0.1)
    
    def _assess_account_risk(self, transaction: Dict[str, Any], user_profile: Dict[str, Any]) -> float:
        """Assess risk based on account characteristics."""
        
        account_age_days = transaction.get("account_age_days", 365)
        is_first_transaction = transaction.get("is_first_transaction", False)
        
        risk_factors = []
        
        # New account
        if account_age_days < 1:
            risk_factors.append(0.8)
        elif account_age_days < 7:
            risk_factors.append(0.5)
        elif account_age_days < 30:
            risk_factors.append(0.3)
        
        # First transaction
        if is_first_transaction:
            risk_factors.append(0.4)
        
        # Account dormancy (would need historical data)
        last_transaction_days = user_profile.get("days_since_last_transaction", 1)
        if last_transaction_days > 180:
            risk_factors.append(0.3)  # Dormant account reactivation
        
        return min(1.0, max(risk_factors) if risk_factors else 0.1)
    
    def _assess_behavioral_risk(self, transaction: Dict[str, Any], user_profile: Dict[str, Any]) -> float:
        """Assess risk based on behavioral patterns."""
        
        device_id = transaction.get("device_id", "")
        user_devices = user_profile.get("devices", [device_id])
        session_duration = transaction.get("session_duration", 300)
        
        risk_factors = []
        
        # New device
        if device_id not in user_devices:
            risk_factors.append(0.4)
        
        # Very short session
        if session_duration < 30:
            risk_factors.append(0.3)
        
        # Unusual browsing pattern
        pages_visited = transaction.get("pages_visited", 3)
        if pages_visited < 2:
            risk_factors.append(0.2)
        
        return min(1.0, max(risk_factors) if risk_factors else 0.1)
    
    def _assess_temporal_risk(self, transaction: Dict[str, Any], user_profile: Dict[str, Any]) -> float:
        """Assess risk based on temporal patterns."""
        
        try:
            timestamp = datetime.fromisoformat(transaction["timestamp"])
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            
            risk_factors = []
            
            # Unusual hours (2 AM - 5 AM)
            if 2 <= hour <= 5:
                risk_factors.append(0.4)
            
            # Hour-specific fraud rate
            hour_fraud_rate = self.global_statistics.get("fraud_rate_by_hour", {}).get(hour, 0.02)
            normalized_hour_risk = min(1.0, hour_fraud_rate * 25)
            risk_factors.append(normalized_hour_risk)
            
            # Weekend vs weekday patterns (simplified)
            if day_of_week >= 5:  # Weekend
                risk_factors.append(0.1)
            
            return min(1.0, max(risk_factors) if risk_factors else 0.1)
            
        except:
            return 0.2  # Default risk if timestamp parsing fails
    
    def _extract_evidence(self, transaction: Dict[str, Any], risk_components: Dict[str, float]) -> Dict[str, bool]:
        """Extract boolean evidence for probabilistic model."""
        
        evidence = {
            "high_amount": risk_components.get("amount_risk", 0) > 0.5,
            "new_device": "new/unknown device" in str(transaction.get("risk_factors", [])),
            "unusual_location": risk_components.get("geographic_risk", 0) > 0.4,
            "high_velocity": risk_components.get("velocity_risk", 0) > 0.5,
            "new_account": transaction.get("account_age_days", 365) < 30
        }
        
        return evidence
    
    def _calculate_final_risk_score(self, risk_components: Dict[str, float], probability_results: Dict[str, float]) -> float:
        """Calculate final weighted risk score."""
        
        # Weighted average of risk components
        component_score = sum(
            risk_components[component] * self.risk_weights[component]
            for component in risk_components
            if component in self.risk_weights
        )
        
        # Probabilistic model score
        prob_score = probability_results["fraud_probability"]
        
        # Combine scores (70% components, 30% probabilistic)
        final_score = 0.7 * component_score + 0.3 * prob_score
        
        return min(1.0, max(0.0, final_score))
    
    def _calculate_assessment_confidence(self, probability_results: Dict[str, float], uncertainty_metrics: Dict[str, float]) -> float:
        """Calculate confidence in the risk assessment."""
        
        # Base confidence from probabilistic model
        prob_confidence = probability_results.get("confidence", 0.5)
        
        # Reduce confidence based on uncertainty
        total_uncertainty = uncertainty_metrics.get("total_uncertainty", 0.5)
        uncertainty_penalty = min(0.4, total_uncertainty)  # Max 40% reduction
        
        final_confidence = max(0.1, prob_confidence - uncertainty_penalty)
        
        return final_confidence
    
    def _generate_detailed_risk_factors(self, risk_components: Dict[str, float], evidence: Dict[str, bool]) -> List[str]:
        """Generate detailed risk factor explanations."""
        
        risk_factors = []
        
        # Component-based factors
        for component, score in risk_components.items():
            if score > 0.5:
                component_name = component.replace("_", " ").title()
                risk_factors.append(f"{component_name}: {score:.2f} risk score")
        
        # Evidence-based factors
        for factor, is_present in evidence.items():
            if is_present:
                factor_name = factor.replace("_", " ").title()
                risk_factors.append(f"{factor_name} detected")
        
        return risk_factors[:5]  # Limit to top 5
    
    def _generate_protective_factors(self, risk_components: Dict[str, float], user_profile: Dict[str, Any]) -> List[str]:
        """Generate protective factor explanations."""
        
        protective_factors = []
        
        # Low risk components
        for component, score in risk_components.items():
            if score < 0.3:
                component_name = component.replace("_", " ").title()
                protective_factors.append(f"Low {component_name.lower()}")
        
        # User profile factors
        if user_profile.get("transaction_count", 0) > 100:
            protective_factors.append("Established transaction history")
        
        if user_profile.get("avg_amount", 0) > 0:
            protective_factors.append("Consistent spending patterns")
        
        return protective_factors[:3]  # Limit to top 3
    
    def _get_user_profile(self, user_id: str, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Get or create user profile."""
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "user_id": user_id,
                "transaction_count": 0,
                "avg_amount": 0.0,
                "countries": [],
                "devices": [],
                "avg_daily_transactions": 1,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "days_since_last_transaction": 0
            }
        
        return self.user_profiles[user_id]
    
    def _update_user_profile(self, user_id: str, transaction: Dict[str, Any], risk_score: float):
        """Update user profile with new transaction."""
        
        profile = self.user_profiles[user_id]
        
        # Update transaction count and average amount
        current_count = profile["transaction_count"]
        current_avg = profile["avg_amount"]
        new_amount = float(transaction.get("amount", 0))
        
        profile["transaction_count"] = current_count + 1
        profile["avg_amount"] = (current_avg * current_count + new_amount) / (current_count + 1)
        
        # Update countries and devices
        country = transaction.get("country", "")
        if country and country not in profile["countries"]:
            profile["countries"].append(country)
        
        device_id = transaction.get("device_id", "")
        if device_id and device_id not in profile["devices"]:
            profile["devices"].append(device_id)
        
        # Update timestamps
        profile["last_seen"] = datetime.now().isoformat()
    
    def _get_historical_data(self, user_id: str) -> List[Dict[str, Any]]:
        """Get historical transaction data for user (placeholder)."""
        # In real implementation, this would query a database
        return []