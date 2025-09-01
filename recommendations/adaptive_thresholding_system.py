"""
Real-Time Adaptive Thresholding System for MCP Fraud Detection

This module implements dynamic threshold adjustment based on real-time
performance metrics and business constraints.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class ThresholdMetrics:
    timestamp: datetime
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    business_cost: float

@dataclass
class BusinessConstraints:
    max_false_positive_rate: float = 0.02  # Max 2% FP rate
    min_fraud_detection_rate: float = 0.95  # Min 95% fraud detection
    false_positive_cost: float = 25.0  # Cost per declined legitimate transaction
    false_negative_cost: float = 500.0  # Cost per approved fraudulent transaction
    processing_cost_per_transaction: float = 0.01
    revenue_per_transaction: float = 10.0

class AdaptiveThresholdingSystem:
    """Real-time adaptive thresholding system for fraud detection."""
    
    def __init__(
        self,
        initial_threshold: float = 0.5,
        adaptation_window: int = 1000,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
        business_constraints: BusinessConstraints = None
    ):
        self.current_threshold = initial_threshold
        self.adaptation_window = adaptation_window
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.business_constraints = business_constraints or BusinessConstraints()
        
        # Performance tracking
        self.recent_decisions = deque(maxlen=adaptation_window)
        self.threshold_history = deque(maxlen=100)
        self.performance_metrics = {}
        
        # Adaptation parameters
        self.adaptation_rate = 0.1
        self.stability_threshold = 0.95  # Confidence level for threshold changes
        self.min_samples_for_adaptation = 50
        
        # Context-aware thresholds
        self.context_thresholds = {
            'high_risk_user': 0.3,
            'new_user': 0.4,
            'high_velocity': 0.3,
            'large_amount': 0.4,
            'foreign_country': 0.4,
            'new_device': 0.45,
            'business_hours': 0.5,
            'weekend': 0.45,
            'holiday': 0.4
        }
    
    async def get_adaptive_threshold(
        self,
        transaction: Dict[str, Any],
        base_risk_score: float,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Get adaptive threshold based on transaction context and recent performance."""
        
        # Start with current global threshold
        adaptive_threshold = self.current_threshold
        
        # Apply context-specific adjustments
        context_adjustments = self._calculate_context_adjustments(transaction, context)
        
        # Apply performance-based adjustments
        performance_adjustment = await self._calculate_performance_adjustment()
        
        # Apply real-time load adjustments
        load_adjustment = self._calculate_load_adjustment()
        
        # Combine adjustments
        total_adjustment = (
            context_adjustments * 0.4 +
            performance_adjustment * 0.4 +
            load_adjustment * 0.2
        )
        
        adaptive_threshold = np.clip(
            self.current_threshold + total_adjustment,
            self.min_threshold,
            self.max_threshold
        )
        
        # Calculate confidence in the threshold
        threshold_confidence = self._calculate_threshold_confidence(
            transaction, adaptive_threshold, base_risk_score
        )
        
        return {
            'threshold': adaptive_threshold,
            'base_threshold': self.current_threshold,
            'context_adjustment': context_adjustments,
            'performance_adjustment': performance_adjustment,
            'load_adjustment': load_adjustment,
            'confidence': threshold_confidence,
            'decision': 'decline' if base_risk_score > adaptive_threshold else 'approve',
            'decision_margin': abs(base_risk_score - adaptive_threshold)
        }
    
    def _calculate_context_adjustments(
        self,
        transaction: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> float:
        """Calculate context-specific threshold adjustments."""
        
        adjustments = []
        
        # User context adjustments
        if context and context.get('user_risk_level') == 'high':
            adjustments.append(-0.2)  # Lower threshold for high-risk users
        
        if transaction.get('is_first_transaction', False):
            adjustments.append(-0.1)  # Lower threshold for new users
        
        # Transaction context adjustments
        amount = transaction.get('amount', 0)
        if amount > 5000:
            adjustments.append(-0.15)  # Lower threshold for large amounts
        elif amount < 5:
            adjustments.append(-0.1)  # Lower threshold for micro transactions
        
        # Velocity adjustments
        velocity_1h = transaction.get('velocity_1h', 0)
        if velocity_1h > 10:
            adjustments.append(-0.2)  # Much lower threshold for high velocity
        elif velocity_1h > 5:
            adjustments.append(-0.1)
        
        # Geographic adjustments
        country = transaction.get('country', 'US')
        if country in ['XX', 'YY', 'ZZ']:  # High-risk countries
            adjustments.append(-0.15)
        
        # Device adjustments
        device_data = transaction.get('device_data', {})
        if device_data.get('is_new_device', False):
            adjustments.append(-0.1)
        
        # Temporal adjustments
        dt = datetime.fromisoformat(transaction['timestamp'])
        if dt.hour < 6 or dt.hour > 22:  # Late night/early morning
            adjustments.append(-0.05)
        
        if dt.weekday() >= 5:  # Weekend
            adjustments.append(-0.05)
        
        # Return average adjustment
        return np.mean(adjustments) if adjustments else 0.0
    
    async def _calculate_performance_adjustment(self) -> float:
        """Calculate performance-based threshold adjustments."""
        
        if len(self.recent_decisions) < self.min_samples_for_adaptation:
            return 0.0
        
        # Calculate recent performance metrics
        recent_metrics = self._calculate_recent_metrics()
        
        adjustments = []
        
        # False positive rate adjustment
        if recent_metrics['false_positive_rate'] > self.business_constraints.max_false_positive_rate:
            # Too many false positives, increase threshold
            fp_excess = recent_metrics['false_positive_rate'] - self.business_constraints.max_false_positive_rate
            adjustments.append(fp_excess * 2.0)  # Aggressive adjustment
        
        # False negative rate adjustment
        fraud_detection_rate = recent_metrics.get('recall', 0.0)
        if fraud_detection_rate < self.business_constraints.min_fraud_detection_rate:
            # Missing too much fraud, decrease threshold
            fn_deficit = self.business_constraints.min_fraud_detection_rate - fraud_detection_rate
            adjustments.append(-fn_deficit * 1.5)  # Aggressive adjustment
        
        # Business cost optimization
        current_cost = self._calculate_business_cost(recent_metrics)
        target_cost = self._calculate_optimal_cost()
        
        if current_cost > target_cost * 1.1:  # 10% tolerance
            cost_ratio = current_cost / target_cost
            if recent_metrics['false_positive_rate'] > recent_metrics.get('false_negative_rate', 0):
                adjustments.append(0.1 * np.log(cost_ratio))  # Increase threshold
            else:
                adjustments.append(-0.1 * np.log(cost_ratio))  # Decrease threshold
        
        return np.mean(adjustments) if adjustments else 0.0
    
    def _calculate_load_adjustment(self) -> float:
        """Calculate load-based threshold adjustments."""
        
        # Simulate system load (in production, this would be real metrics)
        current_time = datetime.now()
        hour = current_time.hour
        
        # Business hours typically have higher load
        if 9 <= hour <= 17:
            base_load = 0.7
        elif 18 <= hour <= 22:
            base_load = 0.5
        else:
            base_load = 0.2
        
        # Add some randomness to simulate varying load
        actual_load = base_load + np.random.normal(0, 0.1)
        actual_load = np.clip(actual_load, 0.1, 1.0)
        
        # Under high load, slightly increase threshold to reduce processing
        if actual_load > 0.8:
            return 0.05 * (actual_load - 0.8) / 0.2
        
        return 0.0
    
    def _calculate_recent_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics from recent decisions."""
        
        if not self.recent_decisions:
            return {}
        
        # Convert to arrays for calculation
        decisions_data = list(self.recent_decisions)
        
        y_true = [d['actual_label'] for d in decisions_data if 'actual_label' in d]
        y_pred = [d['predicted_label'] for d in decisions_data if 'predicted_label' in d]
        y_scores = [d['risk_score'] for d in decisions_data if 'risk_score' in d]
        
        if not y_true or not y_pred:
            return {}
        
        # Calculate confusion matrix
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        
        # Calculate metrics
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-8)
        
        false_positive_rate = fp / max(fp + tn, 1)
        false_negative_rate = fn / max(fn + tp, 1)
        
        accuracy = (tp + tn) / max(len(y_true), 1)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'accuracy': accuracy,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    
    def _calculate_business_cost(self, metrics: Dict[str, float]) -> float:
        """Calculate business cost based on current performance."""
        
        fp_cost = metrics.get('false_positives', 0) * self.business_constraints.false_positive_cost
        fn_cost = metrics.get('false_negatives', 0) * self.business_constraints.false_negative_cost
        processing_cost = len(self.recent_decisions) * self.business_constraints.processing_cost_per_transaction
        
        return fp_cost + fn_cost + processing_cost
    
    def _calculate_optimal_cost(self) -> float:
        """Calculate theoretical optimal cost."""
        
        # This is a simplified calculation
        # In practice, this would be based on historical data and business models
        total_transactions = len(self.recent_decisions)
        expected_fraud_rate = 0.01  # 1% fraud rate
        
        optimal_fp = total_transactions * 0.005  # 0.5% optimal FP rate
        optimal_fn = total_transactions * expected_fraud_rate * 0.05  # 5% miss rate
        
        return (
            optimal_fp * self.business_constraints.false_positive_cost +
            optimal_fn * self.business_constraints.false_negative_cost +
            total_transactions * self.business_constraints.processing_cost_per_transaction
        )
    
    def _calculate_threshold_confidence(
        self,
        transaction: Dict[str, Any],
        threshold: float,
        risk_score: float
    ) -> float:
        """Calculate confidence in the threshold decision."""
        
        # Base confidence on margin between risk score and threshold
        margin = abs(risk_score - threshold)
        margin_confidence = min(margin * 2, 1.0)  # Normalize to 0-1
        
        # Adjust based on recent performance stability
        performance_stability = self._calculate_performance_stability()
        
        # Adjust based on context familiarity
        context_familiarity = self._calculate_context_familiarity(transaction)
        
        # Combined confidence
        confidence = (
            margin_confidence * 0.5 +
            performance_stability * 0.3 +
            context_familiarity * 0.2
        )
        
        return np.clip(confidence, 0.1, 1.0)
    
    def _calculate_performance_stability(self) -> float:
        """Calculate stability of recent performance."""
        
        if len(self.threshold_history) < 10:
            return 0.5
        
        recent_thresholds = list(self.threshold_history)[-10:]
        threshold_variance = np.var(recent_thresholds)
        
        # Lower variance = higher stability
        stability = max(0.1, 1.0 - (threshold_variance / 0.1))
        return min(stability, 1.0)
    
    def _calculate_context_familiarity(self, transaction: Dict[str, Any]) -> float:
        """Calculate how familiar the system is with this transaction context."""
        
        # Check similarity to recent transactions
        if not self.recent_decisions:
            return 0.5
        
        similar_transactions = 0
        total_recent = min(len(self.recent_decisions), 100)
        
        for recent_decision in list(self.recent_decisions)[-total_recent:]:
            similarity_score = self._calculate_transaction_similarity(
                transaction, recent_decision.get('transaction', {})
            )
            if similarity_score > 0.7:
                similar_transactions += 1
        
        familiarity = similar_transactions / total_recent
        return np.clip(familiarity, 0.1, 1.0)
    
    def _calculate_transaction_similarity(
        self,
        tx1: Dict[str, Any],
        tx2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two transactions."""
        
        if not tx2:
            return 0.0
        
        similarities = []
        
        # Amount similarity
        amount1 = tx1.get('amount', 0)
        amount2 = tx2.get('amount', 0)
        if amount1 > 0 and amount2 > 0:
            amount_sim = 1 - abs(np.log(amount1) - np.log(amount2)) / 10
            similarities.append(max(0, amount_sim))
        
        # Country similarity
        if tx1.get('country') == tx2.get('country'):
            similarities.append(1.0)
        else:
            similarities.append(0.0)
        
        # Payment method similarity
        if tx1.get('payment_method') == tx2.get('payment_method'):
            similarities.append(1.0)
        else:
            similarities.append(0.0)
        
        # Velocity similarity
        vel1 = tx1.get('velocity_24h', 0)
        vel2 = tx2.get('velocity_24h', 0)
        if vel1 > 0 or vel2 > 0:
            vel_sim = 1 - abs(vel1 - vel2) / max(vel1 + vel2, 1)
            similarities.append(vel_sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    async def update_threshold_performance(
        self,
        transaction: Dict[str, Any],
        risk_score: float,
        predicted_label: int,
        actual_label: int = None,
        threshold_used: float = None
    ):
        """Update threshold performance with new decision result."""
        
        decision_record = {
            'timestamp': datetime.now(),
            'transaction': transaction,
            'risk_score': risk_score,
            'predicted_label': predicted_label,
            'threshold_used': threshold_used or self.current_threshold,
            'actual_label': actual_label
        }
        
        self.recent_decisions.append(decision_record)
        
        # Update global threshold if we have enough data
        if len(self.recent_decisions) >= self.min_samples_for_adaptation:
            await self._update_global_threshold()
    
    async def _update_global_threshold(self):
        """Update the global threshold based on recent performance."""
        
        if len(self.recent_decisions) < self.min_samples_for_adaptation:
            return
        
        # Calculate current performance
        current_metrics = self._calculate_recent_metrics()
        
        if not current_metrics:
            return
        
        # Determine if threshold adjustment is needed
        adjustment_needed = False
        proposed_adjustment = 0.0
        
        # Check business constraints
        if current_metrics['false_positive_rate'] > self.business_constraints.max_false_positive_rate * 1.1:
            # Significant FP rate violation
            adjustment_needed = True
            proposed_adjustment += 0.05  # Increase threshold
        
        if current_metrics['recall'] < self.business_constraints.min_fraud_detection_rate * 0.9:
            # Significant fraud detection rate violation
            adjustment_needed = True
            proposed_adjustment -= 0.05  # Decrease threshold
        
        # Apply adjustment if needed
        if adjustment_needed:
            old_threshold = self.current_threshold
            self.current_threshold = np.clip(
                self.current_threshold + proposed_adjustment,
                self.min_threshold,
                self.max_threshold
            )
            
            # Record threshold change
            self.threshold_history.append(self.current_threshold)
            
            # Log threshold change
            logger.info(
                f"Threshold updated: {old_threshold:.3f} -> {self.current_threshold:.3f} "
                f"(FP rate: {current_metrics['false_positive_rate']:.3f}, "
                f"Recall: {current_metrics['recall']:.3f})"
            )
    
    def get_threshold_analytics(self) -> Dict[str, Any]:
        """Get comprehensive threshold analytics."""
        
        recent_metrics = self._calculate_recent_metrics()
        
        return {
            'current_threshold': self.current_threshold,
            'threshold_history': list(self.threshold_history),
            'recent_performance': recent_metrics,
            'business_constraints': asdict(self.business_constraints),
            'adaptation_stats': {
                'recent_decisions_count': len(self.recent_decisions),
                'adaptation_window': self.adaptation_window,
                'min_samples_for_adaptation': self.min_samples_for_adaptation,
                'performance_stability': self._calculate_performance_stability()
            },
            'context_thresholds': self.context_thresholds
        }
    
    def optimize_threshold_for_business_objective(
        self,
        objective: str = 'minimize_cost',
        constraint_weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Optimize threshold for specific business objective."""
        
        if not constraint_weights:
            constraint_weights = {
                'cost_weight': 0.4,
                'accuracy_weight': 0.3,
                'customer_experience_weight': 0.3
            }
        
        # Test different thresholds
        test_thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = self.current_threshold
        best_score = float('-inf')
        
        threshold_analysis = []
        
        for threshold in test_thresholds:
            # Simulate performance at this threshold
            simulated_metrics = self._simulate_threshold_performance(threshold)
            
            if objective == 'minimize_cost':
                score = -self._calculate_business_cost(simulated_metrics)
            elif objective == 'maximize_f1':
                score = simulated_metrics.get('f1_score', 0)
            elif objective == 'balanced':
                cost_score = -self._calculate_business_cost(simulated_metrics) / 1000
                f1_score = simulated_metrics.get('f1_score', 0)
                score = cost_score * 0.5 + f1_score * 0.5
            else:
                score = simulated_metrics.get('f1_score', 0)
            
            threshold_analysis.append({
                'threshold': threshold,
                'score': score,
                'metrics': simulated_metrics
            })
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return {
            'recommended_threshold': best_threshold,
            'current_threshold': self.current_threshold,
            'improvement_score': best_score,
            'objective': objective,
            'threshold_analysis': threshold_analysis
        }
    
    def _simulate_threshold_performance(self, threshold: float) -> Dict[str, float]:
        """Simulate performance metrics at a given threshold."""
        
        if not self.recent_decisions:
            return {}
        
        # Apply threshold to recent decisions
        tp = fp = tn = fn = 0
        
        for decision in self.recent_decisions:
            risk_score = decision.get('risk_score', 0.5)
            actual_label = decision.get('actual_label')
            
            if actual_label is None:
                continue
            
            predicted_label = 1 if risk_score > threshold else 0
            
            if actual_label == 1 and predicted_label == 1:
                tp += 1
            elif actual_label == 0 and predicted_label == 1:
                fp += 1
            elif actual_label == 0 and predicted_label == 0:
                tn += 1
            elif actual_label == 1 and predicted_label == 0:
                fn += 1
        
        # Calculate metrics
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-8)
        false_positive_rate = fp / max(fp + tn, 1)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'false_positive_rate': false_positive_rate,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
