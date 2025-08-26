import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
import hashlib
from dataclasses import dataclass

from .base_agent import BaseFraudDetectionAgent
from core.mcp_client import MCPClient

logger = logging.getLogger(__name__)


@dataclass
class FraudPattern:
    pattern_id: str
    pattern_type: str
    features: Dict[str, Any]
    confidence: float
    support_count: int
    accuracy_rate: float
    discovered_at: datetime
    last_seen: datetime


class PatternMiner:
    """Advanced pattern mining for fraud detection."""
    
    def __init__(self):
        self.transaction_sequences: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.pattern_cache: Dict[str, FraudPattern] = {}
        self.min_support = 3  # Minimum occurrences to consider a pattern
        self.confidence_threshold = 0.6
    
    async def mine_sequential_patterns(self, transactions: List[Dict[str, Any]]) -> List[FraudPattern]:
        """Mine sequential patterns from transaction data."""
        
        patterns = []
        
        # Group transactions by user
        user_transactions = defaultdict(list)
        for tx in transactions:
            user_id = tx.get("user_id", "unknown")
            user_transactions[user_id].append(tx)
        
        # Sort transactions by timestamp for each user
        for user_id, user_txs in user_transactions.items():
            user_txs.sort(key=lambda x: x.get("timestamp", ""))
            self.transaction_sequences[user_id] = user_txs
        
        # Mine patterns
        velocity_patterns = await self._mine_velocity_patterns()
        amount_patterns = await self._mine_amount_patterns()
        geographic_patterns = await self._mine_geographic_patterns()
        temporal_patterns = await self._mine_temporal_patterns()
        
        patterns.extend(velocity_patterns)
        patterns.extend(amount_patterns)
        patterns.extend(geographic_patterns)
        patterns.extend(temporal_patterns)
        
        return patterns
    
    async def _mine_velocity_patterns(self) -> List[FraudPattern]:
        """Mine velocity-based fraud patterns."""
        patterns = []
        
        velocity_sequences = []
        for user_id, transactions in self.transaction_sequences.items():
            if len(transactions) < 2:
                continue
            
            # Calculate transaction intervals
            intervals = []
            for i in range(1, len(transactions)):
                prev_time = datetime.fromisoformat(transactions[i-1]["timestamp"])
                curr_time = datetime.fromisoformat(transactions[i]["timestamp"])
                interval_minutes = (curr_time - prev_time).total_seconds() / 60
                intervals.append(interval_minutes)
            
            velocity_sequences.append({
                "user_id": user_id,
                "intervals": intervals,
                "transaction_count": len(transactions)
            })
        
        # Find rapid-fire patterns (multiple transactions in short time)
        rapid_fire_threshold = 5  # minutes
        rapid_sequences = [
            seq for seq in velocity_sequences 
            if any(interval < rapid_fire_threshold for interval in seq["intervals"])
        ]
        
        if len(rapid_sequences) >= self.min_support:
            pattern = FraudPattern(
                pattern_id=f"velocity_rapid_fire_{hash(str(rapid_fire_threshold)) % 10000}",
                pattern_type="velocity_rapid_fire",
                features={
                    "max_interval_minutes": rapid_fire_threshold,
                    "typical_transaction_count": np.mean([seq["transaction_count"] for seq in rapid_sequences]),
                    "affected_users": len(rapid_sequences)
                },
                confidence=0.8,
                support_count=len(rapid_sequences),
                accuracy_rate=0.0,  # Will be updated with feedback
                discovered_at=datetime.now(),
                last_seen=datetime.now()
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _mine_amount_patterns(self) -> List[FraudPattern]:
        """Mine amount-based fraud patterns."""
        patterns = []
        
        # Collect all amounts
        amounts = []
        round_amounts = []
        
        for transactions in self.transaction_sequences.values():
            for tx in transactions:
                amount = float(tx.get("amount", 0))
                amounts.append(amount)
                
                if amount == round(amount) and amount >= 100:
                    round_amounts.append(amount)
        
        # Round amount pattern
        if len(round_amounts) >= self.min_support:
            common_round_amounts = Counter(round_amounts).most_common(5)
            
            pattern = FraudPattern(
                pattern_id=f"amount_round_{hash(str(sorted(round_amounts))) % 10000}",
                pattern_type="round_amount_testing",
                features={
                    "common_amounts": [amount for amount, count in common_round_amounts],
                    "frequency": len(round_amounts) / len(amounts) if amounts else 0,
                    "typical_range": (min(round_amounts), max(round_amounts))
                },
                confidence=0.6,
                support_count=len(round_amounts),
                accuracy_rate=0.0,
                discovered_at=datetime.now(),
                last_seen=datetime.now()
            )
            patterns.append(pattern)
        
        # Escalating amount pattern
        escalating_sequences = []
        for transactions in self.transaction_sequences.values():
            if len(transactions) < 3:
                continue
            
            amounts_seq = [float(tx.get("amount", 0)) for tx in transactions]
            is_escalating = all(amounts_seq[i] <= amounts_seq[i+1] for i in range(len(amounts_seq)-1))
            
            if is_escalating and max(amounts_seq) > min(amounts_seq) * 2:
                escalating_sequences.append(amounts_seq)
        
        if len(escalating_sequences) >= self.min_support:
            pattern = FraudPattern(
                pattern_id=f"amount_escalating_{hash(str(len(escalating_sequences))) % 10000}",
                pattern_type="escalating_amounts",
                features={
                    "average_escalation_factor": np.mean([
                        max(seq) / min(seq) for seq in escalating_sequences
                    ]),
                    "typical_sequence_length": np.mean([len(seq) for seq in escalating_sequences]),
                    "sequences_found": len(escalating_sequences)
                },
                confidence=0.75,
                support_count=len(escalating_sequences),
                accuracy_rate=0.0,
                discovered_at=datetime.now(),
                last_seen=datetime.now()
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _mine_geographic_patterns(self) -> List[FraudPattern]:
        """Mine geographic fraud patterns."""
        patterns = []
        
        # Collect geographic movement patterns
        geographic_sequences = []
        for user_id, transactions in self.transaction_sequences.items():
            if len(transactions) < 2:
                continue
            
            countries = [tx.get("country", "") for tx in transactions]
            unique_countries = list(set(countries))
            
            if len(unique_countries) > 1:
                geographic_sequences.append({
                    "user_id": user_id,
                    "countries": countries,
                    "unique_countries": unique_countries,
                    "country_changes": len(unique_countries)
                })
        
        # Geographic velocity (impossible travel)
        high_velocity_travel = [
            seq for seq in geographic_sequences 
            if seq["country_changes"] > 2 and len(seq["countries"]) <= 5
        ]
        
        if len(high_velocity_travel) >= self.min_support:
            pattern = FraudPattern(
                pattern_id=f"geo_impossible_travel_{hash(str(len(high_velocity_travel))) % 10000}",
                pattern_type="impossible_geographic_velocity",
                features={
                    "average_country_changes": np.mean([seq["country_changes"] for seq in high_velocity_travel]),
                    "common_country_pairs": self._find_common_country_pairs(high_velocity_travel),
                    "affected_users": len(high_velocity_travel)
                },
                confidence=0.85,
                support_count=len(high_velocity_travel),
                accuracy_rate=0.0,
                discovered_at=datetime.now(),
                last_seen=datetime.now()
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _mine_temporal_patterns(self) -> List[FraudPattern]:
        """Mine temporal fraud patterns."""
        patterns = []
        
        # Collect transaction times
        hour_distribution = defaultdict(int)
        unusual_hour_transactions = []
        
        for transactions in self.transaction_sequences.values():
            for tx in transactions:
                try:
                    hour = datetime.fromisoformat(tx["timestamp"]).hour
                    hour_distribution[hour] += 1
                    
                    # Consider 2 AM - 5 AM as unusual
                    if 2 <= hour <= 5:
                        unusual_hour_transactions.append(tx)
                except:
                    continue
        
        # Unusual hours pattern
        if len(unusual_hour_transactions) >= self.min_support:
            pattern = FraudPattern(
                pattern_id=f"temporal_unusual_hours_{hash(str(len(unusual_hour_transactions))) % 10000}",
                pattern_type="unusual_hour_activity",
                features={
                    "suspicious_hours": [2, 3, 4, 5],
                    "transaction_count": len(unusual_hour_transactions),
                    "hour_distribution": dict(hour_distribution)
                },
                confidence=0.7,
                support_count=len(unusual_hour_transactions),
                accuracy_rate=0.0,
                discovered_at=datetime.now(),
                last_seen=datetime.now()
            )
            patterns.append(pattern)
        
        return patterns
    
    def _find_common_country_pairs(self, geographic_sequences: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Find common country transition pairs."""
        pairs = []
        
        for seq in geographic_sequences:
            countries = seq["countries"]
            for i in range(len(countries) - 1):
                if countries[i] != countries[i + 1]:
                    pair = tuple(sorted([countries[i], countries[i + 1]]))
                    pairs.append(pair)
        
        pair_counts = Counter(pairs)
        return pair_counts.most_common(3)


class AnomalyDetector:
    """Statistical anomaly detection for fraud patterns."""
    
    def __init__(self):
        self.statistical_models: Dict[str, Dict[str, float]] = {}
        self.anomaly_threshold = 2.5  # Standard deviations
    
    async def detect_statistical_anomalies(self, transaction: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect statistical anomalies in transaction."""
        
        anomalies = []
        anomaly_scores = {}
        
        if not historical_data:
            return {"anomalies": [], "scores": {}, "baseline_available": False}
        
        # Amount anomaly detection
        amounts = [float(tx.get("amount", 0)) for tx in historical_data if tx.get("amount")]
        if amounts:
            mean_amount = np.mean(amounts)
            std_amount = np.std(amounts)
            current_amount = float(transaction.get("amount", 0))
            
            if std_amount > 0:
                z_score = abs(current_amount - mean_amount) / std_amount
                anomaly_scores["amount_z_score"] = z_score
                
                if z_score > self.anomaly_threshold:
                    anomalies.append(f"Amount anomaly: ${current_amount:.2f} is {z_score:.1f} std devs from mean ${mean_amount:.2f}")
        
        # Velocity anomaly detection
        velocities = [tx.get("velocity_24h", 0) for tx in historical_data]
        current_velocity = transaction.get("velocity_24h", 0)
        
        if velocities and current_velocity:
            mean_velocity = np.mean(velocities)
            std_velocity = np.std(velocities)
            
            if std_velocity > 0:
                velocity_z_score = abs(current_velocity - mean_velocity) / std_velocity
                anomaly_scores["velocity_z_score"] = velocity_z_score
                
                if velocity_z_score > self.anomaly_threshold:
                    anomalies.append(f"Velocity anomaly: {current_velocity} transactions is {velocity_z_score:.1f} std devs from mean {mean_velocity:.1f}")
        
        # Time-of-day anomaly
        hours = []
        for tx in historical_data:
            try:
                hour = datetime.fromisoformat(tx["timestamp"]).hour
                hours.append(hour)
            except:
                continue
        
        if hours:
            try:
                current_hour = datetime.fromisoformat(transaction["timestamp"]).hour
                hour_counts = Counter(hours)
                total_hours = len(hours)
                current_hour_frequency = hour_counts.get(current_hour, 0) / total_hours
                
                anomaly_scores["hour_frequency"] = current_hour_frequency
                
                if current_hour_frequency < 0.05:  # Less than 5% of historical transactions
                    anomalies.append(f"Unusual time: Hour {current_hour} represents only {current_hour_frequency:.1%} of historical activity")
            except:
                pass
        
        return {
            "anomalies": anomalies,
            "scores": anomaly_scores,
            "anomaly_count": len(anomalies),
            "max_z_score": max(anomaly_scores.values()) if anomaly_scores else 0,
            "baseline_available": True
        }


class PatternRecognitionAgent(BaseFraudDetectionAgent):
    """Specialized agent for discovering and recognizing fraud patterns."""
    
    def __init__(self, agent_id: str, mcp_client: MCPClient):
        super().__init__(agent_id, mcp_client, "pattern_recognition")
        
        # Specialized components
        self.pattern_miner = PatternMiner()
        self.anomaly_detector = AnomalyDetector()
        
        # Pattern library
        self.discovered_patterns: Dict[str, FraudPattern] = {}
        self.pattern_matches: Dict[str, List[str]] = defaultdict(list)  # transaction_id -> pattern_ids
        
        # Historical data for analysis
        self.transaction_history: List[Dict[str, Any]] = []
        self.max_history_size = 10000
    
    async def _load_specialized_knowledge(self):
        """Load pattern recognition specific knowledge."""
        
        # Load known fraud patterns
        known_patterns = {
            "card_testing": {
                "description": "Small amounts to test card validity",
                "features": {
                    "amount_range": (1, 10),
                    "velocity_pattern": "rapid_sequential",
                    "success_rate": "mixed"
                },
                "confidence": 0.8
            },
            "account_takeover": {
                "description": "Sudden change in transaction patterns",
                "features": {
                    "geographic_change": True,
                    "amount_escalation": True,
                    "device_change": True
                },
                "confidence": 0.9
            },
            "synthetic_identity": {
                "description": "Artificially created identity patterns",
                "features": {
                    "new_account": True,
                    "perfect_payment_history": True,
                    "sudden_large_transactions": True
                },
                "confidence": 0.85
            }
        }
        
        for pattern_name, pattern_data in known_patterns.items():
            self.knowledge_base.add_pattern(pattern_name, pattern_data)
        
        logger.info(f"Pattern Recognition Agent {self.agent_id} loaded {len(known_patterns)} known patterns")
    
    async def analyze_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Perform pattern recognition analysis on transaction."""
        
        start_time = datetime.now()
        transaction_id = transaction.get("transaction_id", "unknown")
        
        # Add transaction to history
        self.transaction_history.append(transaction)
        if len(self.transaction_history) > self.max_history_size:
            self.transaction_history = self.transaction_history[-self.max_history_size:]
        
        try:
            # Step 1: Match against known patterns
            pattern_matches = await self._match_known_patterns(transaction)
            
            # Step 2: Detect statistical anomalies
            anomaly_results = await self.anomaly_detector.detect_statistical_anomalies(
                transaction, self.transaction_history[-1000:]  # Last 1000 transactions
            )
            
            # Step 3: Mine new patterns periodically
            if len(self.transaction_history) % 100 == 0:  # Every 100 transactions
                await self._discover_new_patterns()
            
            # Step 4: Calculate overall pattern-based risk score
            risk_score = self._calculate_pattern_risk_score(pattern_matches, anomaly_results)
            
            # Step 5: Determine confidence
            confidence = self._calculate_pattern_confidence(pattern_matches, anomaly_results)
            
            # Step 6: Generate risk factors and insights
            risk_factors = self._generate_risk_factors(pattern_matches, anomaly_results)
            protective_factors = self._generate_protective_factors(pattern_matches, anomaly_results)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            analysis_results = {
                "agent_type": "pattern_recognition",
                "risk_score": risk_score,
                "confidence": confidence,
                "risk_factors": risk_factors,
                "protective_factors": protective_factors,
                "pattern_matches": pattern_matches,
                "anomaly_detection": anomaly_results,
                "processing_time_ms": processing_time,
                "patterns_in_library": len(self.discovered_patterns),
                "transaction_history_size": len(self.transaction_history)
            }
            
            # Store pattern matches for this transaction
            matched_pattern_ids = [match["pattern_id"] for match in pattern_matches]
            self.pattern_matches[transaction_id] = matched_pattern_ids
            
            logger.debug(
                f"Pattern analysis completed for {transaction_id}: "
                f"risk={risk_score:.3f}, patterns_matched={len(pattern_matches)}, "
                f"anomalies={anomaly_results['anomaly_count']}"
            )
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Pattern recognition analysis error: {e}")
            return {
                "agent_type": "pattern_recognition",
                "risk_score": 0.5,
                "confidence": 0.1,
                "risk_factors": [f"Pattern analysis error: {str(e)}"],
                "protective_factors": [],
                "error": str(e)
            }
    
    async def _match_known_patterns(self, transaction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Match transaction against known fraud patterns."""
        
        matches = []
        
        # Get relevant patterns from knowledge base
        relevant_patterns = self.knowledge_base.get_relevant_patterns(transaction)
        
        for pattern_name, pattern_info in relevant_patterns:
            pattern_data = pattern_info["data"]
            relevance = pattern_info["relevance"]
            accuracy = pattern_info["accuracy_rate"]
            
            # Calculate match strength
            match_strength = self._calculate_match_strength(transaction, pattern_data)
            
            if match_strength > 0.3:  # Minimum match threshold
                matches.append({
                    "pattern_id": pattern_name,
                    "pattern_type": pattern_data.get("description", "Unknown"),
                    "match_strength": match_strength,
                    "relevance": relevance,
                    "accuracy": accuracy,
                    "confidence": pattern_info["confidence"],
                    "evidence": self._extract_pattern_evidence(transaction, pattern_data)
                })
        
        # Sort by match strength
        matches.sort(key=lambda x: x["match_strength"] * x["confidence"], reverse=True)
        
        return matches
    
    def _calculate_match_strength(self, transaction: Dict[str, Any], pattern: Dict[str, Any]) -> float:
        """Calculate how strongly a transaction matches a pattern."""
        
        match_factors = []
        
        # Amount-based matching
        if "amount_range" in pattern:
            amount = float(transaction.get("amount", 0))
            min_amount, max_amount = pattern["amount_range"]
            if min_amount <= amount <= max_amount:
                match_factors.append(1.0)
            else:
                # Partial match based on distance
                center = (min_amount + max_amount) / 2
                range_size = max_amount - min_amount
                distance = abs(amount - center)
                normalized_distance = distance / max(range_size, 1)
                match_factors.append(max(0, 1 - normalized_distance))
        
        # Velocity pattern matching
        if "velocity_pattern" in pattern:
            velocity_1h = transaction.get("velocity_1h", 0)
            velocity_24h = transaction.get("velocity_24h", 0)
            
            if pattern["velocity_pattern"] == "rapid_sequential":
                if velocity_1h > 5 or velocity_24h > 20:
                    match_factors.append(0.9)
                else:
                    match_factors.append(0.3)
        
        # Geographic matching
        if "geographic_change" in pattern and pattern["geographic_change"]:
            # This would require historical data - simplified here
            if transaction.get("is_new_location", False):
                match_factors.append(0.8)
            else:
                match_factors.append(0.2)
        
        # Device change matching
        if "device_change" in pattern and pattern["device_change"]:
            if "new/unknown device" in transaction.get("risk_factors", []):
                match_factors.append(0.8)
            else:
                match_factors.append(0.2)
        
        # Account age matching
        if "new_account" in pattern and pattern["new_account"]:
            account_age_days = transaction.get("account_age_days", 365)
            if account_age_days < 30:
                match_factors.append(0.9)
            elif account_age_days < 90:
                match_factors.append(0.5)
            else:
                match_factors.append(0.1)
        
        return np.mean(match_factors) if match_factors else 0.0
    
    def _extract_pattern_evidence(self, transaction: Dict[str, Any], pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Extract evidence of pattern matching."""
        
        evidence = {}
        
        if "amount_range" in pattern:
            amount = float(transaction.get("amount", 0))
            evidence["amount"] = {
                "transaction_amount": amount,
                "pattern_range": pattern["amount_range"],
                "within_range": pattern["amount_range"][0] <= amount <= pattern["amount_range"][1]
            }
        
        if "velocity_pattern" in pattern:
            evidence["velocity"] = {
                "velocity_1h": transaction.get("velocity_1h", 0),
                "velocity_24h": transaction.get("velocity_24h", 0),
                "pattern_type": pattern["velocity_pattern"]
            }
        
        return evidence
    
    async def _discover_new_patterns(self):
        """Periodically mine for new fraud patterns."""
        
        try:
            if len(self.transaction_history) < 50:  # Need minimum data
                return
            
            # Mine patterns from recent transaction history
            new_patterns = await self.pattern_miner.mine_sequential_patterns(
                self.transaction_history[-500:]  # Last 500 transactions
            )
            
            # Add discovered patterns to library
            for pattern in new_patterns:
                if pattern.pattern_id not in self.discovered_patterns:
                    self.discovered_patterns[pattern.pattern_id] = pattern
                    
                    # Also add to knowledge base
                    pattern_data = {
                        "description": f"Auto-discovered {pattern.pattern_type}",
                        "features": pattern.features,
                        "confidence": pattern.confidence,
                        "support_count": pattern.support_count
                    }
                    self.knowledge_base.add_pattern(pattern.pattern_id, pattern_data)
                    
                    logger.info(f"Discovered new pattern: {pattern.pattern_id} ({pattern.pattern_type})")
            
            # Update performance metrics
            self.performance_metrics["patterns_learned"] = len(self.discovered_patterns)
            
        except Exception as e:
            logger.error(f"Pattern discovery error: {e}")
    
    def _calculate_pattern_risk_score(self, pattern_matches: List[Dict[str, Any]], anomaly_results: Dict[str, Any]) -> float:
        """Calculate overall risk score based on pattern matches and anomalies."""
        
        risk_components = []
        
        # Pattern match component
        if pattern_matches:
            # Weight by match strength and pattern confidence
            pattern_scores = [
                match["match_strength"] * match["confidence"] 
                for match in pattern_matches
            ]
            pattern_risk = min(1.0, max(pattern_scores))  # Take highest match
            risk_components.append(pattern_risk)
        else:
            risk_components.append(0.2)  # Base risk when no patterns match
        
        # Anomaly detection component
        anomaly_count = anomaly_results.get("anomaly_count", 0)
        max_z_score = anomaly_results.get("max_z_score", 0)
        
        if anomaly_count > 0:
            # Scale z-score to 0-1 range
            anomaly_risk = min(1.0, max_z_score / 5.0)  # z-score of 5 = max risk
            risk_components.append(anomaly_risk)
        else:
            risk_components.append(0.1)  # Low risk when no anomalies
        
        # Combine components with weights
        pattern_weight = 0.7
        anomaly_weight = 0.3
        
        if len(risk_components) >= 2:
            final_risk = (pattern_weight * risk_components[0] + 
                         anomaly_weight * risk_components[1])
        else:
            final_risk = np.mean(risk_components)
        
        return min(1.0, max(0.0, final_risk))
    
    def _calculate_pattern_confidence(self, pattern_matches: List[Dict[str, Any]], anomaly_results: Dict[str, Any]) -> float:
        """Calculate confidence in pattern recognition analysis."""
        
        confidence_factors = []
        
        # Pattern matching confidence
        if pattern_matches:
            # Higher confidence with stronger matches and proven patterns
            match_confidences = [
                match["match_strength"] * match.get("accuracy", 0.5)
                for match in pattern_matches
            ]
            pattern_confidence = max(match_confidences) if match_confidences else 0.5
            confidence_factors.append(pattern_confidence)
        else:
            confidence_factors.append(0.3)  # Lower confidence with no pattern matches
        
        # Anomaly detection confidence
        baseline_available = anomaly_results.get("baseline_available", False)
        if baseline_available:
            # Confidence based on amount of historical data
            history_size = len(self.transaction_history)
            history_confidence = min(1.0, history_size / 1000)  # Full confidence at 1000+ transactions
            confidence_factors.append(history_confidence)
        else:
            confidence_factors.append(0.2)  # Low confidence without baseline
        
        return np.mean(confidence_factors)
    
    def _generate_risk_factors(self, pattern_matches: List[Dict[str, Any]], anomaly_results: Dict[str, Any]) -> List[str]:
        """Generate risk factors based on pattern analysis."""
        
        risk_factors = []
        
        # Pattern-based risk factors
        for match in pattern_matches[:3]:  # Top 3 matches
            risk_factors.append(
                f"Matches {match['pattern_type']} pattern (strength: {match['match_strength']:.2f})"
            )
        
        # Anomaly-based risk factors
        anomalies = anomaly_results.get("anomalies", [])
        risk_factors.extend(anomalies[:2])  # Top 2 anomalies
        
        return risk_factors
    
    def _generate_protective_factors(self, pattern_matches: List[Dict[str, Any]], anomaly_results: Dict[str, Any]) -> List[str]:
        """Generate protective factors based on pattern analysis."""
        
        protective_factors = []
        
        # No strong pattern matches is protective
        if not pattern_matches or max(match["match_strength"] for match in pattern_matches) < 0.5:
            protective_factors.append("No strong fraud pattern matches detected")
        
        # No anomalies is protective
        if anomaly_results.get("anomaly_count", 0) == 0:
            protective_factors.append("Transaction follows normal statistical patterns")
        
        # Historical consistency
        if len(self.transaction_history) > 100:
            protective_factors.append("Sufficient transaction history available for analysis")
        
        return protective_factors