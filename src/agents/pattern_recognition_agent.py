import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import json
from collections import defaultdict, Counter

from .base_agent import BaseAgent
from ..utils.vector_db import InMemoryVectorDB

logger = logging.getLogger(__name__)

class PatternRecognitionAgent(BaseAgent):
    """Agent specialized in discovering and analyzing fraud patterns"""
    
    def __init__(self, agent_id: str = "pattern_recognition_001"):
        super().__init__(agent_id, "pattern_recognition")
        self.vector_db = InMemoryVectorDB()
        self.known_patterns: Dict[str, Dict[str, Any]] = {}
        self.pattern_frequency: Dict[str, int] = defaultdict(int)
        self.suspicious_patterns: List[Dict[str, Any]] = []
        self.pattern_evolution_tracker: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._initialize_base_patterns()
        
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze input data to discover and recognize fraud patterns"""
        start_time = datetime.utcnow()
        
        try:
            # Extract transaction data
            transaction_data = input_data.get("transaction", {})
            historical_data = input_data.get("historical_data", [])
            
            # Discover new patterns
            new_patterns = await self._discover_patterns(transaction_data, historical_data)
            
            # Recognize existing patterns
            recognized_patterns = await self._recognize_patterns(transaction_data)
            
            # Analyze pattern evolution
            evolution_analysis = await self._analyze_pattern_evolution()
            
            # Calculate risk score based on patterns
            pattern_risk_score = await self._calculate_pattern_risk(recognized_patterns, new_patterns)
            
            # Generate insights
            insights = await self._generate_pattern_insights(new_patterns, recognized_patterns)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            confidence = self._calculate_confidence({
                "data_quality": len(transaction_data) / 10,  # Normalize by expected fields
                "pattern_match": len(recognized_patterns) / 5,  # Normalize by typical pattern count
                "historical_accuracy": self.performance_metrics["correct_decisions"] / max(1, self.performance_metrics["total_decisions"])
            })
            
            result = {
                "agent_id": self.agent_id,
                "new_patterns_discovered": len(new_patterns),
                "recognized_patterns": recognized_patterns,
                "new_patterns": new_patterns,
                "pattern_risk_score": pattern_risk_score,
                "evolution_analysis": evolution_analysis,
                "insights": insights,
                "confidence": confidence,
                "processing_time_ms": processing_time
            }
            
            await self.record_decision(input_data, result, processing_time, confidence)
            return result
            
        except Exception as e:
            logger.error(f"Pattern recognition processing failed: {str(e)}")
            return {
                "agent_id": self.agent_id,
                "error": str(e),
                "confidence": 0.0
            }
    
    async def learn(self, feedback: Dict[str, Any]) -> None:
        """Learn from feedback to improve pattern recognition"""
        try:
            pattern_id = feedback.get("pattern_id")
            was_accurate = feedback.get("was_accurate", False)
            actual_fraud = feedback.get("actual_fraud", False)
            
            if pattern_id and pattern_id in self.known_patterns:
                pattern = self.known_patterns[pattern_id]
                
                # Update pattern confidence based on feedback
                if was_accurate:
                    pattern["confidence"] = min(1.0, pattern["confidence"] + 0.05)
                    pattern["true_positives"] = pattern.get("true_positives", 0) + (1 if actual_fraud else 0)
                else:
                    pattern["confidence"] = max(0.0, pattern["confidence"] - 0.1)
                    pattern["false_positives"] = pattern.get("false_positives", 0) + (1 if not actual_fraud else 0)
                
                pattern["last_updated"] = datetime.utcnow().isoformat()
                
                # Update vector database representation
                pattern_text = self._pattern_to_text(pattern)
                self.vector_db.update_pattern(pattern_id, pattern_text, pattern)
                
                logger.info(f"Updated pattern {pattern_id} confidence to {pattern['confidence']:.3f}")
            
        except Exception as e:
            logger.error(f"Pattern learning failed: {str(e)}")
    
    async def get_reasoning(self, input_data: Dict[str, Any]) -> str:
        """Get human-readable reasoning for pattern recognition decisions"""
        transaction = input_data.get("transaction", {})
        
        reasoning_prompt = f"""
        Explain the pattern recognition analysis for this transaction:
        
        Transaction Details:
        - Amount: ${transaction.get('amount', 0):,.2f}
        - Time: {transaction.get('timestamp', 'unknown')}
        - Merchant: {transaction.get('merchant_category', 'unknown')}
        - Location: {transaction.get('location', {}).get('country', 'unknown')}
        
        Known Patterns: {len(self.known_patterns)}
        Recent Suspicious Patterns: {len(self.suspicious_patterns)}
        
        Provide a clear explanation of what patterns were detected and why they might indicate fraud risk.
        """
        
        ai_reasoning = await self._generate_ai_response(reasoning_prompt, max_tokens=300)
        
        if ai_reasoning:
            return ai_reasoning
        
        # Fallback reasoning
        return f"Pattern analysis complete. Analyzed {len(self.known_patterns)} known patterns and discovered new patterns based on transaction characteristics."
    
    async def _discover_patterns(self, transaction_data: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover new fraud patterns from transaction data"""
        new_patterns = []
        
        # Amount-based pattern discovery
        amount_patterns = await self._discover_amount_patterns(transaction_data, historical_data)
        new_patterns.extend(amount_patterns)
        
        # Temporal pattern discovery
        temporal_patterns = await self._discover_temporal_patterns(transaction_data, historical_data)
        new_patterns.extend(temporal_patterns)
        
        # Geographic pattern discovery
        geo_patterns = await self._discover_geographic_patterns(transaction_data, historical_data)
        new_patterns.extend(geo_patterns)
        
        # Velocity pattern discovery
        velocity_patterns = await self._discover_velocity_patterns(transaction_data, historical_data)
        new_patterns.extend(velocity_patterns)
        
        # Store new patterns
        for pattern in new_patterns:
            pattern_id = f"pattern_{len(self.known_patterns) + len(new_patterns)}"
            pattern["pattern_id"] = pattern_id
            pattern["discovered_at"] = datetime.utcnow().isoformat()
            pattern["confidence"] = 0.5  # Initial confidence
            
            # Add to knowledge base
            self.known_patterns[pattern_id] = pattern
            
            # Add to vector database
            pattern_text = self._pattern_to_text(pattern)
            self.vector_db.add_pattern(pattern_id, pattern_text, pattern)
        
        return new_patterns
    
    async def _discover_amount_patterns(self, transaction: Dict[str, Any], historical: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover patterns related to transaction amounts"""
        patterns = []
        current_amount = transaction.get("amount", 0)
        
        if not historical:
            return patterns
        
        # Analyze amount distribution
        amounts = [tx.get("amount", 0) for tx in historical]
        if amounts:
            mean_amount = np.mean(amounts)
            std_amount = np.std(amounts)
            
            # Detect unusual amount patterns
            if current_amount > mean_amount + 3 * std_amount:
                patterns.append({
                    "pattern_type": "unusual_high_amount",
                    "description": f"Transaction amount ${current_amount:,.2f} is {(current_amount - mean_amount) / std_amount:.1f} std devs above normal",
                    "risk_score": min(1.0, (current_amount - mean_amount) / (3 * std_amount)),
                    "features": {
                        "amount": current_amount,
                        "historical_mean": mean_amount,
                        "z_score": (current_amount - mean_amount) / std_amount
                    }
                })
            
            # Detect round number patterns
            if current_amount > 1000 and current_amount % 100 == 0:
                round_pattern_frequency = sum(1 for amt in amounts if amt % 100 == 0 and amt > 1000)
                if round_pattern_frequency < len(amounts) * 0.1:  # Less than 10% are round
                    patterns.append({
                        "pattern_type": "suspicious_round_amount",
                        "description": f"Large round amount ${current_amount:,.2f} is unusual for this user",
                        "risk_score": 0.6,
                        "features": {
                            "amount": current_amount,
                            "is_round": True,
                            "round_frequency": round_pattern_frequency / len(amounts)
                        }
                    })
        
        return patterns
    
    async def _discover_temporal_patterns(self, transaction: Dict[str, Any], historical: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover time-based fraud patterns"""
        patterns = []
        
        try:
            current_time = datetime.fromisoformat(transaction.get("timestamp", "").replace("Z", "+00:00"))
            current_hour = current_time.hour
        except:
            return patterns
        
        if not historical:
            return patterns
        
        # Analyze historical time patterns
        historical_hours = []
        for tx in historical:
            try:
                tx_time = datetime.fromisoformat(tx.get("timestamp", "").replace("Z", "+00:00"))
                historical_hours.append(tx_time.hour)
            except:
                continue
        
        if historical_hours:
            hour_distribution = Counter(historical_hours)
            typical_hours = [hour for hour, count in hour_distribution.items() if count > len(historical_hours) * 0.1]
            
            # Detect unusual time patterns
            if current_hour not in typical_hours and (current_hour < 6 or current_hour > 22):
                patterns.append({
                    "pattern_type": "unusual_time",
                    "description": f"Transaction at {current_hour}:00 is unusual for this user",
                    "risk_score": 0.7,
                    "features": {
                        "transaction_hour": current_hour,
                        "typical_hours": typical_hours,
                        "is_night_time": current_hour < 6 or current_hour > 22
                    }
                })
        
        return patterns
    
    async def _discover_geographic_patterns(self, transaction: Dict[str, Any], historical: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover location-based fraud patterns"""
        patterns = []
        
        current_location = transaction.get("location", {})
        if not current_location:
            return patterns
        
        current_country = current_location.get("country")
        if not current_country:
            return patterns
        
        # Analyze historical locations
        historical_countries = [tx.get("location", {}).get("country") 
                              for tx in historical if tx.get("location", {}).get("country")]
        
        if historical_countries:
            country_distribution = Counter(historical_countries)
            
            # Detect new country pattern
            if current_country not in country_distribution:
                patterns.append({
                    "pattern_type": "new_country",
                    "description": f"First transaction from {current_country}",
                    "risk_score": 0.8,
                    "features": {
                        "current_country": current_country,
                        "historical_countries": list(country_distribution.keys())
                    }
                })
            
            # Detect high-risk country patterns
            high_risk_countries = ["NG", "PK", "RU", "CN", "IR", "KP", "AF", "SY"]
            if current_country in high_risk_countries:
                patterns.append({
                    "pattern_type": "high_risk_country",
                    "description": f"Transaction from high-risk country: {current_country}",
                    "risk_score": 0.9,
                    "features": {
                        "current_country": current_country,
                        "risk_level": "high"
                    }
                })
        
        return patterns
    
    async def _discover_velocity_patterns(self, transaction: Dict[str, Any], historical: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Discover velocity-based fraud patterns"""
        patterns = []
        
        try:
            current_time = datetime.fromisoformat(transaction.get("timestamp", "").replace("Z", "+00:00"))
        except:
            return patterns
        
        # Count recent transactions
        recent_1h = 0
        recent_24h = 0
        
        for tx in historical:
            try:
                tx_time = datetime.fromisoformat(tx.get("timestamp", "").replace("Z", "+00:00"))
                time_diff = (current_time - tx_time).total_seconds()
                
                if time_diff <= 3600:  # 1 hour
                    recent_1h += 1
                if time_diff <= 86400:  # 24 hours
                    recent_24h += 1
            except:
                continue
        
        # Detect high velocity patterns
        if recent_1h > 5:
            patterns.append({
                "pattern_type": "high_velocity_1h",
                "description": f"{recent_1h} transactions in the last hour",
                "risk_score": min(1.0, recent_1h / 10),
                "features": {
                    "transactions_1h": recent_1h,
                    "velocity_score": recent_1h / 10
                }
            })
        
        if recent_24h > 20:
            patterns.append({
                "pattern_type": "high_velocity_24h",
                "description": f"{recent_24h} transactions in the last 24 hours",
                "risk_score": min(1.0, recent_24h / 50),
                "features": {
                    "transactions_24h": recent_24h,
                    "daily_velocity": recent_24h
                }
            })
        
        return patterns
    
    async def _recognize_patterns(self, transaction_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recognize existing patterns in transaction data"""
        recognized = []
        
        # Extract transaction features
        amount = transaction_data.get("amount", 0)
        merchant_category = transaction_data.get("merchant_category", "")
        country = transaction_data.get("location", {}).get("country", "")
        timestamp = transaction_data.get("timestamp", "")
        
        # Parse hour if timestamp available
        hour = None
        if timestamp:
            try:
                tx_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                hour = tx_time.hour
            except:
                pass
        
        # Check each known pattern for explicit matches
        for pattern_id, pattern in self.known_patterns.items():
            match_strength = 0.0
            matched_features = []
            
            features = pattern.get("features", {})
            
            # Check merchant category match
            if "merchant_category" in features and features["merchant_category"] == merchant_category:
                match_strength += 0.4
                matched_features.append("merchant_category")
            
            # Check amount threshold
            if "amount_threshold" in features and amount >= features["amount_threshold"]:
                match_strength += 0.3
                matched_features.append("amount_threshold")
            
            # Check high-risk countries
            if "high_risk_countries" in features and country in features["high_risk_countries"]:
                match_strength += 0.4
                matched_features.append("high_risk_country")
            
            # Check night hours
            if "night_hours" in features and hour is not None and hour in features["night_hours"]:
                match_strength += 0.3
                matched_features.append("night_hours")
            
            # If we have a significant match, add it
            if match_strength >= 0.3:  # Lower threshold for matching
                recognized.append({
                    "pattern_id": pattern_id,
                    "pattern_type": pattern.get("pattern_type", "unknown"),
                    "similarity_score": match_strength,
                    "confidence": pattern.get("confidence", 0.5),
                    "risk_score": pattern.get("risk_score", 0.5),
                    "description": pattern.get("description", ""),
                    "match_strength": match_strength * pattern.get("confidence", 0.5),
                    "matched_features": matched_features
                })
        
        # Also do vector similarity search as backup
        transaction_text = self._transaction_to_text(transaction_data)
        similar_patterns = self.vector_db.search_similar_patterns(transaction_text, top_k=3, threshold=0.6)
        
        for pattern_id, similarity, pattern_data in similar_patterns:
            # Avoid duplicates
            if not any(p["pattern_id"] == pattern_id for p in recognized):
                recognized.append({
                    "pattern_id": pattern_id,
                    "pattern_type": pattern_data.get("pattern_type", "unknown"),
                    "similarity_score": similarity,
                    "confidence": pattern_data.get("confidence", 0.5),
                    "risk_score": pattern_data.get("risk_score", 0.5),
                    "description": pattern_data.get("description", ""),
                    "match_strength": similarity * pattern_data.get("confidence", 0.5),
                    "matched_features": ["vector_similarity"]
                })
        
        return recognized
    
    async def _analyze_pattern_evolution(self) -> Dict[str, Any]:
        """Analyze how patterns are evolving over time"""
        evolution_stats = {
            "total_patterns": len(self.known_patterns),
            "new_patterns_last_24h": 0,
            "evolving_patterns": [],
            "stable_patterns": 0
        }
        
        current_time = datetime.utcnow()
        
        for pattern_id, pattern in self.known_patterns.items():
            try:
                created_time = datetime.fromisoformat(pattern.get("discovered_at", ""))
                age_hours = (current_time - created_time).total_seconds() / 3600
                
                if age_hours <= 24:
                    evolution_stats["new_patterns_last_24h"] += 1
                
                # Check if pattern is evolving (confidence changing)
                if pattern_id in self.pattern_evolution_tracker:
                    confidence_history = [entry.get("confidence", 0.5) 
                                        for entry in self.pattern_evolution_tracker[pattern_id]]
                    if len(confidence_history) > 1:
                        confidence_trend = confidence_history[-1] - confidence_history[0]
                        if abs(confidence_trend) > 0.1:
                            evolution_stats["evolving_patterns"].append({
                                "pattern_id": pattern_id,
                                "trend": "increasing" if confidence_trend > 0 else "decreasing",
                                "confidence_change": confidence_trend
                            })
                        else:
                            evolution_stats["stable_patterns"] += 1
                
            except:
                continue
        
        return evolution_stats
    
    async def _calculate_pattern_risk(self, recognized_patterns: List[Dict[str, Any]], 
                                    new_patterns: List[Dict[str, Any]]) -> float:
        """Calculate overall risk score based on pattern analysis"""
        if not recognized_patterns and not new_patterns:
            return 0.0
        
        # Weight recognized patterns by their match strength
        recognized_risk = 0.0
        for pattern in recognized_patterns:
            risk_contribution = pattern.get("risk_score", 0.5) * pattern.get("match_strength", 0.5)
            recognized_risk += risk_contribution
        
        # Weight new patterns by their initial risk scores
        new_pattern_risk = sum(pattern.get("risk_score", 0.5) for pattern in new_patterns)
        
        # Combine risks (recognized patterns weighted more heavily)
        total_risk = (0.7 * recognized_risk + 0.3 * new_pattern_risk)
        
        return min(1.0, total_risk)
    
    async def _generate_pattern_insights(self, new_patterns: List[Dict[str, Any]], 
                                       recognized_patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable insights from pattern analysis"""
        insights = []
        
        if new_patterns:
            insights.append(f"Discovered {len(new_patterns)} new suspicious patterns")
            
            # Highlight most concerning new patterns
            high_risk_new = [p for p in new_patterns if p.get("risk_score", 0) > 0.7]
            if high_risk_new:
                insights.append(f"{len(high_risk_new)} new patterns show high fraud risk")
        
        if recognized_patterns:
            insights.append(f"Matched {len(recognized_patterns)} known fraud patterns")
            
            # Highlight strongest matches
            strong_matches = [p for p in recognized_patterns if p.get("match_strength", 0) > 0.8]
            if strong_matches:
                insights.append(f"{len(strong_matches)} patterns show strong similarity to known fraud")
        
        # Pattern frequency insights
        if len(self.known_patterns) > 100:
            insights.append("Large pattern database enables sophisticated fraud detection")
        
        return insights
    
    def _pattern_to_text(self, pattern: Dict[str, Any]) -> str:
        """Convert pattern to text representation for vector database"""
        pattern_type = pattern.get("pattern_type", "unknown")
        description = pattern.get("description", "")
        features = pattern.get("features", {})
        
        text_parts = [
            f"Pattern type: {pattern_type}",
            f"Description: {description}"
        ]
        
        # Add feature descriptions
        for key, value in features.items():
            text_parts.append(f"{key}: {value}")
        
        return " | ".join(text_parts)
    
    def _transaction_to_text(self, transaction: Dict[str, Any]) -> str:
        """Convert transaction to text representation for pattern matching"""
        amount = transaction.get("amount", 0)
        timestamp = transaction.get("timestamp", "")
        merchant_category = transaction.get("merchant_category", "unknown")
        location = transaction.get("location", {})
        
        hour = "unknown"
        try:
            hour = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).hour
        except:
            pass
        
        text_parts = [
            f"Amount: ${amount:,.2f}",
            f"Hour: {hour}",
            f"Merchant category: {merchant_category}",
            f"Country: {location.get('country', 'unknown')}"
        ]
        
        return " | ".join(text_parts)
    
    async def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of current pattern knowledge"""
        pattern_types = defaultdict(int)
        high_confidence_patterns = 0
        
        for pattern in self.known_patterns.values():
            pattern_types[pattern.get("pattern_type", "unknown")] += 1
            if pattern.get("confidence", 0) > 0.8:
                high_confidence_patterns += 1
        
        return {
            "total_patterns": len(self.known_patterns),
            "pattern_types": dict(pattern_types),
            "high_confidence_patterns": high_confidence_patterns,
            "vector_db_stats": self.vector_db.get_stats(),
            "suspicious_patterns": len(self.suspicious_patterns)
        }
    
    def _initialize_base_patterns(self):
        """Initialize the agent with basic fraud patterns"""
        base_patterns = [
            {
                "pattern_id": "high_amount_crypto",
                "pattern_type": "amount_merchant",
                "description": "High amount cryptocurrency transactions",
                "risk_score": 0.8,
                "confidence": 0.9,
                "features": {
                    "merchant_category": "cryptocurrency",
                    "amount_threshold": 2000,
                    "risk_factors": ["high_amount", "crypto"]
                }
            },
            {
                "pattern_id": "night_high_risk_country",
                "pattern_type": "temporal_geographic",
                "description": "Night transactions from high-risk countries",
                "risk_score": 0.9,
                "confidence": 0.85,
                "features": {
                    "high_risk_countries": ["NG", "PK", "RU"],
                    "night_hours": [22, 23, 0, 1, 2, 3, 4, 5],
                    "risk_factors": ["night_time", "high_risk_location"]
                }
            },
            {
                "pattern_id": "gambling_large_amount",
                "pattern_type": "merchant_amount",
                "description": "Large gambling transactions",
                "risk_score": 0.7,
                "confidence": 0.8,
                "features": {
                    "merchant_category": "gambling",
                    "amount_threshold": 1000,
                    "risk_factors": ["gambling", "high_amount"]
                }
            },
            {
                "pattern_id": "money_transfer_suspicious",
                "pattern_type": "merchant_geographic",
                "description": "Money transfers to high-risk countries",
                "risk_score": 0.8,
                "confidence": 0.9,
                "features": {
                    "merchant_category": "money_transfer",
                    "high_risk_countries": ["NG", "PK", "AF"],
                    "risk_factors": ["money_transfer", "high_risk_destination"]
                }
            }
        ]
        
        for pattern in base_patterns:
            pattern_id = pattern["pattern_id"]
            self.known_patterns[pattern_id] = pattern
            
            # Add to vector database
            pattern_text = self._pattern_to_text(pattern)
            self.vector_db.add_pattern(pattern_id, pattern_text, pattern)