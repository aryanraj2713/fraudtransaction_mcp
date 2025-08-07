import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from mcp.server import Server
from mcp.types import Tool, TextContent

from ..schemas.transaction_schema import TransactionSchema, TransactionStatus
from ..utils.config import config

logger = logging.getLogger(__name__)

class DataIntelligenceServer:
    def __init__(self):
        self.server = Server("data-intelligence-server")
        self.transaction_buffer: List[Dict[str, Any]] = []
        self.feature_cache: Dict[str, Any] = {}
        self.data_quality_metrics: Dict[str, float] = {}
        self.drift_detector = IsolationForest(contamination=0.1, random_state=42)
        self.baseline_features: Optional[np.ndarray] = None
        self.scaler = StandardScaler()
        self.setup_tools()
        
    def setup_tools(self):
        """Register MCP tools for data intelligence operations"""
        
        @self.server.tool()
        async def ingest_transaction(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
            """Ingest and preprocess transaction data"""
            try:
                # Validate transaction schema
                transaction = TransactionSchema(**transaction_data)
                
                # Add to buffer for batch processing
                self.transaction_buffer.append(transaction.dict())
                
                # Basic preprocessing
                processed_data = await self._preprocess_transaction(transaction)
                
                logger.info(f"Ingested transaction {transaction.transaction_id}")
                
                return {
                    "status": "success",
                    "transaction_id": transaction.transaction_id,
                    "processed_features": processed_data,
                    "buffer_size": len(self.transaction_buffer)
                }
                
            except Exception as e:
                logger.error(f"Transaction ingestion failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e),
                    "transaction_id": transaction_data.get("transaction_id", "unknown")
                }
        
        @self.server.tool()
        async def validate_data(data: Dict[str, Any]) -> Dict[str, Any]:
            """Perform comprehensive data quality assessment"""
            try:
                validation_results = await self._validate_transaction_data(data)
                
                # Update quality metrics
                self._update_quality_metrics(validation_results)
                
                return {
                    "status": "success",
                    "validation_results": validation_results,
                    "quality_score": validation_results.get("overall_score", 0.0),
                    "issues": validation_results.get("issues", [])
                }
                
            except Exception as e:
                logger.error(f"Data validation failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def engineer_features(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
            """Discover and engineer new features from transaction data"""
            try:
                features = await self._engineer_features(transaction_data)
                
                # Cache features for reuse
                transaction_id = transaction_data.get("transaction_id")
                if transaction_id:
                    self.feature_cache[transaction_id] = features
                
                return {
                    "status": "success",
                    "transaction_id": transaction_id,
                    "features": features,
                    "feature_count": len(features)
                }
                
            except Exception as e:
                logger.error(f"Feature engineering failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def detect_drift(reference_period_hours: int = 24) -> Dict[str, Any]:
            """Detect data drift in transaction patterns"""
            try:
                drift_results = await self._detect_data_drift(reference_period_hours)
                
                return {
                    "status": "success",
                    "drift_detected": drift_results["drift_detected"],
                    "drift_score": drift_results["drift_score"],
                    "affected_features": drift_results["affected_features"],
                    "recommendation": drift_results["recommendation"]
                }
                
            except Exception as e:
                logger.error(f"Drift detection failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
    
    async def _preprocess_transaction(self, transaction: TransactionSchema) -> Dict[str, Any]:
        """Preprocess transaction for fraud detection"""
        processed = {}
        
        # Temporal features
        processed["hour_of_day"] = transaction.timestamp.hour
        processed["day_of_week"] = transaction.timestamp.weekday()
        processed["is_weekend"] = transaction.timestamp.weekday() >= 5
        
        # Amount features
        processed["amount"] = transaction.amount
        processed["amount_log"] = np.log1p(transaction.amount)
        processed["amount_rounded"] = transaction.amount % 1 == 0
        
        # Location features
        if transaction.location:
            processed["has_location"] = True
            processed["location_country"] = transaction.location.get("country", "unknown")
        else:
            processed["has_location"] = False
            processed["location_country"] = "unknown"
        
        # Device features
        if transaction.device_info:
            processed["has_device_info"] = True
            processed["device_type"] = transaction.device_info.get("type", "unknown")
        else:
            processed["has_device_info"] = False
            processed["device_type"] = "unknown"
        
        # Merchant features
        processed["has_merchant"] = transaction.merchant_id is not None
        processed["merchant_category"] = transaction.merchant_category or "unknown"
        
        return processed
    
    async def _validate_transaction_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive data validation"""
        issues = []
        scores = {}
        
        # Required field validation
        required_fields = ["transaction_id", "user_id", "amount", "timestamp"]
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]
        if missing_fields:
            issues.append(f"Missing required fields: {missing_fields}")
            scores["completeness"] = 1.0 - (len(missing_fields) / len(required_fields))
        else:
            scores["completeness"] = 1.0
        
        # Data type validation
        type_issues = 0
        if "amount" in data:
            try:
                float(data["amount"])
                if float(data["amount"]) <= 0:
                    issues.append("Amount must be positive")
                    type_issues += 1
            except (ValueError, TypeError):
                issues.append("Amount must be numeric")
                type_issues += 1
        
        scores["type_validity"] = max(0.0, 1.0 - (type_issues / 5))  # Assume 5 key numeric fields
        
        # Consistency checks
        consistency_score = 1.0
        if "timestamp" in data:
            try:
                ts = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
                if ts > datetime.utcnow():
                    issues.append("Transaction timestamp is in the future")
                    consistency_score -= 0.3
            except:
                issues.append("Invalid timestamp format")
                consistency_score -= 0.5
        
        scores["consistency"] = max(0.0, consistency_score)
        
        # Overall quality score
        overall_score = np.mean(list(scores.values()))
        
        return {
            "overall_score": overall_score,
            "component_scores": scores,
            "issues": issues,
            "validation_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _engineer_features(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced feature engineering"""
        features = {}
        
        # Time-based features
        if "timestamp" in transaction_data:
            ts = datetime.fromisoformat(transaction_data["timestamp"].replace("Z", "+00:00"))
            features["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
            features["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)
            features["day_sin"] = np.sin(2 * np.pi * ts.weekday() / 7)
            features["day_cos"] = np.cos(2 * np.pi * ts.weekday() / 7)
            features["is_business_hours"] = 9 <= ts.hour <= 17
            features["is_late_night"] = ts.hour >= 23 or ts.hour <= 5
        
        # Amount-based features
        if "amount" in transaction_data:
            amount = float(transaction_data["amount"])
            features["amount_log"] = np.log1p(amount)
            features["amount_sqrt"] = np.sqrt(amount)
            features["is_round_amount"] = amount % 1 == 0
            features["amount_digits"] = len(str(int(amount)))
        
        # Velocity features (requires historical data)
        user_id = transaction_data.get("user_id")
        if user_id:
            features.update(await self._calculate_velocity_features(user_id, transaction_data))
        
        # Pattern-based features
        features.update(await self._extract_pattern_features(transaction_data))
        
        return features
    
    async def _calculate_velocity_features(self, user_id: str, current_transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate user transaction velocity features"""
        features = {}
        
        # Get recent transactions for this user
        recent_transactions = [
            tx for tx in self.transaction_buffer 
            if tx.get("user_id") == user_id and 
            tx.get("transaction_id") != current_transaction.get("transaction_id")
        ]
        
        if not recent_transactions:
            return {
                "tx_count_1h": 0,
                "tx_count_24h": 0,
                "avg_amount_24h": 0.0,
                "velocity_score": 0.0
            }
        
        current_time = datetime.fromisoformat(current_transaction["timestamp"].replace("Z", "+00:00"))
        
        # Count transactions in different time windows
        tx_1h = sum(1 for tx in recent_transactions 
                   if (current_time - datetime.fromisoformat(tx["timestamp"].replace("Z", "+00:00"))).seconds <= 3600)
        tx_24h = sum(1 for tx in recent_transactions 
                    if (current_time - datetime.fromisoformat(tx["timestamp"].replace("Z", "+00:00"))).days == 0)
        
        # Average amount in 24h
        amounts_24h = [tx["amount"] for tx in recent_transactions 
                      if (current_time - datetime.fromisoformat(tx["timestamp"].replace("Z", "+00:00"))).days == 0]
        avg_amount_24h = np.mean(amounts_24h) if amounts_24h else 0.0
        
        # Velocity score (higher = more suspicious)
        velocity_score = min(1.0, (tx_1h * 0.3 + tx_24h * 0.1) / 10)
        
        features.update({
            "tx_count_1h": tx_1h,
            "tx_count_24h": tx_24h,
            "avg_amount_24h": avg_amount_24h,
            "velocity_score": velocity_score
        })
        
        return features
    
    async def _extract_pattern_features(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pattern-based features"""
        features = {}
        
        # IP-based patterns
        if "ip_address" in transaction_data:
            ip = transaction_data["ip_address"]
            features["ip_is_private"] = any(ip.startswith(prefix) for prefix in ["192.168.", "10.", "172."])
            features["ip_segments"] = len(ip.split("."))
        
        # Merchant patterns
        if "merchant_category" in transaction_data:
            high_risk_categories = ["gambling", "adult", "cryptocurrency", "money_transfer"]
            features["is_high_risk_merchant"] = transaction_data["merchant_category"] in high_risk_categories
        
        # Device patterns
        if "device_info" in transaction_data and transaction_data["device_info"]:
            device_info = transaction_data["device_info"]
            features["is_mobile"] = device_info.get("type", "").lower() in ["mobile", "tablet"]
            features["has_user_agent"] = "user_agent" in device_info
        
        return features
    
    async def _detect_data_drift(self, reference_period_hours: int) -> Dict[str, Any]:
        """Detect data drift using statistical methods"""
        if len(self.transaction_buffer) < 100:  # Need minimum samples
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "affected_features": [],
                "recommendation": "Insufficient data for drift detection"
            }
        
        # Get reference period data
        current_time = datetime.utcnow()
        reference_cutoff = current_time - timedelta(hours=reference_period_hours)
        
        reference_data = []
        current_data = []
        
        for tx in self.transaction_buffer:
            tx_time = datetime.fromisoformat(tx["timestamp"].replace("Z", "+00:00"))
            features = await self._engineer_features(tx)
            feature_vector = [features.get(key, 0) for key in sorted(features.keys()) if isinstance(features.get(key), (int, float))]
            
            if tx_time < reference_cutoff:
                reference_data.append(feature_vector)
            else:
                current_data.append(feature_vector)
        
        if not reference_data or not current_data:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "affected_features": [],
                "recommendation": "Insufficient data in time windows"
            }
        
        # Convert to numpy arrays
        reference_array = np.array(reference_data)
        current_array = np.array(current_data)
        
        # Fit drift detector on reference data
        if self.baseline_features is None:
            self.baseline_features = reference_array
            self.drift_detector.fit(reference_array)
        
        # Detect anomalies in current data
        anomaly_scores = self.drift_detector.decision_function(current_array)
        anomaly_ratio = np.mean(anomaly_scores < 0)  # Negative scores indicate anomalies
        
        drift_detected = anomaly_ratio > 0.15  # 15% threshold
        drift_score = float(anomaly_ratio)
        
        # Identify affected features (simplified)
        affected_features = []
        if drift_detected:
            # Calculate feature-wise drift (simplified approach)
            for i, feature_name in enumerate(sorted([k for k in await self._engineer_features(self.transaction_buffer[0]) if isinstance((await self._engineer_features(self.transaction_buffer[0]))[k], (int, float))])):
                ref_mean = np.mean(reference_array[:, i])
                curr_mean = np.mean(current_array[:, i])
                if abs(ref_mean - curr_mean) > 0.1 * abs(ref_mean):
                    affected_features.append(feature_name)
        
        recommendation = "Model retraining recommended" if drift_detected else "No action needed"
        
        return {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "affected_features": affected_features,
            "recommendation": recommendation
        }
    
    def _update_quality_metrics(self, validation_results: Dict[str, Any]) -> None:
        """Update running data quality metrics"""
        overall_score = validation_results.get("overall_score", 0.0)
        
        if "overall_quality" not in self.data_quality_metrics:
            self.data_quality_metrics["overall_quality"] = overall_score
        else:
            # Exponential moving average
            alpha = 0.1
            self.data_quality_metrics["overall_quality"] = (
                alpha * overall_score + (1 - alpha) * self.data_quality_metrics["overall_quality"]
            )
        
        # Update component metrics
        for component, score in validation_results.get("component_scores", {}).items():
            key = f"{component}_quality"
            if key not in self.data_quality_metrics:
                self.data_quality_metrics[key] = score
            else:
                self.data_quality_metrics[key] = alpha * score + (1 - alpha) * self.data_quality_metrics[key]
    
    def get_server(self) -> Server:
        """Get the MCP server instance"""
        return self.server
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current server metrics"""
        return {
            "buffer_size": len(self.transaction_buffer),
            "feature_cache_size": len(self.feature_cache),
            "quality_metrics": self.data_quality_metrics,
            "drift_detector_fitted": self.baseline_features is not None
        }