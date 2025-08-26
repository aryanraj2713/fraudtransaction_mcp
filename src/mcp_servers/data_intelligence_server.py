import asyncio
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_classif
from scipy import stats
import pickle
import aiofiles

from .base_server import BaseMCPServer
from schemas.mcp_tools_schema import DATA_INTELLIGENCE_TOOLS
from schemas.transaction_schema import Transaction, TransactionBatch

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    completeness: float
    validity: float
    consistency: float
    accuracy: float
    uniqueness: float
    timeliness: float
    overall_score: float


@dataclass
class DriftDetectionResult:
    has_drift: bool
    drift_score: float
    affected_features: List[str]
    drift_method: str
    details: Dict[str, Any]


class FeatureEngineer:
    """Advanced feature engineering component."""
    
    def __init__(self):
        self.feature_history = {}
        self.scalers = {}
        self.encoders = {}
        
    async def engineer_features(self, raw_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer features from raw transaction data."""
        try:
            transaction = Transaction(**raw_data)
            features = {}
            
            # Basic transaction features
            features.update(self._extract_basic_features(transaction))
            
            # Temporal features
            features.update(self._extract_temporal_features(transaction))
            
            # Velocity features
            features.update(await self._extract_velocity_features(transaction))
            
            # Device and behavioral features
            features.update(self._extract_device_features(transaction))
            
            # Geographic features
            features.update(self._extract_geographic_features(transaction))
            
            # Amount-based features
            features.update(self._extract_amount_features(transaction))
            
            # Advanced pattern features
            if config.get("advanced_patterns", True):
                features.update(await self._extract_pattern_features(transaction))
            
            return features
            
        except Exception as e:
            logger.error(f"Feature engineering error: {e}")
            raise
    
    def _extract_basic_features(self, transaction: Transaction) -> Dict[str, Any]:
        """Extract basic features from transaction."""
        features = {
            "amount": float(transaction.amount),
            "transaction_type_encoded": self._encode_categorical(
                "transaction_type", transaction.transaction_type.value
            ),
            "currency_encoded": self._encode_categorical("currency", transaction.currency),
            "payment_method_encoded": self._encode_categorical(
                "payment_method", transaction.payment_method
            ),
            "is_first_transaction": int(transaction.is_first_transaction),
            "account_age_days": transaction.account_age_days or 0,
        }
        
        return features
    
    def _extract_temporal_features(self, transaction: Transaction) -> Dict[str, Any]:
        """Extract time-based features."""
        dt = transaction.timestamp
        
        features = {
            "hour_of_day": dt.hour,
            "day_of_week": dt.weekday(),
            "day_of_month": dt.day,
            "month": dt.month,
            "is_weekend": int(dt.weekday() >= 5),
            "is_business_hours": int(9 <= dt.hour <= 17),
            "is_late_night": int(dt.hour >= 22 or dt.hour <= 5),
        }
        
        return features
    
    async def _extract_velocity_features(self, transaction: Transaction) -> Dict[str, Any]:
        """Extract velocity-based features."""
        features = {
            "velocity_1h": transaction.velocity_1h or 0,
            "velocity_24h": transaction.velocity_24h or 0,
            "velocity_ratio": (transaction.velocity_1h or 0) / max(transaction.velocity_24h or 1, 1),
        }
        
        # Add user-specific velocity patterns
        user_history = self.feature_history.get(transaction.user_id, {})
        avg_daily_transactions = user_history.get("avg_daily_transactions", 1)
        
        features.update({
            "velocity_vs_user_avg": features["velocity_24h"] / max(avg_daily_transactions, 1),
            "is_velocity_spike": int(features["velocity_24h"] > avg_daily_transactions * 3),
        })
        
        return features
    
    def _extract_device_features(self, transaction: Transaction) -> Dict[str, Any]:
        """Extract device and behavioral features."""
        device = transaction.device_data
        behavior = transaction.user_behavior
        
        features = {
            "device_type_encoded": self._encode_categorical("device_type", device.device_type),
            "is_mobile": int(device.is_mobile),
        }
        
        if device.browser:
            features["browser_encoded"] = self._encode_categorical("browser", device.browser)
        
        if device.operating_system:
            features["os_encoded"] = self._encode_categorical("os", device.operating_system)
        
        if behavior:
            features.update({
                "session_duration_minutes": (behavior.session_duration or 0) / 60,
                "pages_visited": behavior.pages_visited or 0,
                "clicks_per_minute": behavior.clicks_per_minute or 0,
                "typing_speed": behavior.typing_speed or 0,
                "mouse_movements": behavior.mouse_movements or 0,
            })
        
        return features
    
    def _extract_geographic_features(self, transaction: Transaction) -> Dict[str, Any]:
        """Extract geographic features."""
        geo = transaction.geographic_data
        
        features = {
            "country_encoded": self._encode_categorical("country", geo.country),
            "has_coordinates": int(geo.latitude is not None and geo.longitude is not None),
        }
        
        if geo.city:
            features["city_encoded"] = self._encode_categorical("city", geo.city)
        
        # Add risk-based geographic scoring
        high_risk_countries = ["XX", "YY", "ZZ"]  # Example high-risk countries
        features["is_high_risk_country"] = int(geo.country in high_risk_countries)
        
        return features
    
    def _extract_amount_features(self, transaction: Transaction) -> Dict[str, Any]:
        """Extract amount-based features."""
        amount = float(transaction.amount)
        
        features = {
            "amount_log": np.log1p(amount),
            "amount_rounded": int(amount == round(amount)),
            "amount_decimal_places": len(str(amount).split(".")[-1]) if "." in str(amount) else 0,
        }
        
        # User-specific amount patterns
        user_history = self.feature_history.get(transaction.user_id, {})
        avg_amount = user_history.get("avg_amount", amount)
        
        features.update({
            "amount_vs_user_avg": amount / max(avg_amount, 1),
            "is_amount_outlier": int(amount > avg_amount * 5),
        })
        
        return features
    
    async def _extract_pattern_features(self, transaction: Transaction) -> Dict[str, Any]:
        """Extract advanced pattern-based features."""
        features = {}
        
        # Create composite identifiers for pattern detection
        device_fingerprint = hashlib.md5(
            f"{transaction.device_data.device_id}_{transaction.device_data.device_type}".encode()
        ).hexdigest()[:8]
        
        geo_fingerprint = hashlib.md5(
            f"{transaction.geographic_data.country}_{transaction.geographic_data.ip_address}".encode()
        ).hexdigest()[:8]
        
        features.update({
            "device_fingerprint_hash": hash(device_fingerprint) % 1000,
            "geo_fingerprint_hash": hash(geo_fingerprint) % 1000,
            "transaction_fingerprint": hash(
                f"{transaction.user_id}_{transaction.payment_method}_{device_fingerprint}"
            ) % 10000,
        })
        
        return features
    
    def _encode_categorical(self, feature_name: str, value: str) -> int:
        """Encode categorical variables consistently."""
        if feature_name not in self.encoders:
            self.encoders[feature_name] = LabelEncoder()
        
        encoder = self.encoders[feature_name]
        
        try:
            return encoder.transform([value])[0]
        except ValueError:
            # Handle unseen categories
            encoder.fit(list(encoder.classes_) + [value])
            return encoder.transform([value])[0]


class DataValidator:
    """Comprehensive data validation system."""
    
    def __init__(self):
        self.validation_rules = self._load_default_rules()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_fitted = False
    
    def _load_default_rules(self) -> Dict[str, Any]:
        """Load default validation rules."""
        return {
            "amount_range": {"min": 0.01, "max": 1000000},
            "required_fields": ["transaction_id", "user_id", "amount", "timestamp"],
            "valid_currencies": ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"],
            "valid_countries": ["US", "GB", "CA", "FR", "DE", "AU", "JP"],
            "max_velocity_1h": 50,
            "max_velocity_24h": 200,
            "max_account_age_days": 36500,  # 100 years
        }
    
    async def validate_data(
        self, 
        data_batch: List[Dict[str, Any]], 
        custom_rules: Optional[Dict[str, Any]] = None
    ) -> DataQualityMetrics:
        """Perform comprehensive data quality assessment."""
        
        rules = {**self.validation_rules, **(custom_rules or {})}
        total_records = len(data_batch)
        
        if total_records == 0:
            return DataQualityMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Initialize counters
        completeness_issues = 0
        validity_issues = 0
        consistency_issues = 0
        accuracy_issues = 0
        uniqueness_issues = 0
        timeliness_issues = 0
        
        seen_ids = set()
        
        for record in data_batch:
            try:
                # Completeness check
                missing_fields = [
                    field for field in rules["required_fields"] 
                    if field not in record or record[field] is None
                ]
                if missing_fields:
                    completeness_issues += 1
                
                # Validity checks
                if "amount" in record:
                    amount = float(record["amount"])
                    if not (rules["amount_range"]["min"] <= amount <= rules["amount_range"]["max"]):
                        validity_issues += 1
                
                if "currency" in record and record["currency"] not in rules["valid_currencies"]:
                    validity_issues += 1
                
                # Consistency checks
                if "velocity_1h" in record and "velocity_24h" in record:
                    if record["velocity_1h"] > record["velocity_24h"]:
                        consistency_issues += 1
                
                # Accuracy checks (business logic)
                if "account_age_days" in record:
                    age = record["account_age_days"]
                    if age is not None and (age < 0 or age > rules["max_account_age_days"]):
                        accuracy_issues += 1
                
                # Uniqueness check
                transaction_id = record.get("transaction_id")
                if transaction_id:
                    if transaction_id in seen_ids:
                        uniqueness_issues += 1
                    seen_ids.add(transaction_id)
                
                # Timeliness check
                if "timestamp" in record:
                    try:
                        timestamp = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
                        age_minutes = (datetime.now().replace(tzinfo=timestamp.tzinfo) - timestamp).total_seconds() / 60
                        if age_minutes > 60:  # Data older than 1 hour
                            timeliness_issues += 1
                    except:
                        timeliness_issues += 1
                        
            except Exception as e:
                logger.warning(f"Validation error for record: {e}")
                validity_issues += 1
        
        # Calculate metrics
        completeness = 1.0 - (completeness_issues / total_records)
        validity = 1.0 - (validity_issues / total_records)
        consistency = 1.0 - (consistency_issues / total_records)
        accuracy = 1.0 - (accuracy_issues / total_records)
        uniqueness = 1.0 - (uniqueness_issues / total_records)
        timeliness = 1.0 - (timeliness_issues / total_records)
        
        overall_score = np.mean([completeness, validity, consistency, accuracy, uniqueness, timeliness])
        
        metrics = DataQualityMetrics(
            completeness=completeness,
            validity=validity,
            consistency=consistency,
            accuracy=accuracy,
            uniqueness=uniqueness,
            timeliness=timeliness,
            overall_score=overall_score
        )
        
        logger.info(f"Data quality assessment completed: Overall score = {overall_score:.3f}")
        return metrics


class DriftDetector:
    """Advanced drift detection system."""
    
    def __init__(self):
        self.reference_distributions = {}
        self.feature_importance = {}
    
    async def detect_drift(
        self,
        current_data: List[Dict[str, Any]],
        reference_data: List[Dict[str, Any]],
        threshold: float = 0.05,
        method: str = "ks_test"
    ) -> DriftDetectionResult:
        """Detect data drift between current and reference datasets."""
        
        try:
            # Convert to DataFrames
            current_df = pd.DataFrame(current_data)
            reference_df = pd.DataFrame(reference_data)
            
            # Find common numeric columns
            numeric_cols = []
            for col in current_df.columns:
                if (col in reference_df.columns and 
                    pd.api.types.is_numeric_dtype(current_df[col]) and
                    pd.api.types.is_numeric_dtype(reference_df[col])):
                    numeric_cols.append(col)
            
            if not numeric_cols:
                return DriftDetectionResult(
                    has_drift=False,
                    drift_score=0.0,
                    affected_features=[],
                    drift_method=method,
                    details={"error": "No numeric columns found for drift detection"}
                )
            
            drift_scores = {}
            affected_features = []
            
            for col in numeric_cols:
                current_values = current_df[col].dropna().values
                reference_values = reference_df[col].dropna().values
                
                if len(current_values) == 0 or len(reference_values) == 0:
                    continue
                
                if method == "ks_test":
                    statistic, p_value = stats.ks_2samp(reference_values, current_values)
                    drift_score = 1 - p_value
                    
                elif method == "psi":
                    drift_score = self._calculate_psi(reference_values, current_values)
                    
                elif method == "js_divergence":
                    drift_score = self._calculate_js_divergence(reference_values, current_values)
                    
                elif method == "wasserstein":
                    drift_score = stats.wasserstein_distance(reference_values, current_values)
                    # Normalize to 0-1 range (approximate)
                    drift_score = min(drift_score / (np.std(reference_values) * 2), 1.0)
                    
                else:
                    raise ValueError(f"Unknown drift detection method: {method}")
                
                drift_scores[col] = drift_score
                
                if drift_score > threshold:
                    affected_features.append(col)
            
            overall_drift_score = np.mean(list(drift_scores.values())) if drift_scores else 0.0
            has_drift = overall_drift_score > threshold
            
            return DriftDetectionResult(
                has_drift=has_drift,
                drift_score=overall_drift_score,
                affected_features=affected_features,
                drift_method=method,
                details={
                    "feature_drift_scores": drift_scores,
                    "threshold": threshold,
                    "num_features_checked": len(numeric_cols),
                    "num_affected_features": len(affected_features)
                }
            )
            
        except Exception as e:
            logger.error(f"Drift detection error: {e}")
            return DriftDetectionResult(
                has_drift=False,
                drift_score=0.0,
                affected_features=[],
                drift_method=method,
                details={"error": str(e)}
            )
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
        """Calculate Population Stability Index (PSI)."""
        try:
            # Create buckets based on reference data
            breakpoints = np.histogram(reference, bins=buckets)[1]
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=breakpoints)
            cur_counts, _ = np.histogram(current, bins=breakpoints)
            
            # Convert to percentages and add small epsilon to avoid log(0)
            ref_percents = ref_counts / len(reference) + 1e-6
            cur_percents = cur_counts / len(current) + 1e-6
            
            # Calculate PSI
            psi = np.sum((cur_percents - ref_percents) * np.log(cur_percents / ref_percents))
            
            return abs(psi)
            
        except Exception:
            return 0.0
    
    def _calculate_js_divergence(self, reference: np.ndarray, current: np.ndarray, bins: int = 50) -> float:
        """Calculate Jensen-Shannon divergence."""
        try:
            # Create histograms
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), current.max())
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
            cur_hist, _ = np.histogram(current, bins=bin_edges, density=True)
            
            # Normalize and add epsilon
            ref_hist = ref_hist / np.sum(ref_hist) + 1e-10
            cur_hist = cur_hist / np.sum(cur_hist) + 1e-10
            
            # Calculate JS divergence
            m = 0.5 * (ref_hist + cur_hist)
            js_div = 0.5 * stats.entropy(ref_hist, m) + 0.5 * stats.entropy(cur_hist, m)
            
            return js_div
            
        except Exception:
            return 0.0


class DataIntelligenceServer(BaseMCPServer):
    """Data Intelligence MCP Server for fraud detection system."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        super().__init__("DataIntelligence", "1.0.0", host, port)
        
        self.feature_engineer = FeatureEngineer()
        self.data_validator = DataValidator()
        self.drift_detector = DriftDetector()
        
        # Data storage
        self.ingested_data = []
        self.processed_features = {}
        self.validation_history = []
        self.drift_history = []
        
        # Performance tracking
        self.processing_stats = {
            "total_ingested": 0,
            "total_validated": 0,
            "total_features_engineered": 0,
            "drift_alerts": 0
        }
    
    async def initialize(self):
        """Initialize the Data Intelligence server."""
        # Register tools
        for tool in DATA_INTELLIGENCE_TOOLS:
            if tool.name == "ingest_transaction":
                self.register_tool(tool, self.ingest_transaction)
            elif tool.name == "validate_data":
                self.register_tool(tool, self.validate_data)
            elif tool.name == "engineer_features":
                self.register_tool(tool, self.engineer_features)
            elif tool.name == "detect_drift":
                self.register_tool(tool, self.detect_drift)
        
        # Add capabilities
        self.add_capability("data_ingestion")
        self.add_capability("data_validation")
        self.add_capability("feature_engineering")
        self.add_capability("drift_detection")
        
        logger.info("Data Intelligence MCP Server initialized successfully")
    
    async def shutdown(self):
        """Clean up resources."""
        logger.info("Data Intelligence MCP Server shutting down")
    
    async def ingest_transaction(
        self, 
        transaction_data: Dict[str, Any], 
        source: str, 
        timestamp: str
    ) -> Dict[str, Any]:
        """Ingest and validate incoming transaction data."""
        try:
            # Parse timestamp
            ingestion_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            
            # Validate basic structure
            if not isinstance(transaction_data, dict):
                raise ValueError("Transaction data must be a dictionary")
            
            # Add metadata
            enriched_data = {
                **transaction_data,
                "ingestion_timestamp": ingestion_time.isoformat(),
                "source": source,
                "processing_id": f"{source}_{int(ingestion_time.timestamp())}"
            }
            
            # Store ingested data
            self.ingested_data.append(enriched_data)
            self.processing_stats["total_ingested"] += 1
            
            # Perform immediate validation
            validation_result = await self.data_validator.validate_data([transaction_data])
            
            result = {
                "status": "success",
                "transaction_id": transaction_data.get("transaction_id", "unknown"),
                "ingestion_timestamp": ingestion_time.isoformat(),
                "source": source,
                "validation_score": validation_result.overall_score,
                "data_quality": {
                    "completeness": validation_result.completeness,
                    "validity": validation_result.validity,
                    "consistency": validation_result.consistency,
                    "accuracy": validation_result.accuracy,
                    "uniqueness": validation_result.uniqueness,
                    "timeliness": validation_result.timeliness
                },
                "processing_stats": self.processing_stats.copy()
            }
            
            logger.info(f"Successfully ingested transaction from {source} with quality score {validation_result.overall_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Transaction ingestion error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "transaction_id": transaction_data.get("transaction_id", "unknown")
            }
    
    async def validate_data(
        self, 
        data_batch: List[Dict[str, Any]], 
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive data quality assessment."""
        try:
            if not data_batch:
                return {
                    "status": "error",
                    "error": "Empty data batch provided"
                }
            
            # Run validation
            metrics = await self.data_validator.validate_data(data_batch, validation_rules)
            
            # Store validation history
            validation_record = {
                "timestamp": datetime.now().isoformat(),
                "batch_size": len(data_batch),
                "metrics": metrics,
                "custom_rules": validation_rules is not None
            }
            self.validation_history.append(validation_record)
            self.processing_stats["total_validated"] += len(data_batch)
            
            # Generate recommendations
            recommendations = []
            if metrics.completeness < 0.95:
                recommendations.append("Improve data completeness - missing required fields detected")
            if metrics.validity < 0.90:
                recommendations.append("Review data validity - invalid values detected")
            if metrics.consistency < 0.95:
                recommendations.append("Check data consistency - logical inconsistencies found")
            if metrics.timeliness < 0.90:
                recommendations.append("Improve data timeliness - delayed data detected")
            
            result = {
                "status": "success",
                "batch_size": len(data_batch),
                "validation_timestamp": datetime.now().isoformat(),
                "data_quality_metrics": {
                    "completeness": metrics.completeness,
                    "validity": metrics.validity,
                    "consistency": metrics.consistency,
                    "accuracy": metrics.accuracy,
                    "uniqueness": metrics.uniqueness,
                    "timeliness": metrics.timeliness,
                    "overall_score": metrics.overall_score
                },
                "quality_grade": self._get_quality_grade(metrics.overall_score),
                "recommendations": recommendations,
                "processing_stats": self.processing_stats.copy()
            }
            
            logger.info(f"Validated batch of {len(data_batch)} records with overall score {metrics.overall_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "batch_size": len(data_batch) if data_batch else 0
            }
    
    async def engineer_features(
        self, 
        raw_data: Dict[str, Any], 
        feature_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Automatically engineer new features from raw data."""
        try:
            config = feature_config or {}
            
            # Engineer features
            features = await self.feature_engineer.engineer_features(raw_data, config)
            
            # Store processed features
            transaction_id = raw_data.get("transaction_id", "unknown")
            self.processed_features[transaction_id] = {
                "timestamp": datetime.now().isoformat(),
                "features": features,
                "config": config
            }
            self.processing_stats["total_features_engineered"] += 1
            
            # Feature importance analysis (simplified)
            feature_importance = {}
            numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
            
            if numeric_features:
                # Calculate basic statistics for feature ranking
                for feature_name, value in numeric_features.items():
                    # Simple importance based on variance from mean (placeholder)
                    feature_importance[feature_name] = abs(value) if value != 0 else 0.1
            
            result = {
                "status": "success",
                "transaction_id": transaction_id,
                "processing_timestamp": datetime.now().isoformat(),
                "features": features,
                "feature_count": len(features),
                "numeric_features": len(numeric_features),
                "categorical_features": len(features) - len(numeric_features),
                "feature_importance": feature_importance,
                "processing_stats": self.processing_stats.copy()
            }
            
            logger.info(f"Engineered {len(features)} features for transaction {transaction_id}")
            return result
            
        except Exception as e:
            logger.error(f"Feature engineering error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "transaction_id": raw_data.get("transaction_id", "unknown")
            }
    
    async def detect_drift(
        self,
        current_data: List[Dict[str, Any]],
        reference_data: List[Dict[str, Any]],
        drift_threshold: float = 0.05,
        drift_method: str = "ks_test"
    ) -> Dict[str, Any]:
        """Monitor and detect data/model drift."""
        try:
            if not current_data or not reference_data:
                return {
                    "status": "error",
                    "error": "Both current and reference data are required"
                }
            
            # Detect drift
            drift_result = await self.drift_detector.detect_drift(
                current_data, reference_data, drift_threshold, drift_method
            )
            
            # Store drift history
            drift_record = {
                "timestamp": datetime.now().isoformat(),
                "method": drift_method,
                "threshold": drift_threshold,
                "result": drift_result,
                "current_data_size": len(current_data),
                "reference_data_size": len(reference_data)
            }
            self.drift_history.append(drift_record)
            
            if drift_result.has_drift:
                self.processing_stats["drift_alerts"] += 1
            
            # Generate alert level
            alert_level = "none"
            if drift_result.has_drift:
                if drift_result.drift_score > 0.3:
                    alert_level = "critical"
                elif drift_result.drift_score > 0.15:
                    alert_level = "high"
                else:
                    alert_level = "medium"
            
            result = {
                "status": "success",
                "detection_timestamp": datetime.now().isoformat(),
                "drift_detected": drift_result.has_drift,
                "drift_score": drift_result.drift_score,
                "alert_level": alert_level,
                "affected_features": drift_result.affected_features,
                "drift_method": drift_result.drift_method,
                "threshold_used": drift_threshold,
                "details": drift_result.details,
                "recommendations": self._generate_drift_recommendations(drift_result),
                "processing_stats": self.processing_stats.copy()
            }
            
            logger.info(
                f"Drift detection completed: {'DRIFT DETECTED' if drift_result.has_drift else 'NO DRIFT'} "
                f"(score: {drift_result.drift_score:.3f}, method: {drift_method})"
            )
            return result
            
        except Exception as e:
            logger.error(f"Drift detection error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "current_data_size": len(current_data) if current_data else 0,
                "reference_data_size": len(reference_data) if reference_data else 0
            }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade."""
        if score >= 0.95:
            return "A"
        elif score >= 0.90:
            return "B"
        elif score >= 0.80:
            return "C"
        elif score >= 0.70:
            return "D"
        else:
            return "F"
    
    def _generate_drift_recommendations(self, drift_result: DriftDetectionResult) -> List[str]:
        """Generate recommendations based on drift detection results."""
        recommendations = []
        
        if not drift_result.has_drift:
            recommendations.append("No drift detected - monitoring systems are stable")
            return recommendations
        
        if drift_result.drift_score > 0.3:
            recommendations.append("CRITICAL: Significant drift detected - immediate model retraining recommended")
        elif drift_result.drift_score > 0.15:
            recommendations.append("HIGH: Notable drift detected - schedule model retraining within 24 hours")
        else:
            recommendations.append("MEDIUM: Mild drift detected - monitor closely and consider retraining")
        
        if len(drift_result.affected_features) > 0:
            recommendations.append(f"Focus on features: {', '.join(drift_result.affected_features[:5])}")
        
        recommendations.append("Update reference dataset with recent stable data")
        recommendations.append("Review data collection and preprocessing pipelines")
        
        return recommendations


async def main():
    """Main function to run the Data Intelligence MCP Server."""
    server = DataIntelligenceServer()
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await server.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())