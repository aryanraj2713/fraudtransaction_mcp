import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pickle
import uuid
from mcp.server import Server
from mcp.types import Tool, TextContent

from ..schemas.transaction_schema import ModelUpdateSchema, UpdateResultSchema
from ..utils.config import config

logger = logging.getLogger(__name__)

class ModelOrchestrationServer:
    def __init__(self):
        self.server = Server("model-orchestration-server")
        self.models: Dict[str, Dict[str, Any]] = {}
        self.ensemble_config: Dict[str, Any] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        self.active_experiments: Dict[str, Dict[str, Any]] = {}
        self.training_data: List[Dict[str, Any]] = []
        self.model_versions: Dict[str, List[str]] = {}
        self.setup_tools()
        self._initialize_default_models()
        
    def setup_tools(self):
        """Register MCP tools for model orchestration operations"""
        
        @self.server.tool()
        async def train_model(model_config: Dict[str, Any]) -> Dict[str, Any]:
            """Train a new fraud detection model"""
            try:
                model_id = model_config.get("model_id", f"model_{uuid.uuid4().hex[:8]}")
                model_type = model_config.get("model_type", "random_forest")
                parameters = model_config.get("parameters", {})
                
                # Train the model
                training_result = await self._train_model(model_id, model_type, parameters)
                
                logger.info(f"Trained model {model_id} with performance: {training_result['performance']}")
                
                return {
                    "status": "success",
                    "model_id": model_id,
                    "model_type": model_type,
                    "performance": training_result["performance"],
                    "training_time": training_result["training_time"],
                    "version": training_result["version"]
                }
                
            except Exception as e:
                logger.error(f"Model training failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e),
                    "model_id": model_config.get("model_id", "unknown")
                }
        
        @self.server.tool()
        async def evaluate_ensemble(ensemble_config: Dict[str, Any]) -> Dict[str, Any]:
            """Evaluate ensemble model performance"""
            try:
                model_ids = ensemble_config.get("model_ids", [])
                weights = ensemble_config.get("weights", None)
                
                if not model_ids:
                    model_ids = list(self.models.keys())
                
                evaluation_result = await self._evaluate_ensemble(model_ids, weights)
                
                return {
                    "status": "success",
                    "ensemble_performance": evaluation_result["performance"],
                    "model_contributions": evaluation_result["contributions"],
                    "recommended_weights": evaluation_result["optimal_weights"]
                }
                
            except Exception as e:
                logger.error(f"Ensemble evaluation failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def select_best_model(criteria: Dict[str, Any]) -> Dict[str, Any]:
            """Select the best performing model based on criteria"""
            try:
                metric = criteria.get("metric", "f1_score")
                min_precision = criteria.get("min_precision", 0.8)
                min_recall = criteria.get("min_recall", 0.7)
                
                best_model = await self._select_best_model(metric, min_precision, min_recall)
                
                return {
                    "status": "success",
                    "best_model_id": best_model["model_id"],
                    "performance": best_model["performance"],
                    "selection_reason": best_model["reason"]
                }
                
            except Exception as e:
                logger.error(f"Model selection failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def deploy_model(deployment_config: Dict[str, Any]) -> Dict[str, Any]:
            """Deploy a model for production use"""
            try:
                model_id = deployment_config["model_id"]
                deployment_strategy = deployment_config.get("strategy", "replace")
                
                deployment_result = await self._deploy_model(model_id, deployment_strategy)
                
                return {
                    "status": "success",
                    "model_id": model_id,
                    "deployment_strategy": deployment_strategy,
                    "deployment_timestamp": deployment_result["timestamp"],
                    "previous_model": deployment_result.get("previous_model"),
                    "rollback_available": deployment_result["rollback_available"]
                }
                
            except Exception as e:
                logger.error(f"Model deployment failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e),
                    "model_id": deployment_config.get("model_id", "unknown")
                }
    
    def _initialize_default_models(self):
        """Initialize default model configurations"""
        default_models = {
            "random_forest": {
                "class": RandomForestClassifier,
                "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42}
            },
            "gradient_boosting": {
                "class": GradientBoostingClassifier,
                "params": {"n_estimators": 100, "max_depth": 6, "random_state": 42}
            },
            "logistic_regression": {
                "class": LogisticRegression,
                "params": {"random_state": 42, "max_iter": 1000}
            },
            "neural_network": {
                "class": MLPClassifier,
                "params": {"hidden_layer_sizes": (100, 50), "random_state": 42, "max_iter": 500}
            }
        }
        
        self.default_model_configs = default_models
    
    async def _train_model(self, model_id: str, model_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Train a fraud detection model"""
        start_time = datetime.utcnow()
        
        # Get model class and default parameters
        if model_type not in self.default_model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_config = self.default_model_configs[model_type]
        model_params = {**model_config["params"], **parameters}
        
        # Create model instance
        model = model_config["class"](**model_params)
        
        # Generate synthetic training data if none available
        if not self.training_data:
            await self._generate_synthetic_training_data()
        
        # Prepare training data
        X, y = await self._prepare_training_data()
        
        if len(X) < 10:  # Minimum samples needed
            raise ValueError("Insufficient training data")
        
        # Train model
        model.fit(X, y)
        
        # Evaluate model performance
        performance = await self._evaluate_model_performance(model, X, y)
        
        # Create model version
        version = f"v{len(self.model_versions.get(model_id, [])) + 1}"
        
        # Store model
        self.models[model_id] = {
            "model": model,
            "model_type": model_type,
            "parameters": model_params,
            "version": version,
            "created_at": start_time.isoformat(),
            "training_samples": len(X)
        }
        
        # Store performance metrics
        self.model_performance[model_id] = performance
        
        # Update version history
        if model_id not in self.model_versions:
            self.model_versions[model_id] = []
        self.model_versions[model_id].append(version)
        
        training_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "performance": performance,
            "training_time": training_time,
            "version": version
        }
    
    async def _generate_synthetic_training_data(self):
        """Generate synthetic training data for model training"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic transaction features
        for i in range(n_samples):
            # Normal transactions (80%)
            is_fraud = np.random.random() < 0.2
            
            if is_fraud:
                # Fraudulent transaction patterns
                amount = np.random.exponential(500) + 1000  # Higher amounts
                hour = np.random.choice([2, 3, 4, 23, 0, 1])  # Unusual hours
                velocity = np.random.poisson(5)  # Higher velocity
                is_weekend = np.random.choice([True, False], p=[0.7, 0.3])  # More weekend fraud
            else:
                # Normal transaction patterns
                amount = np.random.exponential(100) + 10  # Lower amounts
                hour = np.random.choice(range(6, 23))  # Business hours
                velocity = np.random.poisson(1)  # Lower velocity
                is_weekend = np.random.choice([True, False], p=[0.3, 0.7])  # Less weekend activity
            
            transaction_data = {
                "transaction_id": f"synth_{i}",
                "amount": amount,
                "hour_of_day": hour,
                "day_of_week": np.random.randint(0, 7),
                "is_weekend": is_weekend,
                "velocity_score": min(1.0, velocity / 10),
                "amount_log": np.log1p(amount),
                "is_round_amount": amount % 1 == 0,
                "has_location": np.random.choice([True, False], p=[0.8, 0.2]),
                "is_mobile": np.random.choice([True, False], p=[0.6, 0.4]),
                "is_fraud": is_fraud
            }
            
            self.training_data.append(transaction_data)
    
    async def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for model training"""
        if not self.training_data:
            raise ValueError("No training data available")
        
        # Extract features and labels
        feature_columns = [
            "amount", "hour_of_day", "day_of_week", "velocity_score", 
            "amount_log", "is_round_amount", "has_location", "is_mobile"
        ]
        
        X = []
        y = []
        
        for transaction in self.training_data:
            features = []
            for col in feature_columns:
                value = transaction.get(col, 0)
                if isinstance(value, bool):
                    value = int(value)
                features.append(float(value))
            
            X.append(features)
            y.append(int(transaction.get("is_fraud", False)))
        
        return np.array(X), np.array(y)
    
    async def _evaluate_model_performance(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance using cross-validation"""
        cv = StratifiedKFold(n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=42)
        
        # Cross-validation scores
        precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision', n_jobs=-1)
        recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall', n_jobs=-1)
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
        roc_auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        return {
            "precision": float(np.mean(precision_scores)),
            "recall": float(np.mean(recall_scores)),
            "f1_score": float(np.mean(f1_scores)),
            "roc_auc": float(np.mean(roc_auc_scores)),
            "precision_std": float(np.std(precision_scores)),
            "recall_std": float(np.std(recall_scores)),
            "f1_std": float(np.std(f1_scores)),
            "roc_auc_std": float(np.std(roc_auc_scores))
        }
    
    async def _evaluate_ensemble(self, model_ids: List[str], weights: Optional[List[float]]) -> Dict[str, Any]:
        """Evaluate ensemble model performance"""
        if not model_ids:
            raise ValueError("No models specified for ensemble")
        
        # Validate all models exist
        missing_models = [mid for mid in model_ids if mid not in self.models]
        if missing_models:
            raise ValueError(f"Models not found: {missing_models}")
        
        # Prepare test data
        X, y = await self._prepare_training_data()
        
        # Get predictions from each model
        predictions = {}
        for model_id in model_ids:
            model = self.models[model_id]["model"]
            pred_proba = model.predict_proba(X)[:, 1]  # Fraud probability
            predictions[model_id] = pred_proba
        
        # If no weights provided, use equal weights
        if weights is None:
            weights = [1.0 / len(model_ids)] * len(model_ids)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate ensemble predictions
        ensemble_pred_proba = np.zeros(len(X))
        for i, model_id in enumerate(model_ids):
            ensemble_pred_proba += weights[i] * predictions[model_id]
        
        # Convert probabilities to binary predictions
        ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        
        # Calculate performance metrics
        precision = precision_score(y, ensemble_pred)
        recall = recall_score(y, ensemble_pred)
        f1 = f1_score(y, ensemble_pred)
        roc_auc = roc_auc_score(y, ensemble_pred_proba)
        
        # Calculate individual model contributions
        contributions = {}
        for i, model_id in enumerate(model_ids):
            contributions[model_id] = {
                "weight": weights[i],
                "individual_performance": self.model_performance.get(model_id, {}),
                "contribution_score": weights[i] * self.model_performance.get(model_id, {}).get("f1_score", 0)
            }
        
        # Optimize weights (simple grid search)
        optimal_weights = await self._optimize_ensemble_weights(model_ids, predictions, y)
        
        return {
            "performance": {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc)
            },
            "contributions": contributions,
            "optimal_weights": optimal_weights
        }
    
    async def _optimize_ensemble_weights(self, model_ids: List[str], predictions: Dict[str, np.ndarray], y: np.ndarray) -> List[float]:
        """Optimize ensemble weights using grid search"""
        best_weights = None
        best_score = 0
        
        # Simple grid search over weight combinations
        n_models = len(model_ids)
        if n_models == 2:
            weight_options = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for w1 in weight_options:
                w2 = 1.0 - w1
                weights = [w1, w2]
                score = await self._evaluate_weight_combination(model_ids, predictions, y, weights)
                if score > best_score:
                    best_score = score
                    best_weights = weights
        else:
            # For more than 2 models, use equal weights as baseline
            weights = [1.0 / n_models] * n_models
            best_weights = weights
        
        return best_weights or [1.0 / n_models] * n_models
    
    async def _evaluate_weight_combination(self, model_ids: List[str], predictions: Dict[str, np.ndarray], y: np.ndarray, weights: List[float]) -> float:
        """Evaluate a specific weight combination"""
        ensemble_pred_proba = np.zeros(len(y))
        for i, model_id in enumerate(model_ids):
            ensemble_pred_proba += weights[i] * predictions[model_id]
        
        ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        return f1_score(y, ensemble_pred)
    
    async def _select_best_model(self, metric: str, min_precision: float, min_recall: float) -> Dict[str, Any]:
        """Select the best model based on specified criteria"""
        if not self.model_performance:
            raise ValueError("No model performance data available")
        
        eligible_models = []
        
        for model_id, performance in self.model_performance.items():
            precision = performance.get("precision", 0)
            recall = performance.get("recall", 0)
            
            if precision >= min_precision and recall >= min_recall:
                eligible_models.append((model_id, performance))
        
        if not eligible_models:
            raise ValueError(f"No models meet criteria: precision >= {min_precision}, recall >= {min_recall}")
        
        # Select best model based on metric
        best_model_id = None
        best_score = -1
        
        for model_id, performance in eligible_models:
            score = performance.get(metric, 0)
            if score > best_score:
                best_score = score
                best_model_id = model_id
        
        best_performance = self.model_performance[best_model_id]
        
        return {
            "model_id": best_model_id,
            "performance": best_performance,
            "reason": f"Best {metric} score: {best_score:.3f}"
        }
    
    async def _deploy_model(self, model_id: str, strategy: str) -> Dict[str, Any]:
        """Deploy a model for production use"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        deployment_timestamp = datetime.utcnow().isoformat()
        
        # Get current production model (if any)
        current_production = self.ensemble_config.get("production_model")
        
        # Update ensemble configuration
        if strategy == "replace":
            previous_model = current_production
            self.ensemble_config["production_model"] = model_id
            self.ensemble_config["deployment_timestamp"] = deployment_timestamp
            rollback_available = previous_model is not None
        elif strategy == "canary":
            # Implement canary deployment
            self.ensemble_config["canary_model"] = model_id
            self.ensemble_config["canary_traffic_percentage"] = 10
            rollback_available = True
        else:
            raise ValueError(f"Unknown deployment strategy: {strategy}")
        
        return {
            "timestamp": deployment_timestamp,
            "previous_model": current_production,
            "rollback_available": rollback_available
        }
    
    def get_server(self) -> Server:
        """Get the MCP server instance"""
        return self.server
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current server metrics"""
        return {
            "total_models": len(self.models),
            "model_performance": self.model_performance,
            "ensemble_config": self.ensemble_config,
            "active_experiments": len(self.active_experiments),
            "training_data_size": len(self.training_data)
        }
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific model"""
        return self.models.get(model_id)
    
    def predict(self, model_id: str, features: List[float]) -> Dict[str, Any]:
        """Make prediction using a specific model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]["model"]
        features_array = np.array([features])
        
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        return {
            "prediction": int(prediction),
            "fraud_probability": float(probability[1]),
            "model_id": model_id,
            "model_version": self.models[model_id]["version"]
        }