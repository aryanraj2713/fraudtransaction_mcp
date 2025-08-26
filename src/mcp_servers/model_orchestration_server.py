import asyncio
import logging
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import uuid
import os
from pathlib import Path
import joblib
import aiofiles

# ML imports
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# OpenAI for advanced model features
import openai

from .base_server import BaseMCPServer
from schemas.mcp_tools_schema import MODEL_ORCHESTRATION_TOOLS
from schemas.transaction_schema import Transaction, FraudScore, ModelPrediction

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    model_id: str
    model_type: str
    version: str
    created_at: datetime
    last_updated: datetime
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_data_info: Dict[str, Any]
    is_active: bool = True
    deployment_status: str = ""


@dataclass
class EnsembleConfiguration:
    ensemble_id: str
    model_ids: List[str]
    weights: Dict[str, float]
    combination_method: str  # weighted_average, voting, stacking
    performance_threshold: float
    created_at: datetime
    last_updated: datetime


@dataclass
class ABTestConfiguration:
    test_id: str
    model_a_id: str
    model_b_id: str
    traffic_split: float
    start_time: datetime
    end_time: Optional[datetime]
    success_metrics: List[str]
    current_results: Dict[str, Any]
    status: str  # running, paused, completed


class ModelTrainer:
    """Advanced model training system with multiple algorithms."""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.scalers_dir = self.models_dir / "scalers"
        self.scalers_dir.mkdir(exist_ok=True)
        
    async def train_model(
        self,
        model_config: Dict[str, Any],
        training_data: List[Dict[str, Any]],
        validation_split: float = 0.2
    ) -> Tuple[str, ModelMetadata]:
        """Train a fraud detection model."""
        
        try:
            # Prepare data
            df = pd.DataFrame(training_data)
            
            # Separate features and target
            if 'is_fraud' not in df.columns:
                raise ValueError("Training data must include 'is_fraud' target column")
            
            # Feature engineering
            feature_cols = [col for col in df.columns if col not in ['is_fraud', 'transaction_id']]
            X = df[feature_cols]
            y = df['is_fraud']
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model based on type
            model_type = model_config['model_type']
            hyperparams = model_config.get('hyperparameters', {})
            
            if model_type == 'xgboost':
                model = self._train_xgboost(X_train_scaled, y_train, hyperparams)
            elif model_type == 'lightgbm':
                model = self._train_lightgbm(X_train_scaled, y_train, hyperparams)
            elif model_type == 'random_forest':
                model = self._train_random_forest(X_train_scaled, y_train, hyperparams)
            elif model_type == 'neural_network':
                model = await self._train_neural_network(X_train_scaled, y_train, hyperparams)
            elif model_type == 'ensemble':
                model = self._train_ensemble(X_train_scaled, y_train, hyperparams)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Evaluate model
            val_predictions = model.predict_proba(X_val_scaled)[:, 1]
            val_pred_binary = model.predict(X_val_scaled)
            
            performance_metrics = {
                'auc_roc': float(roc_auc_score(y_val, val_predictions)),
                'accuracy': float(np.mean(val_pred_binary == y_val)),
                'precision': float(np.sum((val_pred_binary == 1) & (y_val == 1)) / max(np.sum(val_pred_binary == 1), 1)),
                'recall': float(np.sum((val_pred_binary == 1) & (y_val == 1)) / max(np.sum(y_val == 1), 1)),
                'validation_samples': len(y_val),
                'training_samples': len(y_train)
            }
            
            # Calculate F1 score
            precision = performance_metrics['precision']
            recall = performance_metrics['recall']
            performance_metrics['f1_score'] = float(2 * (precision * recall) / max(precision + recall, 1e-8))
            
            # Generate model ID and save
            model_id = f"{model_type}_{uuid.uuid4().hex[:8]}"
            
            # Save model and scaler
            model_path = self.models_dir / f"{model_id}.pkl"
            scaler_path = self.scalers_dir / f"{model_id}_scaler.pkl"
            
            async with aiofiles.open(model_path, 'wb') as f:
                await f.write(pickle.dumps(model))
            
            async with aiofiles.open(scaler_path, 'wb') as f:
                await f.write(pickle.dumps(scaler))
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                model_type=model_type,
                version="1.0.0",
                created_at=datetime.now(),
                last_updated=datetime.now(),
                performance_metrics=performance_metrics,
                hyperparameters=hyperparams,
                training_data_info={
                    'num_samples': len(training_data),
                    'num_features': len(feature_cols),
                    'fraud_rate': float(y.mean()),
                    'feature_names': feature_cols
                },
                is_active=True,
                deployment_status="trained"
            )
            
            # Save metadata
            metadata_path = self.models_dir / f"{model_id}_metadata.json"
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(asdict(metadata), default=str))
            
            logger.info(f"Successfully trained {model_type} model {model_id} with AUC: {performance_metrics['auc_roc']:.3f}")
            return model_id, metadata
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
            raise
    
    def _train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, hyperparams: Dict[str, Any]) -> xgb.XGBClassifier:
        """Train XGBoost model."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        params = {**default_params, **hyperparams}
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        return model
    
    def _train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, hyperparams: Dict[str, Any]) -> lgb.LGBMClassifier:
        """Train LightGBM model."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
        params = {**default_params, **hyperparams}
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        return model
    
    def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, hyperparams: Dict[str, Any]) -> RandomForestClassifier:
        """Train Random Forest model."""
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        params = {**default_params, **hyperparams}
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        return model
    
    async def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray, hyperparams: Dict[str, Any]):
        """Train neural network using sklearn MLPClassifier as simplified version."""
        from sklearn.neural_network import MLPClassifier
        
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'learning_rate': 'adaptive',
            'max_iter': 500,
            'random_state': 42
        }
        params = {**default_params, **hyperparams}
        
        model = MLPClassifier(**params)
        model.fit(X_train, y_train)
        return model
    
    def _train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray, hyperparams: Dict[str, Any]) -> VotingClassifier:
        """Train ensemble model."""
        # Create base models
        xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
        lgb_model = lgb.LGBMClassifier(n_estimators=50, max_depth=4, random_state=42, verbose=-1)
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model),
                ('lgb', lgb_model)
            ],
            voting='soft'
        )
        
        ensemble.fit(X_train, y_train)
        return ensemble


class ModelPredictor:
    """High-performance model prediction system."""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        self.loaded_scalers = {}
        self.model_metadata = {}
        
    async def load_model(self, model_id: str) -> bool:
        """Load a model into memory for fast prediction."""
        try:
            if model_id in self.loaded_models:
                return True
            
            model_path = self.models_dir / f"{model_id}.pkl"
            scaler_path = self.models_dir / "scalers" / f"{model_id}_scaler.pkl"
            metadata_path = self.models_dir / f"{model_id}_metadata.json"
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load model
            async with aiofiles.open(model_path, 'rb') as f:
                model_data = await f.read()
                self.loaded_models[model_id] = pickle.loads(model_data)
            
            # Load scaler
            if scaler_path.exists():
                async with aiofiles.open(scaler_path, 'rb') as f:
                    scaler_data = await f.read()
                    self.loaded_scalers[model_id] = pickle.loads(scaler_data)
            
            # Load metadata
            if metadata_path.exists():
                async with aiofiles.open(metadata_path, 'r') as f:
                    metadata_json = await f.read()
                    self.model_metadata[model_id] = json.loads(metadata_json)
            
            logger.info(f"Successfully loaded model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return False
    
    async def predict_fraud(
        self,
        transaction: Dict[str, Any],
        model_ensemble: List[str],
        explain: bool = False
    ) -> Dict[str, Any]:
        """Generate fraud predictions using ensemble models."""
        try:
            # Prepare transaction data
            transaction_df = pd.DataFrame([transaction])
            
            # Load required models
            for model_id in model_ensemble:
                if not await self.load_model(model_id):
                    logger.warning(f"Could not load model {model_id}")
            
            predictions = {}
            explanations = {}
            
            for model_id in model_ensemble:
                if model_id not in self.loaded_models:
                    continue
                
                try:
                    model = self.loaded_models[model_id]
                    scaler = self.loaded_scalers.get(model_id)
                    metadata = self.model_metadata.get(model_id, {})
                    
                    # Get feature names from metadata
                    feature_names = metadata.get('training_data_info', {}).get('feature_names', list(transaction.keys()))
                    
                    # Prepare features
                    available_features = [f for f in feature_names if f in transaction]
                    X = transaction_df[available_features].fillna(0)
                    
                    # Scale if scaler is available
                    if scaler:
                        X_scaled = scaler.transform(X)
                    else:
                        X_scaled = X.values
                    
                    # Make prediction
                    prob_fraud = model.predict_proba(X_scaled)[:, 1][0]
                    confidence = max(prob_fraud, 1 - prob_fraud)  # Distance from 0.5
                    
                    predictions[model_id] = {
                        'probability': float(prob_fraud),
                        'confidence': float(confidence),
                        'features_used': available_features,
                        'model_type': metadata.get('model_type', 'unknown')
                    }
                    
                    # Generate explanation if requested
                    if explain:
                        explanations[model_id] = await self._generate_explanation(
                            model, X_scaled, available_features, prob_fraud
                        )
                        
                except Exception as e:
                    logger.error(f"Prediction error for model {model_id}: {e}")
                    continue
            
            if not predictions:
                raise ValueError("No models were able to generate predictions")
            
            # Ensemble prediction (simple average)
            ensemble_prob = np.mean([p['probability'] for p in predictions.values()])
            ensemble_confidence = np.mean([p['confidence'] for p in predictions.values()])
            
            # Determine recommendation
            if ensemble_prob > 0.7:
                recommendation = "decline"
            elif ensemble_prob > 0.3:
                recommendation = "review"
            else:
                recommendation = "approve"
            
            result = {
                'ensemble_prediction': {
                    'probability': float(ensemble_prob),
                    'confidence': float(ensemble_confidence),
                    'recommendation': recommendation,
                    'risk_level': self._get_risk_level(ensemble_prob)
                },
                'individual_predictions': predictions,
                'model_count': len(predictions),
                'transaction_id': transaction.get('transaction_id', 'unknown'),
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            if explain and explanations:
                result['explanations'] = explanations
            
            return result
            
        except Exception as e:
            logger.error(f"Fraud prediction error: {e}")
            raise
    
    async def _generate_explanation(
        self,
        model,
        X_scaled: np.ndarray,
        feature_names: List[str],
        probability: float
    ) -> Dict[str, Any]:
        """Generate explanation for model prediction."""
        try:
            explanation = {
                'risk_factors': [],
                'protective_factors': [],
                'confidence': 'medium'
            }
            
            # Simple feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = list(zip(feature_names, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                top_features = feature_importance[:5]
                explanation['top_features'] = [
                    {'feature': name, 'importance': float(importance)}
                    for name, importance in top_features
                ]
            
            # Rule-based explanations
            if probability > 0.7:
                explanation['risk_factors'].extend([
                    'High fraud probability detected',
                    'Multiple risk indicators present'
                ])
            elif probability > 0.3:
                explanation['risk_factors'].append('Moderate risk indicators detected')
            else:
                explanation['protective_factors'].append('Low risk profile')
            
            return explanation
            
        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
            return {'error': str(e)}
    
    def _get_risk_level(self, probability: float) -> str:
        """Convert probability to risk level."""
        if probability > 0.8:
            return "critical"
        elif probability > 0.6:
            return "high"
        elif probability > 0.4:
            return "medium"
        elif probability > 0.2:
            return "low"
        else:
            return "minimal"


class EnsembleManager:
    """Manages ensemble configurations and dynamic weighting."""
    
    def __init__(self):
        self.ensembles: Dict[str, EnsembleConfiguration] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def update_model_weights(
        self,
        performance_metrics: Dict[str, Dict[str, float]],
        feedback_data: List[Dict[str, Any]],
        learning_rate: float = 0.01
    ) -> Dict[str, float]:
        """Update ensemble model weights based on performance."""
        try:
            if not performance_metrics:
                raise ValueError("Performance metrics are required")
            
            # Calculate new weights based on performance
            new_weights = {}
            total_performance = 0
            
            # Simple performance-based weighting
            for model_id, metrics in performance_metrics.items():
                # Use AUC as primary metric, with F1 as secondary
                auc = metrics.get('auc_roc', 0.5)
                f1 = metrics.get('f1_score', 0.0)
                performance_score = 0.7 * auc + 0.3 * f1
                
                new_weights[model_id] = performance_score
                total_performance += performance_score
            
            # Normalize weights
            if total_performance > 0:
                for model_id in new_weights:
                    new_weights[model_id] /= total_performance
            
            # Apply learning rate for gradual updates
            if hasattr(self, 'current_weights'):
                for model_id in new_weights:
                    if model_id in self.current_weights:
                        old_weight = self.current_weights[model_id]
                        new_weights[model_id] = (1 - learning_rate) * old_weight + learning_rate * new_weights[model_id]
            
            self.current_weights = new_weights
            
            # Store performance history
            timestamp = datetime.now().isoformat()
            for model_id in performance_metrics:
                if model_id not in self.performance_history:
                    self.performance_history[model_id] = []
                
                self.performance_history[model_id].append({
                    'timestamp': timestamp,
                    'metrics': performance_metrics[model_id],
                    'weight': new_weights.get(model_id, 0),
                    'feedback_samples': len(feedback_data)
                })
            
            logger.info(f"Updated model weights: {new_weights}")
            return new_weights
            
        except Exception as e:
            logger.error(f"Weight update error: {e}")
            raise


class ABTestManager:
    """Manages A/B testing between model versions."""
    
    def __init__(self):
        self.active_tests: Dict[str, ABTestConfiguration] = {}
        self.test_results: Dict[str, Dict[str, Any]] = {}
    
    async def start_ab_test(
        self,
        model_a: str,
        model_b: str,
        traffic_split: float = 0.5,
        test_duration_hours: int = 24,
        success_metrics: List[str] = None
    ) -> str:
        """Start A/B test between two models."""
        try:
            test_id = f"abtest_{uuid.uuid4().hex[:8]}"
            
            if success_metrics is None:
                success_metrics = ['auc_roc', 'precision', 'recall', 'f1_score']
            
            config = ABTestConfiguration(
                test_id=test_id,
                model_a_id=model_a,
                model_b_id=model_b,
                traffic_split=traffic_split,
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(hours=test_duration_hours),
                success_metrics=success_metrics,
                current_results={
                    'model_a': {'predictions': 0, 'correct': 0, 'total_score': 0},
                    'model_b': {'predictions': 0, 'correct': 0, 'total_score': 0}
                },
                status="running"
            )
            
            self.active_tests[test_id] = config
            logger.info(f"Started A/B test {test_id} between {model_a} and {model_b}")
            
            return test_id
            
        except Exception as e:
            logger.error(f"A/B test start error: {e}")
            raise
    
    def get_model_for_prediction(self, test_id: str, user_id: str) -> str:
        """Determine which model to use for prediction in A/B test."""
        if test_id not in self.active_tests:
            raise ValueError(f"Unknown test ID: {test_id}")
        
        config = self.active_tests[test_id]
        
        # Simple hash-based assignment for consistent user experience
        user_hash = hash(user_id) % 100
        threshold = int(config.traffic_split * 100)
        
        return config.model_a_id if user_hash < threshold else config.model_b_id


class ModelOrchestrationServer(BaseMCPServer):
    """Model Orchestration MCP Server for fraud detection system."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8002):
        super().__init__("ModelOrchestration", "1.0.0", host, port)
        
        # Initialize components
        self.model_trainer = ModelTrainer()
        self.model_predictor = ModelPredictor()
        self.ensemble_manager = EnsembleManager()
        self.ab_test_manager = ABTestManager()
        
        # Model registry
        self.model_registry: Dict[str, ModelMetadata] = {}
        
        # Performance tracking
        self.orchestration_stats = {
            "models_trained": 0,
            "predictions_made": 0,
            "ensembles_created": 0,
            "ab_tests_running": 0
        }
        
        # Initialize OpenAI (if API key available)
        openai.api_key = os.getenv("OPENAI_API_KEY", "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")
    
    async def initialize(self):
        """Initialize the Model Orchestration server."""
        # Register tools
        for tool in MODEL_ORCHESTRATION_TOOLS:
            if tool.name == "train_model":
                self.register_tool(tool, self.train_model)
            elif tool.name == "predict_fraud":
                self.register_tool(tool, self.predict_fraud)
            elif tool.name == "update_model_weights":
                self.register_tool(tool, self.update_model_weights)
            elif tool.name == "a_b_test_models":
                self.register_tool(tool, self.a_b_test_models)
        
        # Add capabilities
        self.add_capability("model_training")
        self.add_capability("ensemble_prediction")
        self.add_capability("model_management")
        self.add_capability("ab_testing")
        
        logger.info("Model Orchestration MCP Server initialized successfully")
    
    async def shutdown(self):
        """Clean up resources."""
        logger.info("Model Orchestration MCP Server shutting down")
    
    async def train_model(
        self,
        model_config: Dict[str, Any],
        training_data: List[Dict[str, Any]],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train or retrain a fraud detection model."""
        try:
            if not training_data:
                raise ValueError("Training data is required")
            
            if not isinstance(model_config, dict) or 'model_type' not in model_config:
                raise ValueError("Valid model configuration with 'model_type' is required")
            
            # Train the model
            model_id, metadata = await self.model_trainer.train_model(
                model_config, training_data, validation_split
            )
            
            # Register in model registry
            self.model_registry[model_id] = metadata
            self.orchestration_stats["models_trained"] += 1
            
            result = {
                "status": "success",
                "model_id": model_id,
                "model_type": metadata.model_type,
                "training_timestamp": metadata.created_at.isoformat(),
                "performance_metrics": metadata.performance_metrics,
                "model_info": {
                    "version": metadata.version,
                    "num_training_samples": metadata.training_data_info["num_samples"],
                    "num_features": metadata.training_data_info["num_features"],
                    "fraud_rate": metadata.training_data_info["fraud_rate"]
                },
                "deployment_status": metadata.deployment_status,
                "orchestration_stats": self.orchestration_stats.copy()
            }
            
            logger.info(f"Successfully trained model {model_id} ({metadata.model_type}) with AUC: {metadata.performance_metrics['auc_roc']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Model training error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model_config": model_config.get("model_type", "unknown")
            }
    
    async def predict_fraud(
        self,
        transaction: Dict[str, Any],
        model_ensemble: List[str] = None,
        explain: bool = False
    ) -> Dict[str, Any]:
        """Generate fraud predictions using ensemble models."""
        try:
            if not transaction:
                raise ValueError("Transaction data is required")
            
            # Use default ensemble if none provided
            if not model_ensemble:
                # Get all active models
                active_models = [mid for mid, metadata in self.model_registry.items() if metadata.is_active]
                if not active_models:
                    raise ValueError("No active models available for prediction")
                model_ensemble = active_models[:3]  # Use top 3 models
            
            # Generate prediction
            prediction_result = await self.model_predictor.predict_fraud(
                transaction, model_ensemble, explain
            )
            
            self.orchestration_stats["predictions_made"] += 1
            
            # Enhance result with orchestration metadata
            result = {
                **prediction_result,
                "orchestration_metadata": {
                    "ensemble_size": len(model_ensemble),
                    "models_used": model_ensemble,
                    "prediction_latency_ms": 50,  # Placeholder - would measure actual latency
                    "orchestration_stats": self.orchestration_stats.copy()
                }
            }
            
            logger.info(
                f"Generated fraud prediction for transaction {transaction.get('transaction_id', 'unknown')}: "
                f"probability={prediction_result['ensemble_prediction']['probability']:.3f}, "
                f"recommendation={prediction_result['ensemble_prediction']['recommendation']}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Fraud prediction error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "transaction_id": transaction.get("transaction_id", "unknown")
            }
    
    async def update_model_weights(
        self,
        performance_metrics: Dict[str, Dict[str, float]],
        feedback_data: List[Dict[str, Any]] = None,
        learning_rate: float = 0.01
    ) -> Dict[str, Any]:
        """Update ensemble model weights based on performance."""
        try:
            if not performance_metrics:
                raise ValueError("Performance metrics are required")
            
            if feedback_data is None:
                feedback_data = []
            
            # Update weights
            new_weights = await self.ensemble_manager.update_model_weights(
                performance_metrics, feedback_data, learning_rate
            )
            
            # Update model metadata with new performance
            for model_id, metrics in performance_metrics.items():
                if model_id in self.model_registry:
                    self.model_registry[model_id].performance_metrics.update(metrics)
                    self.model_registry[model_id].last_updated = datetime.now()
            
            result = {
                "status": "success",
                "updated_weights": new_weights,
                "models_updated": list(performance_metrics.keys()),
                "learning_rate": learning_rate,
                "feedback_samples": len(feedback_data),
                "update_timestamp": datetime.now().isoformat(),
                "performance_summary": {
                    model_id: {
                        "auc_roc": metrics.get("auc_roc", 0.0),
                        "f1_score": metrics.get("f1_score", 0.0),
                        "new_weight": new_weights.get(model_id, 0.0)
                    }
                    for model_id, metrics in performance_metrics.items()
                },
                "orchestration_stats": self.orchestration_stats.copy()
            }
            
            logger.info(f"Updated weights for {len(new_weights)} models based on performance feedback")
            return result
            
        except Exception as e:
            logger.error(f"Weight update error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "models_count": len(performance_metrics) if performance_metrics else 0
            }
    
    async def a_b_test_models(
        self,
        model_a: str,
        model_b: str,
        traffic_split: float = 0.5,
        test_duration_hours: int = 24,
        success_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Conduct A/B testing between different model versions."""
        try:
            # Validate models exist
            if model_a not in self.model_registry:
                raise ValueError(f"Model A not found: {model_a}")
            if model_b not in self.model_registry:
                raise ValueError(f"Model B not found: {model_b}")
            
            # Start A/B test
            test_id = await self.ab_test_manager.start_ab_test(
                model_a, model_b, traffic_split, test_duration_hours, success_metrics
            )
            
            self.orchestration_stats["ab_tests_running"] += 1
            
            result = {
                "status": "success",
                "test_id": test_id,
                "model_a": {
                    "id": model_a,
                    "type": self.model_registry[model_a].model_type,
                    "performance": self.model_registry[model_a].performance_metrics
                },
                "model_b": {
                    "id": model_b,
                    "type": self.model_registry[model_b].model_type,
                    "performance": self.model_registry[model_b].performance_metrics
                },
                "test_configuration": {
                    "traffic_split": traffic_split,
                    "duration_hours": test_duration_hours,
                    "success_metrics": success_metrics or ['auc_roc', 'precision', 'recall', 'f1_score'],
                    "start_time": datetime.now().isoformat(),
                    "expected_end_time": (datetime.now() + timedelta(hours=test_duration_hours)).isoformat()
                },
                "orchestration_stats": self.orchestration_stats.copy()
            }
            
            logger.info(f"Started A/B test {test_id} between models {model_a} and {model_b}")
            return result
            
        except Exception as e:
            logger.error(f"A/B test error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "model_a": model_a,
                "model_b": model_b
            }


async def main():
    """Main function to run the Model Orchestration MCP Server."""
    server = ModelOrchestrationServer()
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await server.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())