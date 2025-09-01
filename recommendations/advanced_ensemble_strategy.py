"""
Advanced Ensemble Strategy for MCP Fraud Detection System

This module demonstrates advanced ensemble techniques specifically designed
for fraud detection with imbalanced datasets.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    IsolationForest, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours

class FraudDetectionEnsemble:
    """Advanced ensemble system optimized for fraud detection."""
    
    def __init__(self, fraud_threshold: float = 0.5):
        self.fraud_threshold = fraud_threshold
        self.models = {}
        self.model_weights = {}
        self.calibrators = {}
        self.is_trained = False
        
    def create_fraud_optimized_ensemble(self) -> Dict[str, Any]:
        """Create an ensemble specifically optimized for fraud detection."""
        
        # 1. Gradient Boosting Models (High precision for fraud)
        xgb_fraud = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=10,  # Handle class imbalance
            random_state=42,
            eval_metric='aucpr'  # Optimize for precision-recall
        )
        
        lgb_fraud = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            objective='binary',
            metric='auc'
        )
        
        # 2. Random Forest with Balanced Sampling
        rf_balanced = BalancedBaggingClassifier(
            base_estimator=RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            n_estimators=10,
            sampling_strategy='auto',
            random_state=42
        )
        
        # 3. Anomaly Detection Models
        isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,  # Expected fraud rate
            random_state=42
        )
        
        # 4. Neural Network for Complex Patterns
        nn_classifier = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        # 5. Logistic Regression for Baseline
        lr_calibrated = LogisticRegression(
            class_weight='balanced',
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        
        ensemble_config = {
            'xgboost_fraud': {
                'model': xgb_fraud,
                'weight': 0.25,
                'type': 'supervised',
                'specialization': 'gradient_boosting'
            },
            'lightgbm_fraud': {
                'model': lgb_fraud,
                'weight': 0.25,
                'type': 'supervised',
                'specialization': 'gradient_boosting'
            },
            'random_forest_balanced': {
                'model': rf_balanced,
                'weight': 0.20,
                'type': 'supervised',
                'specialization': 'ensemble_bagging'
            },
            'neural_network': {
                'model': nn_classifier,
                'weight': 0.15,
                'type': 'supervised',
                'specialization': 'deep_learning'
            },
            'logistic_regression': {
                'model': lr_calibrated,
                'weight': 0.10,
                'type': 'supervised',
                'specialization': 'linear'
            },
            'isolation_forest': {
                'model': isolation_forest,
                'weight': 0.05,
                'type': 'unsupervised',
                'specialization': 'anomaly_detection'
            }
        }
        
        return ensemble_config
    
    def train_with_imbalanced_techniques(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train ensemble with techniques for imbalanced datasets."""
        
        training_results = {}
        
        # 1. Apply SMOTE for oversampling
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # 2. Apply ADASYN for adaptive oversampling
        adasyn = ADASYN(random_state=42, n_neighbors=3)
        try:
            X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)
        except ValueError:
            # Fallback if not enough minority samples
            X_train_adasyn, y_train_adasyn = X_train_smote, y_train_smote
        
        # 3. Apply Edited Nearest Neighbours for undersampling
        enn = EditedNearestNeighbours()
        X_train_enn, y_train_enn = enn.fit_resample(X_train, y_train)
        
        # Create ensemble configuration
        ensemble_config = self.create_fraud_optimized_ensemble()
        
        # Train each model with appropriate sampling technique
        for model_name, config in ensemble_config.items():
            model = config['model']
            
            try:
                if model_name in ['xgboost_fraud', 'lightgbm_fraud']:
                    # Use SMOTE for gradient boosting
                    model.fit(X_train_smote, y_train_smote)
                elif model_name == 'random_forest_balanced':
                    # Balanced bagging handles imbalance internally
                    model.fit(X_train, y_train)
                elif model_name == 'neural_network':
                    # Use ADASYN for neural network
                    model.fit(X_train_adasyn, y_train_adasyn)
                elif model_name == 'isolation_forest':
                    # Unsupervised - use original data
                    model.fit(X_train)
                else:
                    # Use ENN for other models
                    model.fit(X_train_enn, y_train_enn)
                
                self.models[model_name] = model
                self.model_weights[model_name] = config['weight']
                
                training_results[model_name] = {
                    'status': 'success',
                    'specialization': config['specialization']
                }
                
            except Exception as e:
                training_results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Calibrate probability outputs
        self._calibrate_models(X_train, y_train)
        
        self.is_trained = True
        return training_results
    
    def _calibrate_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Calibrate model probability outputs for better confidence estimates."""
        
        for model_name, model in self.models.items():
            if model_name == 'isolation_forest':
                continue  # Skip anomaly detection models
            
            try:
                # Use Platt scaling for probability calibration
                calibrator = CalibratedClassifierCV(
                    base_estimator=model,
                    method='sigmoid',
                    cv=3
                )
                calibrator.fit(X_train, y_train)
                self.calibrators[model_name] = calibrator
                
            except Exception as e:
                print(f"Calibration failed for {model_name}: {e}")
                self.calibrators[model_name] = model
    
    def predict_fraud_probability(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Predict fraud probability using ensemble."""
        
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        individual_predictions = {}
        weighted_probabilities = []
        weights = []
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'isolation_forest':
                    # Anomaly detection: convert anomaly scores to probabilities
                    anomaly_scores = model.decision_function(X)
                    # Convert to probability-like scores (0-1 range)
                    prob = 1 / (1 + np.exp(-anomaly_scores))
                    individual_predictions[model_name] = prob
                else:
                    # Use calibrated model if available
                    predictor = self.calibrators.get(model_name, model)
                    
                    if hasattr(predictor, 'predict_proba'):
                        prob = predictor.predict_proba(X)[:, 1]
                    else:
                        # Fallback for models without predict_proba
                        prob = predictor.decision_function(X)
                        prob = 1 / (1 + np.exp(-prob))  # Sigmoid transformation
                    
                    individual_predictions[model_name] = prob
                
                # Add to weighted ensemble
                weight = self.model_weights[model_name]
                weighted_probabilities.append(prob * weight)
                weights.append(weight)
                
            except Exception as e:
                print(f"Prediction failed for {model_name}: {e}")
                continue
        
        # Calculate ensemble probability
        if weighted_probabilities:
            ensemble_prob = np.sum(weighted_probabilities, axis=0) / np.sum(weights)
        else:
            ensemble_prob = np.zeros(len(X))
        
        # Calculate confidence based on agreement between models
        if len(individual_predictions) > 1:
            pred_matrix = np.column_stack(list(individual_predictions.values()))
            confidence = 1 - np.std(pred_matrix, axis=1)  # Higher agreement = higher confidence
        else:
            confidence = np.ones(len(X)) * 0.5  # Default confidence
        
        return {
            'ensemble_probability': ensemble_prob,
            'individual_predictions': individual_predictions,
            'confidence': confidence,
            'decision': (ensemble_prob > self.fraud_threshold).astype(int)
        }
    
    def optimize_threshold(
        self, 
        X_val: pd.DataFrame, 
        y_val: pd.Series,
        cost_matrix: Dict[str, float] = None
    ) -> float:
        """Optimize decision threshold based on business costs."""
        
        if cost_matrix is None:
            cost_matrix = {
                'false_positive_cost': 10,    # Cost of declining legitimate transaction
                'false_negative_cost': 100,   # Cost of approving fraudulent transaction
                'true_positive_benefit': 90,  # Benefit of catching fraud
                'true_negative_benefit': 1    # Benefit of approving legitimate
            }
        
        predictions = self.predict_fraud_probability(X_val)
        probabilities = predictions['ensemble_probability']
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_score = float('-inf')
        
        threshold_results = []
        
        for threshold in thresholds:
            y_pred = (probabilities > threshold).astype(int)
            
            # Calculate confusion matrix
            tp = np.sum((y_pred == 1) & (y_val == 1))
            tn = np.sum((y_pred == 0) & (y_val == 0))
            fp = np.sum((y_pred == 1) & (y_val == 0))
            fn = np.sum((y_pred == 0) & (y_val == 1))
            
            # Calculate business value
            business_value = (
                tp * cost_matrix['true_positive_benefit'] +
                tn * cost_matrix['true_negative_benefit'] -
                fp * cost_matrix['false_positive_cost'] -
                fn * cost_matrix['false_negative_cost']
            )
            
            threshold_results.append({
                'threshold': threshold,
                'business_value': business_value,
                'precision': tp / max(tp + fp, 1),
                'recall': tp / max(tp + fn, 1),
                'f1_score': 2 * tp / max(2 * tp + fp + fn, 1)
            })
            
            if business_value > best_score:
                best_score = business_value
                best_threshold = threshold
        
        self.fraud_threshold = best_threshold
        
        return {
            'optimal_threshold': best_threshold,
            'business_value': best_score,
            'threshold_analysis': threshold_results
        }
    
    def update_model_weights_dynamically(
        self, 
        X_val: pd.DataFrame, 
        y_val: pd.Series,
        performance_window: int = 1000
    ) -> Dict[str, float]:
        """Dynamically update model weights based on recent performance."""
        
        individual_predictions = {}
        
        # Get individual model predictions
        for model_name, model in self.models.items():
            try:
                if model_name == 'isolation_forest':
                    anomaly_scores = model.decision_function(X_val)
                    prob = 1 / (1 + np.exp(-anomaly_scores))
                else:
                    predictor = self.calibrators.get(model_name, model)
                    if hasattr(predictor, 'predict_proba'):
                        prob = predictor.predict_proba(X_val)[:, 1]
                    else:
                        prob = predictor.decision_function(X_val)
                        prob = 1 / (1 + np.exp(-prob))
                
                individual_predictions[model_name] = prob
                
            except Exception as e:
                print(f"Failed to get predictions for {model_name}: {e}")
                continue
        
        # Calculate performance metrics for each model
        new_weights = {}
        
        for model_name, predictions in individual_predictions.items():
            try:
                # Calculate AUC-PR (better for imbalanced datasets)
                precision, recall, _ = precision_recall_curve(y_val, predictions)
                auc_pr = np.trapz(recall, precision)
                
                # Calculate AUC-ROC
                auc_roc = roc_auc_score(y_val, predictions)
                
                # Combine metrics with emphasis on precision-recall
                performance_score = 0.7 * auc_pr + 0.3 * auc_roc
                
                new_weights[model_name] = max(performance_score, 0.01)  # Minimum weight
                
            except Exception as e:
                print(f"Failed to calculate performance for {model_name}: {e}")
                new_weights[model_name] = self.model_weights.get(model_name, 0.1)
        
        # Normalize weights
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v / total_weight for k, v in new_weights.items()}
            self.model_weights.update(new_weights)
        
        return new_weights
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from ensemble models."""
        
        feature_importance = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_[0])
                else:
                    continue
                
                feature_importance[model_name] = {
                    f'feature_{i}': float(imp) for i, imp in enumerate(importance)
                }
                
            except Exception as e:
                print(f"Failed to get feature importance for {model_name}: {e}")
                continue
        
        return feature_importance
