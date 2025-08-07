import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
import uuid
from mcp.server import Server
from mcp.types import Tool, TextContent
from collections import defaultdict, deque

from ..utils.config import config

logger = logging.getLogger(__name__)

class MonitoringResponseServer:
    def __init__(self):
        self.server = Server("monitoring-response-server")
        self.performance_metrics = defaultdict(deque)
        self.alert_history: List[Dict[str, Any]] = []
        self.active_incidents: List[Dict[str, Any]] = []
        self.retraining_triggers: List[Dict[str, Any]] = []
        self.business_impact_data: Dict[str, Any] = {}
        self.metric_thresholds = {
            "processing_time_ms": {"warning": 50, "critical": 100},
            "fraud_score_accuracy": {"warning": 0.85, "critical": 0.80},
            "false_positive_rate": {"warning": 0.05, "critical": 0.10},
            "system_availability": {"warning": 0.99, "critical": 0.95}
        }
        self.setup_tools()
        
    def setup_tools(self):
        """Register MCP tools for monitoring and response operations"""
        
        @self.server.tool()
        async def monitor_performance(metrics_data: Dict[str, Any]) -> Dict[str, Any]:
            """Monitor system performance and detect anomalies"""
            try:
                timestamp = datetime.utcnow()
                
                # Process incoming metrics
                processed_metrics = await self._process_performance_metrics(metrics_data, timestamp)
                
                # Detect anomalies
                anomalies = await self._detect_performance_anomalies(processed_metrics)
                
                # Generate alerts if needed
                alerts = await self._generate_performance_alerts(anomalies, processed_metrics)
                
                # Update metric history
                self._update_metric_history(processed_metrics, timestamp)
                
                return {
                    "status": "success",
                    "timestamp": timestamp.isoformat(),
                    "processed_metrics": processed_metrics,
                    "anomalies_detected": len(anomalies),
                    "alerts_generated": len(alerts),
                    "system_health": await self._calculate_system_health()
                }
                
            except Exception as e:
                logger.error(f"Performance monitoring failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def trigger_retraining(trigger_data: Dict[str, Any]) -> Dict[str, Any]:
            """Trigger model retraining based on performance degradation"""
            try:
                trigger_reason = trigger_data.get("reason", "performance_degradation")
                model_ids = trigger_data.get("model_ids", [])
                priority = trigger_data.get("priority", "medium")
                
                retraining_plan = await self._create_retraining_plan(trigger_reason, model_ids, priority)
                
                # Add to retraining queue
                self.retraining_triggers.append(retraining_plan)
                
                return {
                    "status": "success",
                    "retraining_id": retraining_plan["retraining_id"],
                    "estimated_completion": retraining_plan["estimated_completion"],
                    "affected_models": retraining_plan["affected_models"],
                    "priority": priority
                }
                
            except Exception as e:
                logger.error(f"Retraining trigger failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def analyze_impact(impact_data: Dict[str, Any]) -> Dict[str, Any]:
            """Analyze business impact of fraud detection decisions"""
            try:
                time_window = impact_data.get("time_window_hours", 24)
                analysis_type = impact_data.get("analysis_type", "comprehensive")
                
                impact_analysis = await self._analyze_business_impact(time_window, analysis_type)
                
                # Update business impact data
                self.business_impact_data.update({
                    "last_analysis": datetime.utcnow().isoformat(),
                    "current_impact": impact_analysis
                })
                
                return {
                    "status": "success",
                    "impact_analysis": impact_analysis,
                    "recommendations": impact_analysis["recommendations"],
                    "financial_impact": impact_analysis["financial_impact"]
                }
                
            except Exception as e:
                logger.error(f"Impact analysis failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
        
        @self.server.tool()
        async def generate_alerts(alert_config: Dict[str, Any]) -> Dict[str, Any]:
            """Generate and manage system alerts"""
            try:
                alert_type = alert_config.get("alert_type", "performance")
                severity = alert_config.get("severity", "medium")
                conditions = alert_config.get("conditions", {})
                
                alerts = await self._generate_system_alerts(alert_type, severity, conditions)
                
                # Process and store alerts
                processed_alerts = []
                for alert in alerts:
                    alert["id"] = str(uuid.uuid4())
                    alert["created_at"] = datetime.utcnow().isoformat()
                    self.alert_history.append(alert)
                    processed_alerts.append(alert)
                
                return {
                    "status": "success",
                    "alerts_generated": len(processed_alerts),
                    "alerts": processed_alerts,
                    "total_active_alerts": len([a for a in self.alert_history 
                                              if a.get("status") == "active"])
                }
                
            except Exception as e:
                logger.error(f"Alert generation failed: {str(e)}")
                return {
                    "status": "error",
                    "error": str(e)
                }
    
    async def _process_performance_metrics(self, metrics_data: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
        """Process and normalize performance metrics"""
        processed = {}
        
        # Processing time metrics
        if "processing_times" in metrics_data:
            times = metrics_data["processing_times"]
            processed["avg_processing_time_ms"] = np.mean(times)
            processed["p95_processing_time_ms"] = np.percentile(times, 95)
            processed["p99_processing_time_ms"] = np.percentile(times, 99)
            processed["max_processing_time_ms"] = np.max(times)
        
        # Accuracy metrics
        if "accuracy_data" in metrics_data:
            accuracy_data = metrics_data["accuracy_data"]
            processed["fraud_detection_accuracy"] = accuracy_data.get("accuracy", 0.0)
            processed["precision"] = accuracy_data.get("precision", 0.0)
            processed["recall"] = accuracy_data.get("recall", 0.0)
            processed["f1_score"] = accuracy_data.get("f1_score", 0.0)
            processed["false_positive_rate"] = accuracy_data.get("false_positive_rate", 0.0)
            processed["false_negative_rate"] = accuracy_data.get("false_negative_rate", 0.0)
        
        # System metrics
        if "system_metrics" in metrics_data:
            system = metrics_data["system_metrics"]
            processed["cpu_usage"] = system.get("cpu_usage", 0.0)
            processed["memory_usage"] = system.get("memory_usage", 0.0)
            processed["requests_per_second"] = system.get("requests_per_second", 0.0)
            processed["error_rate"] = system.get("error_rate", 0.0)
            processed["availability"] = system.get("availability", 1.0)
        
        # Business metrics
        if "business_metrics" in metrics_data:
            business = metrics_data["business_metrics"]
            processed["transactions_processed"] = business.get("transactions_processed", 0)
            processed["fraud_detected"] = business.get("fraud_detected", 0)
            processed["fraud_prevented_amount"] = business.get("fraud_prevented_amount", 0.0)
            processed["false_positives"] = business.get("false_positives", 0)
        
        processed["timestamp"] = timestamp.isoformat()
        return processed
    
    async def _detect_performance_anomalies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in performance metrics"""
        anomalies = []
        
        # Check against thresholds
        for metric_name, value in metrics.items():
            if metric_name in self.metric_thresholds and isinstance(value, (int, float)):
                thresholds = self.metric_thresholds[metric_name]
                
                if value >= thresholds.get("critical", float('inf')):
                    anomalies.append({
                        "metric": metric_name,
                        "value": value,
                        "severity": "critical",
                        "threshold": thresholds["critical"],
                        "description": f"{metric_name} exceeded critical threshold"
                    })
                elif value >= thresholds.get("warning", float('inf')):
                    anomalies.append({
                        "metric": metric_name,
                        "value": value,
                        "severity": "warning",
                        "threshold": thresholds["warning"],
                        "description": f"{metric_name} exceeded warning threshold"
                    })
        
        # Statistical anomaly detection
        statistical_anomalies = await self._detect_statistical_anomalies(metrics)
        anomalies.extend(statistical_anomalies)
        
        return anomalies
    
    async def _detect_statistical_anomalies(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect statistical anomalies using historical data"""
        anomalies = []
        
        for metric_name, current_value in current_metrics.items():
            if not isinstance(current_value, (int, float)) or metric_name == "timestamp":
                continue
                
            if metric_name not in self.performance_metrics:
                continue
                
            historical_values = list(self.performance_metrics[metric_name])
            if len(historical_values) < 10:  # Need minimum history
                continue
            
            # Calculate statistical bounds
            mean = np.mean(historical_values)
            std = np.std(historical_values)
            
            # Z-score based anomaly detection
            z_score = abs(current_value - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero
            
            if z_score > 3:  # 3 sigma rule
                anomalies.append({
                    "metric": metric_name,
                    "value": current_value,
                    "severity": "statistical_anomaly",
                    "z_score": float(z_score),
                    "historical_mean": float(mean),
                    "description": f"{metric_name} is {z_score:.2f} standard deviations from historical mean"
                })
        
        return anomalies
    
    async def _generate_performance_alerts(self, anomalies: List[Dict[str, Any]], metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on detected anomalies"""
        alerts = []
        
        for anomaly in anomalies:
            severity = anomaly["severity"]
            metric = anomaly["metric"]
            
            # Create alert
            alert = {
                "type": "performance_anomaly",
                "severity": severity,
                "metric": metric,
                "current_value": anomaly["value"],
                "description": anomaly["description"],
                "status": "active",
                "requires_action": severity in ["critical", "statistical_anomaly"]
            }
            
            # Add context-specific information
            if metric == "avg_processing_time_ms":
                alert["recommended_action"] = "Check system resources and optimize processing pipeline"
            elif metric == "false_positive_rate":
                alert["recommended_action"] = "Review and tune fraud detection thresholds"
            elif metric == "fraud_detection_accuracy":
                alert["recommended_action"] = "Consider model retraining or feature engineering"
            else:
                alert["recommended_action"] = f"Investigate {metric} performance degradation"
            
            alerts.append(alert)
        
        return alerts
    
    def _update_metric_history(self, metrics: Dict[str, Any], timestamp: datetime) -> None:
        """Update historical metrics for trend analysis"""
        max_history_size = 1000  # Keep last 1000 data points
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)) and metric_name != "timestamp":
                if len(self.performance_metrics[metric_name]) >= max_history_size:
                    self.performance_metrics[metric_name].popleft()
                self.performance_metrics[metric_name].append(value)
    
    async def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        if not self.performance_metrics:
            return {"score": 1.0, "status": "healthy", "components": {}}
        
        component_scores = {}
        
        # Performance component
        if "avg_processing_time_ms" in self.performance_metrics:
            recent_times = list(self.performance_metrics["avg_processing_time_ms"])[-10:]
            avg_time = np.mean(recent_times) if recent_times else 0
            performance_score = max(0.0, 1.0 - (avg_time / 100.0))  # Normalize by 100ms
            component_scores["performance"] = performance_score
        
        # Accuracy component
        if "fraud_detection_accuracy" in self.performance_metrics:
            recent_accuracy = list(self.performance_metrics["fraud_detection_accuracy"])[-10:]
            avg_accuracy = np.mean(recent_accuracy) if recent_accuracy else 1.0
            component_scores["accuracy"] = avg_accuracy
        
        # Availability component
        if "availability" in self.performance_metrics:
            recent_availability = list(self.performance_metrics["availability"])[-10:]
            avg_availability = np.mean(recent_availability) if recent_availability else 1.0
            component_scores["availability"] = avg_availability
        
        # Overall score
        if component_scores:
            overall_score = np.mean(list(component_scores.values()))
        else:
            overall_score = 1.0
        
        # Determine status
        if overall_score >= 0.9:
            status = "healthy"
        elif overall_score >= 0.7:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "score": float(overall_score),
            "status": status,
            "components": {k: float(v) for k, v in component_scores.items()}
        }
    
    async def _create_retraining_plan(self, reason: str, model_ids: List[str], priority: str) -> Dict[str, Any]:
        """Create a model retraining plan"""
        retraining_id = str(uuid.uuid4())
        
        # Estimate completion time based on priority and number of models
        priority_multipliers = {"low": 4, "medium": 2, "high": 1, "critical": 0.5}
        base_time_hours = 2  # 2 hours base time per model
        estimated_hours = len(model_ids or [1]) * base_time_hours * priority_multipliers.get(priority, 2)
        
        estimated_completion = (datetime.utcnow() + timedelta(hours=estimated_hours)).isoformat()
        
        return {
            "retraining_id": retraining_id,
            "reason": reason,
            "affected_models": model_ids or ["all_models"],
            "priority": priority,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "estimated_completion": estimated_completion,
            "estimated_duration_hours": estimated_hours
        }
    
    async def _analyze_business_impact(self, time_window_hours: int, analysis_type: str) -> Dict[str, Any]:
        """Analyze business impact of fraud detection system"""
        
        # Simulate business impact analysis (in production, this would query real data)
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        # Simulated metrics
        total_transactions = np.random.poisson(10000)
        fraud_detected = np.random.poisson(200)
        false_positives = np.random.poisson(50)
        false_negatives = np.random.poisson(20)
        
        # Financial impact calculations
        avg_fraud_amount = 500
        avg_transaction_amount = 100
        fraud_prevented_amount = fraud_detected * avg_fraud_amount
        false_positive_cost = false_positives * avg_transaction_amount * 0.1  # 10% cost of lost transaction
        false_negative_cost = false_negatives * avg_fraud_amount
        
        net_benefit = fraud_prevented_amount - false_positive_cost - false_negative_cost
        
        # Performance metrics
        precision = fraud_detected / (fraud_detected + false_positives) if (fraud_detected + false_positives) > 0 else 0
        recall = fraud_detected / (fraud_detected + false_negatives) if (fraud_detected + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Generate recommendations
        recommendations = []
        if precision < 0.8:
            recommendations.append("Consider increasing fraud detection thresholds to reduce false positives")
        if recall < 0.7:
            recommendations.append("Consider lowering thresholds or improving model sensitivity to catch more fraud")
        if false_negative_cost > fraud_prevented_amount * 0.1:
            recommendations.append("High false negative cost - prioritize reducing missed fraud cases")
        
        return {
            "time_window_hours": time_window_hours,
            "total_transactions": total_transactions,
            "fraud_detected": fraud_detected,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "financial_impact": {
                "fraud_prevented_amount": float(fraud_prevented_amount),
                "false_positive_cost": float(false_positive_cost),
                "false_negative_cost": float(false_negative_cost),
                "net_benefit": float(net_benefit)
            },
            "performance_metrics": {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score)
            },
            "recommendations": recommendations
        }
    
    async def _generate_system_alerts(self, alert_type: str, severity: str, conditions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate system alerts based on conditions"""
        alerts = []
        
        if alert_type == "performance":
            # Check current system health
            system_health = await self._calculate_system_health()
            if system_health["score"] < conditions.get("min_health_score", 0.8):
                alerts.append({
                    "type": "system_health",
                    "severity": severity,
                    "message": f"System health score ({system_health['score']:.2f}) below threshold",
                    "details": system_health,
                    "status": "active"
                })
        
        elif alert_type == "business":
            # Check business impact
            if self.business_impact_data:
                impact = self.business_impact_data.get("current_impact", {})
                financial = impact.get("financial_impact", {})
                net_benefit = financial.get("net_benefit", 0)
                
                if net_benefit < conditions.get("min_net_benefit", 0):
                    alerts.append({
                        "type": "business_impact",
                        "severity": severity,
                        "message": f"Negative business impact detected: ${net_benefit:,.2f}",
                        "details": financial,
                        "status": "active"
                    })
        
        elif alert_type == "model":
            # Check for model performance degradation
            if "fraud_detection_accuracy" in self.performance_metrics:
                recent_accuracy = list(self.performance_metrics["fraud_detection_accuracy"])[-5:]
                if recent_accuracy and np.mean(recent_accuracy) < conditions.get("min_accuracy", 0.8):
                    alerts.append({
                        "type": "model_performance",
                        "severity": severity,
                        "message": f"Model accuracy degraded to {np.mean(recent_accuracy):.2f}",
                        "details": {"recent_accuracy": recent_accuracy},
                        "status": "active"
                    })
        
        return alerts
    
    def get_server(self) -> Server:
        """Get the MCP server instance"""
        return self.server
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics"""
        recent_alerts = [a for a in self.alert_history 
                        if (datetime.utcnow() - datetime.fromisoformat(a["created_at"])).seconds < 3600]
        
        return {
            "total_alerts": len(self.alert_history),
            "recent_alerts_1h": len(recent_alerts),
            "active_incidents": len(self.active_incidents),
            "retraining_queue_size": len(self.retraining_triggers),
            "monitored_metrics": list(self.performance_metrics.keys()),
            "metric_history_size": {k: len(v) for k, v in self.performance_metrics.items()},
            "system_health": None  # Will be calculated when requested
        }
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified time window"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [alert for alert in self.alert_history 
                if datetime.fromisoformat(alert["created_at"]) > cutoff]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alert_history:
            if alert.get("id") == alert_id:
                alert["status"] = "acknowledged"
                alert["acknowledged_at"] = datetime.utcnow().isoformat()
                return True
        return False