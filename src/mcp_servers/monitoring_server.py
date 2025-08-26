import asyncio
import logging
import json
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import uuid
from enum import Enum

from .base_server import BaseMCPServer
from schemas.mcp_tools_schema import MONITORING_TOOLS

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ResourceType(str, Enum):
    CPU = "cpu"
    MEMORY = "memory"
    INSTANCES = "instances"
    STORAGE = "storage"


@dataclass
class PerformanceMetric:
    metric_name: str
    metric_value: float
    timestamp: datetime
    component: str
    tags: Dict[str, Any]
    unit: str = ""


@dataclass
class SystemAlert:
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    component: str
    triggered_at: datetime
    resolved_at: Optional[datetime]
    metadata: Dict[str, Any]
    status: str = "active"  # active, acknowledged, resolved


@dataclass
class HealthCheckResult:
    component: str
    status: str  # healthy, degraded, unhealthy
    response_time_ms: float
    last_check: datetime
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class ScalingAction:
    action_id: str
    resource_type: ResourceType
    action: str  # scale_up, scale_down
    current_value: float
    target_value: float
    reason: str
    timestamp: datetime
    status: str = "pending"  # pending, executing, completed, failed


class MetricsCollector:
    """Advanced metrics collection system."""
    
    def __init__(self, retention_hours: int = 24):
        self.metrics_store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.retention_hours = retention_hours
        self.collection_interval = 60  # seconds
        self.running = False
        self.collection_thread = None
        
        # System thresholds
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "response_time_ms": 1000.0,
            "error_rate": 0.05,
            "throughput_transactions_per_second": 100.0
        }
    
    async def log_performance_metric(
        self,
        metric_name: str,
        metric_value: float,
        component: str,
        timestamp: str = None,
        tags: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Log system performance metrics."""
        
        try:
            if tags is None:
                tags = {}
            
            if timestamp is None:
                timestamp_dt = datetime.now()
            else:
                if isinstance(timestamp, str):
                    timestamp_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                elif isinstance(timestamp, datetime):
                    timestamp_dt = timestamp
                else:
                    logger.error(f"Invalid timestamp type: {type(timestamp)}")
                    timestamp_dt = datetime.now()
            
            metric = PerformanceMetric(
                metric_name=metric_name,
                metric_value=metric_value,
                timestamp=timestamp_dt,
                component=component,
                tags=tags
            )
            
            # Store metric
            metric_key = f"{component}.{metric_name}"
            self.metrics_store[metric_key].append(asdict(metric))
            
            # Clean old metrics
            await self._clean_old_metrics()
            
            # Check for threshold violations
            alert = await self._check_thresholds(metric)
            
            result = {
                "status": "success",
                "metric_stored": True,
                "metric_key": metric_key,
                "timestamp": timestamp_dt.isoformat(),
                "threshold_alert": alert is not None,
                "current_value": metric_value,
                "threshold": self.thresholds.get(metric_name)
            }
            
            if alert:
                result["alert_triggered"] = asdict(alert)
            
            logger.debug(f"Logged metric {metric_name}={metric_value} for {component}")
            return result
            
        except Exception as e:
            logger.error(f"Metric logging error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "metric_name": metric_name,
                "component": component
            }
    
    async def _clean_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        for metric_key, metrics_queue in self.metrics_store.items():
            # Remove old metrics from the front of the queue
            while metrics_queue and datetime.fromisoformat(metrics_queue[0]["timestamp"]) < cutoff_time:
                metrics_queue.popleft()
    
    async def _check_thresholds(self, metric: PerformanceMetric) -> Optional[SystemAlert]:
        """Check if metric violates thresholds."""
        threshold = self.thresholds.get(metric.metric_name)
        if not threshold:
            return None
        
        if metric.metric_value > threshold:
            alert = SystemAlert(
                alert_id=f"threshold_{uuid.uuid4().hex[:8]}",
                alert_type="threshold_violation",
                severity=AlertSeverity.WARNING if metric.metric_value < threshold * 1.2 else AlertSeverity.ERROR,
                message=f"{metric.metric_name} exceeded threshold: {metric.metric_value} > {threshold}",
                component=metric.component,
                triggered_at=metric.timestamp,
                resolved_at=None,
                metadata={
                    "metric_name": metric.metric_name,
                    "current_value": metric.metric_value,
                    "threshold": threshold,
                    "tags": metric.tags
                }
            )
            return alert
        
        return None
    
    def get_metrics_summary(self, component: str = None, hours: int = 1) -> Dict[str, Any]:
        """Get summary of metrics for a component or all components."""
        summary = {}
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for metric_key, metrics_queue in self.metrics_store.items():
            comp, metric_name = metric_key.split(".", 1)
            
            if component and comp != component:
                continue
            
            # Filter recent metrics
            recent_metrics = [
                m for m in metrics_queue 
                if datetime.fromisoformat(m["timestamp"]) >= cutoff_time
            ]
            
            if recent_metrics:
                values = [m["metric_value"] for m in recent_metrics]
                summary[metric_key] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1],
                    "component": comp,
                    "metric_name": metric_name
                }
        
        return summary


class AlertManager:
    """Comprehensive alert management system."""
    
    def __init__(self):
        self.active_alerts: Dict[str, SystemAlert] = {}
        self.alert_history: List[SystemAlert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_channels: List[Callable] = []
        
        # Default alert rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alerting rules."""
        self.alert_rules = {
            "high_cpu": {
                "condition": "cpu_usage > 80",
                "severity": AlertSeverity.WARNING,
                "cooldown_minutes": 5
            },
            "high_memory": {
                "condition": "memory_usage > 85",
                "severity": AlertSeverity.ERROR,
                "cooldown_minutes": 5
            },
            "slow_response": {
                "condition": "response_time_ms > 1000",
                "severity": AlertSeverity.WARNING,
                "cooldown_minutes": 2
            },
            "high_error_rate": {
                "condition": "error_rate > 0.05",
                "severity": AlertSeverity.ERROR,
                "cooldown_minutes": 1
            }
        }
    
    async def trigger_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        component: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Trigger system alerts based on conditions."""
        
        try:
            if metadata is None:
                metadata = {}
            
            alert_severity = AlertSeverity(severity.lower())
            
            # Check if similar alert is already active (deduplication)
            duplicate_alert = self._find_duplicate_alert(alert_type, component)
            if duplicate_alert:
                return {
                    "status": "deduplicated",
                    "alert_id": duplicate_alert.alert_id,
                    "message": "Similar alert already active",
                    "existing_alert": asdict(duplicate_alert)
                }
            
            # Create new alert
            alert = SystemAlert(
                alert_id=f"alert_{uuid.uuid4().hex[:8]}",
                alert_type=alert_type,
                severity=alert_severity,
                message=message,
                component=component,
                triggered_at=datetime.now(),
                resolved_at=None,
                metadata=metadata
            )
            
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Send notifications
            await self._send_notifications(alert)
            
            result = {
                "status": "success",
                "alert_triggered": True,
                "alert_id": alert.alert_id,
                "severity": severity,
                "component": component,
                "triggered_at": alert.triggered_at.isoformat(),
                "notification_sent": len(self.notification_channels) > 0
            }
            
            logger.warning(f"Alert triggered [{alert.alert_id}]: {message} (severity: {severity})")
            return result
            
        except Exception as e:
            logger.error(f"Alert triggering error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "alert_type": alert_type,
                "component": component
            }
    
    def _find_duplicate_alert(self, alert_type: str, component: str) -> Optional[SystemAlert]:
        """Find duplicate active alerts."""
        for alert in self.active_alerts.values():
            if (alert.alert_type == alert_type and 
                alert.component == component and 
                alert.status == "active"):
                return alert
        return None
    
    async def _send_notifications(self, alert: SystemAlert):
        """Send alert notifications through configured channels."""
        for notification_func in self.notification_channels:
            try:
                await notification_func(alert)
            except Exception as e:
                logger.error(f"Notification sending error: {e}")
    
    def resolve_alert(self, alert_id: str, resolution_note: str = "") -> Dict[str, Any]:
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            return {
                "status": "error",
                "error": f"Alert {alert_id} not found or already resolved"
            }
        
        alert = self.active_alerts[alert_id]
        alert.resolved_at = datetime.now()
        alert.status = "resolved"
        alert.metadata["resolution_note"] = resolution_note
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert resolved [{alert_id}]: {resolution_note}")
        
        return {
            "status": "success",
            "alert_id": alert_id,
            "resolved_at": alert.resolved_at.isoformat(),
            "resolution_note": resolution_note
        }
    
    def get_active_alerts(self, severity: str = None, component: str = None) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity.value == severity.lower()]
        
        if component:
            alerts = [a for a in alerts if a.component == component]
        
        return [asdict(alert) for alert in alerts]


class HealthChecker:
    """System health monitoring and checking."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, HealthCheckResult] = {}
        self.check_interval = 30  # seconds
        self.running = False
        self.check_thread = None
    
    def register_health_check(self, component: str, check_func: Callable):
        """Register a health check function for a component."""
        self.health_checks[component] = check_func
        logger.info(f"Registered health check for {component}")
    
    async def health_check(
        self,
        components: List[str] = None,
        include_performance: bool = True
    ) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        
        try:
            if components is None:
                components = list(self.health_checks.keys())
                # Add default system components
                if not components:
                    components = ["system", "database", "cache", "external_services"]
            
            health_results = {}
            overall_status = "healthy"
            
            for component in components:
                try:
                    if component in self.health_checks:
                        # Run custom health check
                        result = await self.health_checks[component]()
                    else:
                        # Run default health check
                        result = await self._default_health_check(component)
                    
                    health_results[component] = asdict(result)
                    
                    # Update overall status
                    if result.status == "unhealthy":
                        overall_status = "unhealthy"
                    elif result.status == "degraded" and overall_status == "healthy":
                        overall_status = "degraded"
                    
                    # Store result
                    self.health_status[component] = result
                    
                except Exception as e:
                    error_result = HealthCheckResult(
                        component=component,
                        status="unhealthy",
                        response_time_ms=0,
                        last_check=datetime.now(),
                        details={},
                        error_message=str(e)
                    )
                    health_results[component] = asdict(error_result)
                    overall_status = "unhealthy"
            
            # Add performance metrics if requested
            performance_data = {}
            if include_performance:
                performance_data = await self._collect_performance_data()
            
            result = {
                "status": "success",
                "overall_health": overall_status,
                "check_timestamp": datetime.now().isoformat(),
                "components_checked": len(components),
                "healthy_components": len([r for r in health_results.values() if r["status"] == "healthy"]),
                "component_health": health_results
            }
            
            if include_performance:
                result["performance_data"] = performance_data
            
            logger.info(f"Health check completed: {overall_status} ({len(components)} components)")
            return result
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "components_requested": components or []
            }
    
    async def _default_health_check(self, component: str) -> HealthCheckResult:
        """Default health check implementation."""
        start_time = time.time()
        
        try:
            # Simulate component-specific checks
            if component == "system":
                details = await self._check_system_resources()
                status = "healthy" if details["cpu_usage"] < 80 and details["memory_usage"] < 85 else "degraded"
            
            elif component == "database":
                # Simulate database check
                details = {"connection": "active", "response_time_ms": 50}
                status = "healthy"
            
            elif component == "cache":
                # Simulate cache check
                details = {"connection": "active", "hit_ratio": 0.85}
                status = "healthy"
            
            else:
                details = {"check": "basic_connectivity"}
                status = "healthy"
            
            response_time_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component=component,
                status=status,
                response_time_ms=response_time_ms,
                last_check=datetime.now(),
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=component,
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(),
                details={},
                error_message=str(e)
            )
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_free_gb": disk.free / (1024**3)
            }
        except Exception as e:
            logger.error(f"System resource check error: {e}")
            return {"error": str(e)}
    
    async def _collect_performance_data(self) -> Dict[str, Any]:
        """Collect current performance data."""
        try:
            system_data = await self._check_system_resources()
            
            return {
                "system_resources": system_data,
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": time.time() - psutil.boot_time()
            }
        except Exception as e:
            return {"error": str(e)}


class AutoScaler:
    """Intelligent auto-scaling system."""
    
    def __init__(self):
        self.scaling_policies = {
            "cpu": {
                "scale_up_threshold": 80.0,
                "scale_down_threshold": 30.0,
                "cooldown_minutes": 5
            },
            "memory": {
                "scale_up_threshold": 85.0,
                "scale_down_threshold": 40.0,
                "cooldown_minutes": 5
            }
        }
        
        self.scaling_history: List[ScalingAction] = []
        self.last_scaling_action: Dict[str, datetime] = {}
    
    async def auto_scale_resources(
        self,
        resource_type: str,
        target_utilization: float = None,
        scaling_policy: str = "conservative"
    ) -> Dict[str, Any]:
        """Automatically scale system resources."""
        
        try:
            resource_enum = ResourceType(resource_type.lower())
            
            # Get current resource utilization
            current_metrics = await self._get_current_utilization(resource_enum)
            current_value = current_metrics["utilization"]
            
            # Determine if scaling is needed
            scaling_decision = await self._make_scaling_decision(
                resource_enum, current_value, target_utilization, scaling_policy
            )
            
            if not scaling_decision["should_scale"]:
                return {
                    "status": "no_action",
                    "resource_type": resource_type,
                    "current_utilization": current_value,
                    "reason": scaling_decision["reason"],
                    "target_utilization": target_utilization
                }
            
            # Check cooldown period
            if self._is_in_cooldown(resource_enum):
                last_action_time = self.last_scaling_action.get(resource_type, datetime.min)
                return {
                    "status": "cooldown",
                    "resource_type": resource_type,
                    "current_utilization": current_value,
                    "last_scaling_action": last_action_time.isoformat(),
                    "cooldown_remaining_minutes": self._get_cooldown_remaining(resource_enum)
                }
            
            # Execute scaling action
            scaling_action = await self._execute_scaling(
                resource_enum, scaling_decision["action"], current_metrics, scaling_decision["reason"]
            )
            
            # Record scaling action
            self.scaling_history.append(scaling_action)
            self.last_scaling_action[resource_type] = datetime.now()
            
            result = {
                "status": "success",
                "scaling_action": asdict(scaling_action),
                "resource_type": resource_type,
                "action_taken": scaling_decision["action"],
                "current_utilization": current_value,
                "target_utilization": target_utilization,
                "scaling_policy": scaling_policy
            }
            
            logger.info(f"Auto-scaling executed: {resource_type} {scaling_decision['action']}")
            return result
            
        except Exception as e:
            logger.error(f"Auto-scaling error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "resource_type": resource_type
            }
    
    async def _get_current_utilization(self, resource_type: ResourceType) -> Dict[str, Any]:
        """Get current resource utilization."""
        try:
            if resource_type == ResourceType.CPU:
                utilization = psutil.cpu_percent(interval=1)
                return {"utilization": utilization, "unit": "percent"}
            
            elif resource_type == ResourceType.MEMORY:
                memory = psutil.virtual_memory()
                return {"utilization": memory.percent, "unit": "percent"}
            
            elif resource_type == ResourceType.INSTANCES:
                # Simulate instance count
                return {"utilization": 3, "unit": "count"}
            
            elif resource_type == ResourceType.STORAGE:
                disk = psutil.disk_usage('/')
                return {"utilization": disk.percent, "unit": "percent"}
            
            else:
                return {"utilization": 50.0, "unit": "percent"}
                
        except Exception as e:
            logger.error(f"Resource utilization check error: {e}")
            return {"utilization": 0.0, "unit": "percent", "error": str(e)}
    
    async def _make_scaling_decision(
        self,
        resource_type: ResourceType,
        current_value: float,
        target_utilization: float,
        scaling_policy: str
    ) -> Dict[str, Any]:
        """Make intelligent scaling decision."""
        
        policy_config = self.scaling_policies.get(resource_type.value, {})
        scale_up_threshold = policy_config.get("scale_up_threshold", 80.0)
        scale_down_threshold = policy_config.get("scale_down_threshold", 30.0)
        
        # Adjust thresholds based on target utilization
        if target_utilization:
            scale_up_threshold = min(target_utilization * 1.2, 95.0)
            scale_down_threshold = max(target_utilization * 0.6, 10.0)
        
        # Adjust for scaling policy
        if scaling_policy == "aggressive":
            scale_up_threshold *= 0.9
            scale_down_threshold *= 1.1
        elif scaling_policy == "conservative":
            scale_up_threshold *= 1.1
            scale_down_threshold *= 0.9
        
        # Make decision
        if current_value > scale_up_threshold:
            return {
                "should_scale": True,
                "action": "scale_up",
                "reason": f"Utilization {current_value:.1f}% exceeds scale-up threshold {scale_up_threshold:.1f}%"
            }
        elif current_value < scale_down_threshold:
            return {
                "should_scale": True,
                "action": "scale_down",
                "reason": f"Utilization {current_value:.1f}% below scale-down threshold {scale_down_threshold:.1f}%"
            }
        else:
            return {
                "should_scale": False,
                "action": None,
                "reason": f"Utilization {current_value:.1f}% within normal range"
            }
    
    def _is_in_cooldown(self, resource_type: ResourceType) -> bool:
        """Check if resource is in cooldown period."""
        last_action_time = self.last_scaling_action.get(resource_type.value)
        if not last_action_time:
            return False
        
        policy_config = self.scaling_policies.get(resource_type.value, {})
        cooldown_minutes = policy_config.get("cooldown_minutes", 5)
        
        cooldown_period = timedelta(minutes=cooldown_minutes)
        return datetime.now() - last_action_time < cooldown_period
    
    def _get_cooldown_remaining(self, resource_type: ResourceType) -> int:
        """Get remaining cooldown time in minutes."""
        last_action_time = self.last_scaling_action.get(resource_type.value)
        if not last_action_time:
            return 0
        
        policy_config = self.scaling_policies.get(resource_type.value, {})
        cooldown_minutes = policy_config.get("cooldown_minutes", 5)
        
        elapsed = datetime.now() - last_action_time
        remaining = cooldown_minutes - (elapsed.total_seconds() / 60)
        return max(0, int(remaining))
    
    async def _execute_scaling(
        self,
        resource_type: ResourceType,
        action: str,
        current_metrics: Dict[str, Any],
        reason: str
    ) -> ScalingAction:
        """Execute the scaling action."""
        
        current_value = current_metrics["utilization"]
        
        # Calculate target value (simplified)
        if action == "scale_up":
            if resource_type in [ResourceType.CPU, ResourceType.MEMORY]:
                target_value = current_value * 0.7  # Reduce utilization by 30%
            else:
                target_value = current_value + 1  # Add one instance/unit
        else:  # scale_down
            if resource_type in [ResourceType.CPU, ResourceType.MEMORY]:
                target_value = current_value * 1.3  # Accept higher utilization
            else:
                target_value = max(1, current_value - 1)  # Remove one instance/unit
        
        scaling_action = ScalingAction(
            action_id=f"scale_{uuid.uuid4().hex[:8]}",
            resource_type=resource_type,
            action=action,
            current_value=current_value,
            target_value=target_value,
            reason=reason,
            timestamp=datetime.now(),
            status="completed"  # Simplified - would be "executing" in real implementation
        )
        
        return scaling_action


class MonitoringServer(BaseMCPServer):
    """Monitoring & Response MCP Server for fraud detection system."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8004):
        super().__init__("Monitoring", "1.0.0", host, port)
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()
        self.auto_scaler = AutoScaler()
        
        # System monitoring stats
        self.monitoring_stats = {
            "metrics_collected": 0,
            "alerts_triggered": 0,
            "health_checks_performed": 0,
            "scaling_actions_taken": 0,
            "system_uptime_seconds": 0
        }
        
        self.start_time = time.time()
    
    async def initialize(self):
        """Initialize the Monitoring server."""
        # Register tools
        for tool in MONITORING_TOOLS:
            if tool.name == "log_performance_metric":
                self.register_tool(tool, self.log_performance_metric)
            elif tool.name == "trigger_alert":
                self.register_tool(tool, self.trigger_alert)
            elif tool.name == "health_check":
                self.register_tool(tool, self.health_check)
            elif tool.name == "auto_scale_resources":
                self.register_tool(tool, self.auto_scale_resources)
        
        # Add capabilities
        self.add_capability("performance_monitoring")
        self.add_capability("alerting")
        self.add_capability("health_checking")
        self.add_capability("auto_scaling")
        
        # Setup default health checks
        self.health_checker.register_health_check("fraud_detection_system", self._fraud_system_health_check)
        
        logger.info("Monitoring MCP Server initialized successfully")
    
    async def shutdown(self):
        """Clean up resources."""
        logger.info("Monitoring MCP Server shutting down")
    
    async def log_performance_metric(
        self,
        metric_name: str,
        metric_value: float,
        component: str,
        timestamp: str = None,
        tags: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Log system performance metrics."""
        try:
            result = await self.metrics_collector.log_performance_metric(
                metric_name, metric_value, component, timestamp, tags
            )
            
            if result.get("status") == "success":
                self.monitoring_stats["metrics_collected"] += 1
            
            # Add monitoring context
            result["monitoring_stats"] = self.monitoring_stats.copy()
            result["monitoring_stats"]["system_uptime_seconds"] = int(time.time() - self.start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Performance metric logging error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "metric_name": metric_name,
                "component": component
            }
    
    async def trigger_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        component: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Trigger system alerts based on conditions."""
        try:
            result = await self.alert_manager.trigger_alert(
                alert_type, severity, message, component, metadata
            )
            
            if result.get("status") == "success":
                self.monitoring_stats["alerts_triggered"] += 1
            
            # Add monitoring context
            result["monitoring_stats"] = self.monitoring_stats.copy()
            result["active_alerts_count"] = len(self.alert_manager.active_alerts)
            
            return result
            
        except Exception as e:
            logger.error(f"Alert triggering error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "alert_type": alert_type,
                "component": component
            }
    
    async def health_check(
        self,
        components: List[str] = None,
        include_performance: bool = True
    ) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        try:
            result = await self.health_checker.health_check(components, include_performance)
            
            if result.get("status") == "success":
                self.monitoring_stats["health_checks_performed"] += 1
            
            # Add monitoring context
            result["monitoring_stats"] = self.monitoring_stats.copy()
            result["monitoring_stats"]["system_uptime_seconds"] = int(time.time() - self.start_time)
            
            # Add recent alerts
            result["active_alerts"] = self.alert_manager.get_active_alerts()
            
            return result
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "components_requested": components or []
            }
    
    async def auto_scale_resources(
        self,
        resource_type: str,
        target_utilization: float = None,
        scaling_policy: str = "conservative"
    ) -> Dict[str, Any]:
        """Automatically scale system resources."""
        try:
            result = await self.auto_scaler.auto_scale_resources(
                resource_type, target_utilization, scaling_policy
            )
            
            if result.get("status") == "success":
                self.monitoring_stats["scaling_actions_taken"] += 1
            
            # Add monitoring context
            result["monitoring_stats"] = self.monitoring_stats.copy()
            result["scaling_history_count"] = len(self.auto_scaler.scaling_history)
            
            return result
            
        except Exception as e:
            logger.error(f"Auto-scaling error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "resource_type": resource_type
            }
    
    async def _fraud_system_health_check(self) -> HealthCheckResult:
        """Health check specific to fraud detection system."""
        start_time = time.time()
        
        try:
            # Check system components
            details = {
                "data_intelligence_server": "healthy",
                "model_orchestration_server": "healthy", 
                "decision_engine_server": "healthy",
                "agent_coordination": "healthy",
                "real_time_processing": "healthy"
            }
            
            # Simulate some checks
            processing_latency = 45  # ms
            if processing_latency > 100:
                status = "degraded"
                details["processing_latency_warning"] = f"High latency: {processing_latency}ms"
            else:
                status = "healthy"
            
            details["avg_processing_latency_ms"] = processing_latency
            details["active_agents"] = 5
            details["processed_transactions_last_hour"] = 1250
            
            response_time_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component="fraud_detection_system",
                status=status,
                response_time_ms=response_time_ms,
                last_check=datetime.now(),
                details=details
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="fraud_detection_system",
                status="unhealthy",
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(),
                details={},
                error_message=str(e)
            )


async def main():
    """Main function to run the Monitoring MCP Server."""
    server = MonitoringServer()
    
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await server.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())