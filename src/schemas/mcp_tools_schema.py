from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class MCPTool(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, Any]


# Data Intelligence MCP Server Tools
DATA_INTELLIGENCE_TOOLS = [
    MCPTool(
        name="ingest_transaction",
        description="Ingest and validate incoming transaction data",
        inputSchema={
            "type": "object",
            "properties": {
                "transaction_data": {
                    "type": "object",
                    "description": "Raw transaction data to be processed"
                },
                "source": {
                    "type": "string",
                    "description": "Source system or channel of the transaction"
                },
                "timestamp": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Timestamp when data was received"
                }
            },
            "required": ["transaction_data", "source", "timestamp"]
        }
    ),
    MCPTool(
        name="validate_data",
        description="Perform comprehensive data quality assessment",
        inputSchema={
            "type": "object",
            "properties": {
                "data_batch": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Batch of transaction data to validate"
                },
                "validation_rules": {
                    "type": "object",
                    "description": "Custom validation rules and thresholds"
                }
            },
            "required": ["data_batch"]
        }
    ),
    MCPTool(
        name="engineer_features",
        description="Automatically engineer new features from raw data",
        inputSchema={
            "type": "object",
            "properties": {
                "raw_data": {
                    "type": "object",
                    "description": "Raw transaction and user data"
                },
                "feature_config": {
                    "type": "object",
                    "properties": {
                        "feature_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Types of features to generate"
                        },
                        "lookback_windows": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Time windows for aggregation features"
                        }
                    }
                }
            },
            "required": ["raw_data"]
        }
    ),
    MCPTool(
        name="detect_drift",
        description="Monitor and detect data/model drift",
        inputSchema={
            "type": "object",
            "properties": {
                "current_data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Current data batch for drift analysis"
                },
                "reference_data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Reference/baseline data for comparison"
                },
                "drift_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Threshold for drift detection"
                },
                "drift_method": {
                    "type": "string",
                    "enum": ["ks_test", "psi", "js_divergence", "wasserstein"],
                    "description": "Statistical method for drift detection"
                }
            },
            "required": ["current_data", "reference_data"]
        }
    )
]

# Model Orchestration MCP Server Tools
MODEL_ORCHESTRATION_TOOLS = [
    MCPTool(
        name="train_model",
        description="Train or retrain a fraud detection model",
        inputSchema={
            "type": "object",
            "properties": {
                "model_config": {
                    "type": "object",
                    "properties": {
                        "model_type": {
                            "type": "string",
                            "enum": ["xgboost", "lightgbm", "neural_network", "ensemble"]
                        },
                        "hyperparameters": {"type": "object"},
                        "training_data_filter": {"type": "object"}
                    }
                },
                "training_data": {
                    "type": "array",
                    "items": {"type": "object"}
                },
                "validation_split": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 0.5,
                    "default": 0.2
                }
            },
            "required": ["model_config", "training_data"]
        }
    ),
    MCPTool(
        name="predict_fraud",
        description="Generate fraud predictions using ensemble models",
        inputSchema={
            "type": "object",
            "properties": {
                "transaction": {
                    "type": "object",
                    "description": "Transaction data for prediction"
                },
                "model_ensemble": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of model IDs to use in ensemble"
                },
                "explain": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to include prediction explanations"
                }
            },
            "required": ["transaction"]
        }
    ),
    MCPTool(
        name="update_model_weights",
        description="Update ensemble model weights based on performance",
        inputSchema={
            "type": "object",
            "properties": {
                "performance_metrics": {
                    "type": "object",
                    "description": "Performance metrics for each model"
                },
                "feedback_data": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Recent feedback and ground truth data"
                },
                "learning_rate": {
                    "type": "number",
                    "minimum": 0.001,
                    "maximum": 1.0,
                    "default": 0.01
                }
            },
            "required": ["performance_metrics"]
        }
    ),
    MCPTool(
        name="a_b_test_models",
        description="Conduct A/B testing between different model versions",
        inputSchema={
            "type": "object",
            "properties": {
                "model_a": {"type": "string", "description": "First model ID"},
                "model_b": {"type": "string", "description": "Second model ID"},
                "traffic_split": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 0.9,
                    "default": 0.5,
                    "description": "Fraction of traffic for model A"
                },
                "test_duration_hours": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 24
                },
                "success_metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics to optimize for"
                }
            },
            "required": ["model_a", "model_b"]
        }
    )
]

# Decision Engine MCP Server Tools
DECISION_ENGINE_TOOLS = [
    MCPTool(
        name="assess_risk",
        description="Perform comprehensive risk assessment",
        inputSchema={
            "type": "object",
            "properties": {
                "transaction": {
                    "type": "object",
                    "description": "Transaction data for risk assessment"
                },
                "user_context": {
                    "type": "object",
                    "description": "Additional user context and history"
                },
                "risk_tolerance": {
                    "type": "string",
                    "enum": ["conservative", "moderate", "aggressive"],
                    "default": "moderate"
                }
            },
            "required": ["transaction"]
        }
    ),
    MCPTool(
        name="make_decision",
        description="Make final fraud decision with confidence scoring",
        inputSchema={
            "type": "object",
            "properties": {
                "risk_scores": {
                    "type": "object",
                    "description": "Risk scores from different models/agents"
                },
                "business_rules": {
                    "type": "object",
                    "description": "Business rules and constraints"
                },
                "cost_matrix": {
                    "type": "object",
                    "description": "Cost matrix for false positives/negatives"
                }
            },
            "required": ["risk_scores"]
        }
    ),
    MCPTool(
        name="generate_explanation",
        description="Generate human-readable explanation for decisions",
        inputSchema={
            "type": "object",
            "properties": {
                "decision": {
                    "type": "object",
                    "description": "Decision object with scores and reasoning"
                },
                "explanation_type": {
                    "type": "string",
                    "enum": ["technical", "business", "customer_facing"],
                    "default": "business"
                },
                "include_recommendations": {
                    "type": "boolean",
                    "default": True
                }
            },
            "required": ["decision"]
        }
    ),
    MCPTool(
        name="escalate_for_review",
        description="Escalate transactions for human review",
        inputSchema={
            "type": "object",
            "properties": {
                "transaction_id": {"type": "string"},
                "escalation_reason": {"type": "string"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "default": "medium"
                },
                "reviewer_expertise": {
                    "type": "string",
                    "description": "Required reviewer expertise level"
                },
                "context": {
                    "type": "object",
                    "description": "Additional context for reviewers"
                }
            },
            "required": ["transaction_id", "escalation_reason"]
        }
    )
]

# Monitoring & Response MCP Server Tools
MONITORING_TOOLS = [
    MCPTool(
        name="log_performance_metric",
        description="Log system performance metrics",
        inputSchema={
            "type": "object",
            "properties": {
                "metric_name": {"type": "string"},
                "metric_value": {"type": "number"},
                "timestamp": {"type": "string", "format": "date-time"},
                "tags": {
                    "type": "object",
                    "description": "Additional metric tags"
                },
                "component": {"type": "string"}
            },
            "required": ["metric_name", "metric_value", "component"]
        }
    ),
    MCPTool(
        name="trigger_alert",
        description="Trigger system alerts based on conditions",
        inputSchema={
            "type": "object",
            "properties": {
                "alert_type": {"type": "string"},
                "severity": {
                    "type": "string",
                    "enum": ["info", "warning", "error", "critical"]
                },
                "message": {"type": "string"},
                "component": {"type": "string"},
                "metadata": {"type": "object"}
            },
            "required": ["alert_type", "severity", "message", "component"]
        }
    ),
    MCPTool(
        name="health_check",
        description="Perform comprehensive system health check",
        inputSchema={
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Components to check"
                },
                "include_performance": {
                    "type": "boolean",
                    "default": True
                }
            }
        }
    ),
    MCPTool(
        name="auto_scale_resources",
        description="Automatically scale system resources",
        inputSchema={
            "type": "object",
            "properties": {
                "resource_type": {
                    "type": "string",
                    "enum": ["cpu", "memory", "instances", "storage"]
                },
                "target_utilization": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 0.9
                },
                "scaling_policy": {
                    "type": "string",
                    "enum": ["aggressive", "conservative", "predictive"]
                }
            },
            "required": ["resource_type"]
        }
    )
]


# Combined tools registry
ALL_MCP_TOOLS = {
    "data_intelligence": DATA_INTELLIGENCE_TOOLS,
    "model_orchestration": MODEL_ORCHESTRATION_TOOLS,
    "decision_engine": DECISION_ENGINE_TOOLS,
    "monitoring": MONITORING_TOOLS
}