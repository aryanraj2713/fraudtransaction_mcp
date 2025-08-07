from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class TransactionType(str, Enum):
    PURCHASE = "purchase"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    DEPOSIT = "deposit"

class TransactionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DECLINED = "declined"
    FLAGGED = "flagged"

class TransactionSchema(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="User identifier")
    amount: float = Field(..., description="Transaction amount")
    currency: str = Field(default="USD", description="Transaction currency")
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")
    merchant_category: Optional[str] = Field(None, description="Merchant category code")
    location: Optional[Dict[str, Any]] = Field(None, description="Transaction location")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    device_info: Optional[Dict[str, Any]] = Field(None, description="Device information")
    ip_address: Optional[str] = Field(None, description="IP address")
    status: TransactionStatus = Field(default=TransactionStatus.PENDING)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class FraudScoreSchema(BaseModel):
    transaction_id: str
    fraud_score: float = Field(..., ge=0.0, le=1.0, description="Fraud probability score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the score")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    decision: str = Field(..., description="Fraud decision: approve, decline, review")
    explanation: str = Field(..., description="Human-readable explanation")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class ModelUpdateSchema(BaseModel):
    model_id: str
    update_type: str = Field(..., description="Type of update: retrain, tune, deploy")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    trigger_reason: str = Field(..., description="Reason for the update")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class UpdateResultSchema(BaseModel):
    model_id: str
    success: bool
    new_version: str
    performance_metrics: Dict[str, float]
    deployment_status: str
    message: str

class PatternSchema(BaseModel):
    pattern_id: str
    pattern_type: str = Field(..., description="Type of pattern detected")
    features: Dict[str, Any] = Field(..., description="Pattern features")
    confidence: float = Field(..., ge=0.0, le=1.0)
    occurrences: int = Field(..., description="Number of times pattern observed")
    first_seen: datetime
    last_seen: datetime

class InvestigationResultSchema(BaseModel):
    pattern_id: str
    investigation_status: str
    findings: List[str]
    recommended_actions: List[str]
    risk_level: str = Field(..., description="low, medium, high, critical")
    affected_transactions: List[str] = Field(default_factory=list)