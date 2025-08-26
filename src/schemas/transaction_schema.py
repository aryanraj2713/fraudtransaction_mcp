from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from enum import Enum


class TransactionType(str, Enum):
    PURCHASE = "purchase"
    TRANSFER = "transfer"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"
    REFUND = "refund"


class TransactionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DECLINED = "declined"
    FLAGGED = "flagged"
    UNDER_REVIEW = "under_review"


class GeographicData(BaseModel):
    country: str
    city: Optional[str] = None
    ip_address: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timezone: Optional[str] = None


class MerchantData(BaseModel):
    merchant_id: str
    merchant_name: str
    merchant_category: str
    merchant_risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class DeviceData(BaseModel):
    device_id: str
    device_type: str
    browser: Optional[str] = None
    operating_system: Optional[str] = None
    screen_resolution: Optional[str] = None
    is_mobile: bool = False


class UserBehaviorData(BaseModel):
    user_id: str
    session_duration: Optional[int] = None  # in seconds
    pages_visited: Optional[int] = None
    clicks_per_minute: Optional[float] = None
    typing_speed: Optional[float] = None  # characters per minute
    mouse_movements: Optional[int] = None


class Transaction(BaseModel):
    transaction_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    amount: Decimal = Field(..., gt=0)
    currency: str = Field(..., min_length=3, max_length=3)
    transaction_type: TransactionType
    timestamp: datetime
    status: TransactionStatus = TransactionStatus.PENDING
    
    # Geographic and location data
    geographic_data: GeographicData
    
    # Merchant information
    merchant_data: Optional[MerchantData] = None
    
    # Device and technical data
    device_data: DeviceData
    
    # User behavior patterns
    user_behavior: Optional[UserBehaviorData] = None
    
    # Payment method details
    payment_method: str
    card_last_four: Optional[str] = Field(default=None, min_length=4, max_length=4)
    
    # Risk indicators
    is_first_transaction: bool = False
    account_age_days: Optional[int] = Field(default=None, ge=0)
    velocity_1h: Optional[int] = Field(default=0, ge=0)  # transactions in last hour
    velocity_24h: Optional[int] = Field(default=0, ge=0)  # transactions in last 24h
    
    # Custom features and metadata
    features: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('currency')
    def validate_currency(cls, v):
        return v.upper()

    @validator('card_last_four')
    def validate_card_last_four(cls, v):
        if v and not v.isdigit():
            raise ValueError('Card last four must be digits only')
        return v


class TransactionBatch(BaseModel):
    batch_id: str
    transactions: List[Transaction]
    batch_timestamp: datetime
    source: str
    total_count: int
    
    @validator('total_count')
    def validate_total_count(cls, v, values):
        if 'transactions' in values and v != len(values['transactions']):
            raise ValueError('Total count must match transactions list length')
        return v


class FraudScore(BaseModel):
    transaction_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_factors: List[str] = Field(default_factory=list)
    model_version: str
    processing_time_ms: float
    explanation: Optional[str] = None
    recommended_action: str  # approve, decline, review
    
    # Agent-specific scores
    pattern_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    velocity_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    device_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    behavioral_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    
    # Ensemble metadata
    ensemble_weights: Dict[str, float] = Field(default_factory=dict)
    model_consensus: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class FraudAlert(BaseModel):
    alert_id: str
    transaction_id: str
    alert_type: str
    severity: str  # low, medium, high, critical
    description: str
    created_at: datetime
    status: str = "open"  # open, investigating, resolved, false_positive
    assigned_to: Optional[str] = None
    
    # Context and evidence
    evidence: Dict[str, Any] = Field(default_factory=dict)
    similar_transactions: List[str] = Field(default_factory=list)
    user_history_flags: List[str] = Field(default_factory=list)


class ModelPrediction(BaseModel):
    model_id: str
    model_version: str
    prediction: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    features_used: List[str]
    prediction_time_ms: float
    model_type: str  # ensemble, xgboost, neural_network, etc.