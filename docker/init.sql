-- Initialize fraud detection database schema

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS fraud_detection;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS model_data;

-- Set search path
SET search_path TO fraud_detection, public;

-- Transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    transaction_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    
    -- Geographic data
    country VARCHAR(3),
    city VARCHAR(255),
    ip_address INET,
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    
    -- Device data
    device_id VARCHAR(255),
    device_type VARCHAR(100),
    browser VARCHAR(100),
    operating_system VARCHAR(100),
    is_mobile BOOLEAN DEFAULT FALSE,
    
    -- Risk indicators
    velocity_1h INTEGER DEFAULT 0,
    velocity_24h INTEGER DEFAULT 0,
    account_age_days INTEGER,
    is_first_transaction BOOLEAN DEFAULT FALSE,
    
    -- Payment data
    payment_method VARCHAR(100),
    card_last_four VARCHAR(4),
    
    -- Metadata
    features JSONB,
    metadata JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Fraud scores table
CREATE TABLE IF NOT EXISTS fraud_scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id VARCHAR(255) NOT NULL REFERENCES transactions(transaction_id),
    risk_score DECIMAL(5,4) NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    risk_factors TEXT[],
    model_version VARCHAR(50),
    processing_time_ms DECIMAL(10,2),
    explanation TEXT,
    recommended_action VARCHAR(50),
    
    -- Agent-specific scores
    pattern_score DECIMAL(5,4),
    velocity_score DECIMAL(5,4), 
    device_score DECIMAL(5,4),
    behavioral_score DECIMAL(5,4),
    
    -- Ensemble metadata
    ensemble_weights JSONB,
    model_consensus DECIMAL(5,4),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Fraud alerts table
CREATE TABLE IF NOT EXISTS fraud_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id VARCHAR(255) UNIQUE NOT NULL,
    transaction_id VARCHAR(255) NOT NULL REFERENCES transactions(transaction_id),
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    description TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'resolved', 'false_positive')),
    assigned_to VARCHAR(255),
    
    -- Context and evidence
    evidence JSONB,
    similar_transactions TEXT[],
    user_history_flags TEXT[],
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User profiles table
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) UNIQUE NOT NULL,
    
    -- Transaction statistics
    transaction_count INTEGER DEFAULT 0,
    avg_amount DECIMAL(15,2) DEFAULT 0,
    total_amount DECIMAL(15,2) DEFAULT 0,
    first_transaction_date TIMESTAMP WITH TIME ZONE,
    last_transaction_date TIMESTAMP WITH TIME ZONE,
    
    -- Behavioral patterns
    common_countries TEXT[],
    common_devices TEXT[],
    avg_daily_transactions DECIMAL(8,2) DEFAULT 0,
    typical_transaction_times INTEGER[], -- hours of day
    
    -- Risk indicators
    fraud_score_history DECIMAL(5,4)[],
    account_flags TEXT[],
    risk_level VARCHAR(20) DEFAULT 'low',
    
    -- Metadata
    profile_data JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Model performance table (in model_data schema)
CREATE TABLE IF NOT EXISTS model_data.model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    
    -- Performance metrics
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_roc DECIMAL(5,4),
    
    -- Training information
    training_samples INTEGER,
    validation_samples INTEGER,
    training_date TIMESTAMP WITH TIME ZONE,
    
    -- Deployment status
    is_active BOOLEAN DEFAULT FALSE,
    deployment_date TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    hyperparameters JSONB,
    training_config JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System metrics table (in monitoring schema)
CREATE TABLE IF NOT EXISTS monitoring.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    component VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    tags JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent decisions table
CREATE TABLE IF NOT EXISTS agent_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    decision_id VARCHAR(255) UNIQUE NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    transaction_id VARCHAR(255) NOT NULL REFERENCES transactions(transaction_id),
    
    decision_type VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    risk_score DECIMAL(5,4) NOT NULL,
    reasoning TEXT[],
    evidence JSONB,
    processing_time_ms DECIMAL(10,2),
    
    -- Agent metadata
    agent_specialization VARCHAR(100),
    agent_version VARCHAR(50),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);
CREATE INDEX IF NOT EXISTS idx_transactions_country ON transactions(country);
CREATE INDEX IF NOT EXISTS idx_transactions_amount ON transactions(amount);

CREATE INDEX IF NOT EXISTS idx_fraud_scores_transaction_id ON fraud_scores(transaction_id);
CREATE INDEX IF NOT EXISTS idx_fraud_scores_risk_score ON fraud_scores(risk_score);
CREATE INDEX IF NOT EXISTS idx_fraud_scores_created_at ON fraud_scores(created_at);

CREATE INDEX IF NOT EXISTS idx_fraud_alerts_status ON fraud_alerts(status);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_severity ON fraud_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_created_at ON fraud_alerts(created_at);

CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_risk_level ON user_profiles(risk_level);

CREATE INDEX IF NOT EXISTS idx_system_metrics_component ON monitoring.system_metrics(component);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON monitoring.system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON monitoring.system_metrics(metric_name);

CREATE INDEX IF NOT EXISTS idx_agent_decisions_agent_id ON agent_decisions(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_transaction_id ON agent_decisions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_created_at ON agent_decisions(created_at);

-- Create functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_transactions_updated_at 
    BEFORE UPDATE ON transactions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_fraud_alerts_updated_at 
    BEFORE UPDATE ON fraud_alerts 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_profiles_updated_at 
    BEFORE UPDATE ON user_profiles 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT USAGE ON SCHEMA fraud_detection TO fraud_user;
GRANT USAGE ON SCHEMA monitoring TO fraud_user;
GRANT USAGE ON SCHEMA model_data TO fraud_user;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA fraud_detection TO fraud_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO fraud_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA model_data TO fraud_user;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA fraud_detection TO fraud_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO fraud_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA model_data TO fraud_user;

-- Insert sample data for testing
INSERT INTO transactions (
    transaction_id, user_id, amount, currency, transaction_type, timestamp,
    country, device_id, device_type, is_mobile, velocity_1h, velocity_24h,
    account_age_days, is_first_transaction, payment_method
) VALUES 
(
    'tx_sample_001', 'user_12345', 299.99, 'USD', 'purchase', CURRENT_TIMESTAMP,
    'US', 'device_abc123', 'desktop', false, 1, 3,
    90, false, 'credit_card'
),
(
    'tx_sample_002', 'user_67890', 1500.00, 'USD', 'transfer', CURRENT_TIMESTAMP,
    'GB', 'device_def456', 'mobile', true, 5, 12,
    365, false, 'bank_transfer'
);

-- Insert corresponding user profiles
INSERT INTO user_profiles (
    user_id, transaction_count, avg_amount, total_amount,
    first_transaction_date, last_transaction_date,
    common_countries, avg_daily_transactions, risk_level
) VALUES 
(
    'user_12345', 45, 187.50, 8437.50,
    CURRENT_TIMESTAMP - INTERVAL '90 days', CURRENT_TIMESTAMP,
    ARRAY['US'], 0.5, 'low'
),
(
    'user_67890', 123, 567.33, 69781.59,
    CURRENT_TIMESTAMP - INTERVAL '365 days', CURRENT_TIMESTAMP,
    ARRAY['GB', 'US'], 0.34, 'medium'
);

-- Log database initialization
INSERT INTO monitoring.system_metrics (
    metric_name, metric_value, component, timestamp
) VALUES (
    'database_initialized', 1.0, 'postgresql', CURRENT_TIMESTAMP
);