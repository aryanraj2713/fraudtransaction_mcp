"""
Enhanced Feature Engineering Recommendations for MCP Fraud Detection System

This module demonstrates advanced feature engineering techniques that can significantly
improve fraud detection accuracy.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import hashlib

class AdvancedFeatureEngineer:
    """Enhanced feature engineering with fraud-specific patterns."""
    
    def __init__(self):
        self.user_profiles = {}
        self.merchant_profiles = {}
        self.device_profiles = {}
        self.location_risk_scores = {}
        
    def engineer_advanced_features(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Engineer advanced fraud detection features."""
        
        features = {}
        
        # 1. BEHAVIORAL DEVIATION FEATURES (High Impact)
        features.update(self._behavioral_deviation_features(transaction))
        
        # 2. NETWORK ANALYSIS FEATURES (High Impact)
        features.update(self._network_analysis_features(transaction))
        
        # 3. TEMPORAL ANOMALY FEATURES (Medium Impact)
        features.update(self._temporal_anomaly_features(transaction))
        
        # 4. AMOUNT PATTERN FEATURES (High Impact)
        features.update(self._amount_pattern_features(transaction))
        
        # 5. DEVICE FINGERPRINTING FEATURES (Medium Impact)
        features.update(self._device_fingerprinting_features(transaction))
        
        # 6. GEOGRAPHIC RISK FEATURES (High Impact)
        features.update(self._geographic_risk_features(transaction))
        
        # 7. SEQUENCE PATTERN FEATURES (High Impact)
        features.update(self._sequence_pattern_features(transaction))
        
        return features
    
    def _behavioral_deviation_features(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Extract behavioral deviation features."""
        
        user_id = tx.get('user_id')
        user_profile = self.user_profiles.get(user_id, {})
        
        features = {}
        
        # Amount deviation from user's normal behavior
        user_avg_amount = user_profile.get('avg_amount', tx['amount'])
        user_std_amount = user_profile.get('std_amount', 0)
        
        if user_std_amount > 0:
            features['amount_zscore'] = (tx['amount'] - user_avg_amount) / user_std_amount
        else:
            features['amount_zscore'] = 0
        
        # Time deviation from user's normal behavior
        user_avg_hour = user_profile.get('avg_transaction_hour', 12)
        tx_hour = datetime.fromisoformat(tx['timestamp']).hour
        features['hour_deviation'] = min(abs(tx_hour - user_avg_hour), 24 - abs(tx_hour - user_avg_hour))
        
        # Frequency deviation
        user_avg_freq = user_profile.get('avg_daily_transactions', 1)
        features['frequency_multiplier'] = tx.get('velocity_24h', 1) / max(user_avg_freq, 1)
        
        # Payment method deviation
        user_common_payment = user_profile.get('most_common_payment_method', tx['payment_method'])
        features['payment_method_deviation'] = int(tx['payment_method'] != user_common_payment)
        
        # Location deviation
        user_common_country = user_profile.get('most_common_country', tx['country'])
        features['location_deviation'] = int(tx['country'] != user_common_country)
        
        return features
    
    def _network_analysis_features(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Extract network-based fraud indicators."""
        
        features = {}
        
        # Device sharing patterns
        device_id = tx.get('device_id')
        device_profile = self.device_profiles.get(device_id, {})
        
        features['device_user_count'] = len(device_profile.get('associated_users', []))
        features['device_velocity_24h'] = device_profile.get('transactions_24h', 0)
        
        # IP address risk scoring
        ip_address = tx.get('ip_address', '')
        ip_hash = hashlib.md5(ip_address.encode()).hexdigest()[:8]
        
        # Simulate IP risk database lookup
        features['ip_risk_score'] = self._get_ip_risk_score(ip_address)
        features['ip_transaction_count'] = self._get_ip_transaction_count(ip_hash)
        
        # Email domain analysis (if available)
        if 'email' in tx:
            email_domain = tx['email'].split('@')[-1] if '@' in tx['email'] else ''
            features['email_domain_risk'] = self._get_email_domain_risk(email_domain)
        
        return features
    
    def _temporal_anomaly_features(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal anomaly features."""
        
        features = {}
        dt = datetime.fromisoformat(tx['timestamp'])
        
        # Time-based risk factors
        features['is_unusual_hour'] = int(dt.hour < 6 or dt.hour > 22)
        features['is_holiday'] = int(self._is_holiday(dt))
        features['is_weekend'] = int(dt.weekday() >= 5)
        
        # Velocity patterns
        velocity_1h = tx.get('velocity_1h', 0)
        velocity_24h = tx.get('velocity_24h', 0)
        
        features['velocity_acceleration'] = velocity_1h / max(velocity_24h / 24, 1)
        features['velocity_burst_indicator'] = int(velocity_1h > 5 and velocity_24h < 10)
        
        # Time since last transaction patterns
        features['rapid_succession'] = int(velocity_1h > 3)
        features['dormant_reactivation'] = int(
            tx.get('account_age_days', 0) > 30 and velocity_24h == 1
        )
        
        return features
    
    def _amount_pattern_features(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Extract amount-based fraud patterns."""
        
        features = {}
        amount = tx['amount']
        
        # Round number patterns (fraudsters often use round numbers)
        features['is_round_amount'] = int(amount % 1 == 0)
        features['is_very_round'] = int(amount % 100 == 0)
        features['amount_digit_variance'] = self._calculate_digit_variance(amount)
        
        # Amount range patterns
        features['is_micro_transaction'] = int(amount < 10)
        features['is_large_transaction'] = int(amount > 1000)
        features['is_testing_amount'] = int(amount in [1, 2, 3, 5, 10, 100])
        
        # Statistical amount features
        features['amount_log'] = np.log1p(amount)
        features['amount_sqrt'] = np.sqrt(amount)
        
        # Benford's law deviation (first digit distribution)
        first_digit = int(str(int(amount))[0])
        expected_benford = np.log10(1 + 1/first_digit)
        features['benford_deviation'] = abs(0.1 - expected_benford)  # Simplified
        
        return features
    
    def _device_fingerprinting_features(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Extract device fingerprinting features."""
        
        features = {}
        device_data = tx.get('device_data', {})
        
        # Device consistency features
        features['device_os_encoded'] = self._encode_device_os(device_data.get('os', 'unknown'))
        features['device_browser_encoded'] = self._encode_browser(device_data.get('browser', 'unknown'))
        
        # Device risk indicators
        features['is_new_device'] = int(device_data.get('is_new_device', False))
        features['device_fingerprint_hash'] = self._hash_device_fingerprint(device_data)
        
        # Mobile vs desktop patterns
        features['is_mobile'] = int(device_data.get('is_mobile', False))
        features['mobile_fraud_risk'] = self._get_mobile_fraud_risk(device_data)
        
        return features
    
    def _geographic_risk_features(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Extract geographic risk features."""
        
        features = {}
        country = tx.get('country', 'US')
        
        # Country risk scoring
        features['country_risk_score'] = self._get_country_risk_score(country)
        features['is_high_risk_country'] = int(features['country_risk_score'] > 0.7)
        
        # Geographic consistency
        geo_data = tx.get('geographic_data', {})
        ip_country = geo_data.get('ip_country', country)
        billing_country = geo_data.get('billing_country', country)
        
        features['geo_ip_country_match'] = int(ip_country == country)
        features['geo_billing_country_match'] = int(billing_country == country)
        features['geo_consistency_score'] = sum([
            features['geo_ip_country_match'],
            features['geo_billing_country_match']
        ]) / 2
        
        # Distance-based features
        distance_km = geo_data.get('distance_km', 0)
        features['geographic_distance_log'] = np.log1p(distance_km)
        features['is_impossible_travel'] = int(distance_km > 1000 and tx.get('velocity_1h', 0) > 0)
        
        return features
    
    def _sequence_pattern_features(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sequence pattern features."""
        
        features = {}
        user_id = tx.get('user_id')
        
        # Transaction sequence patterns
        user_profile = self.user_profiles.get(user_id, {})
        last_transactions = user_profile.get('last_5_transactions', [])
        
        if last_transactions:
            # Amount sequence patterns
            amounts = [t.get('amount', 0) for t in last_transactions]
            features['amount_sequence_trend'] = self._calculate_trend(amounts)
            features['amount_sequence_variance'] = np.var(amounts) if amounts else 0
            
            # Time interval patterns
            intervals = self._calculate_time_intervals(last_transactions)
            features['interval_consistency'] = np.std(intervals) if len(intervals) > 1 else 0
            
            # Merchant sequence patterns
            merchants = [t.get('merchant_id', '') for t in last_transactions]
            features['merchant_diversity'] = len(set(merchants))
            features['merchant_repetition_rate'] = 1 - features['merchant_diversity'] / max(len(merchants), 1)
        
        return features
    
    # Helper methods
    def _get_ip_risk_score(self, ip_address: str) -> float:
        """Get IP address risk score (simulated)."""
        # In production, this would query a threat intelligence database
        ip_hash = hash(ip_address) % 100
        if ip_hash < 5:  # 5% high risk
            return 0.8 + (ip_hash / 100)
        elif ip_hash < 15:  # 10% medium risk
            return 0.4 + (ip_hash / 200)
        else:  # 85% low risk
            return ip_hash / 500
    
    def _get_country_risk_score(self, country: str) -> float:
        """Get country-based fraud risk score."""
        high_risk_countries = ['XX', 'YY', 'ZZ']  # Placeholder high-risk countries
        medium_risk_countries = ['CN', 'RU', 'BR', 'IN']
        
        if country in high_risk_countries:
            return 0.8
        elif country in medium_risk_countries:
            return 0.4
        else:
            return 0.1
    
    def _calculate_digit_variance(self, amount: float) -> float:
        """Calculate variance in digits of amount."""
        digits = [int(d) for d in str(int(amount)) if d.isdigit()]
        return np.var(digits) if len(digits) > 1 else 0
    
    def _hash_device_fingerprint(self, device_data: Dict[str, Any]) -> int:
        """Create a hash of device fingerprint."""
        fingerprint = f"{device_data.get('os', '')}{device_data.get('browser', '')}"
        return hash(fingerprint) % 10000
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a sequence of values."""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _calculate_time_intervals(self, transactions: List[Dict[str, Any]]) -> List[float]:
        """Calculate time intervals between transactions."""
        intervals = []
        for i in range(1, len(transactions)):
            prev_time = datetime.fromisoformat(transactions[i-1]['timestamp'])
            curr_time = datetime.fromisoformat(transactions[i]['timestamp'])
            interval_hours = (curr_time - prev_time).total_seconds() / 3600
            intervals.append(interval_hours)
        return intervals
    
    # Additional helper methods would be implemented here...
    def _get_ip_transaction_count(self, ip_hash: str) -> int:
        return hash(ip_hash) % 50
    
    def _get_email_domain_risk(self, domain: str) -> float:
        suspicious_domains = ['tempmail.com', '10minutemail.com']
        return 0.8 if domain in suspicious_domains else 0.1
    
    def _is_holiday(self, dt: datetime) -> bool:
        # Simplified holiday detection
        return dt.month == 12 and dt.day == 25  # Christmas
    
    def _encode_device_os(self, os: str) -> int:
        os_mapping = {'iOS': 1, 'Android': 2, 'Windows': 3, 'macOS': 4, 'Linux': 5}
        return os_mapping.get(os, 0)
    
    def _encode_browser(self, browser: str) -> int:
        browser_mapping = {'Chrome': 1, 'Safari': 2, 'Firefox': 3, 'Edge': 4}
        return browser_mapping.get(browser, 0)
    
    def _get_mobile_fraud_risk(self, device_data: Dict[str, Any]) -> float:
        # Mobile devices have different fraud patterns
        if device_data.get('is_mobile', False):
            return 0.3  # Slightly higher base risk for mobile
        return 0.1
