import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np
from faker import Faker

fake = Faker()

class SyntheticDataGenerator:
    """Generate synthetic transaction data for testing the fraud detection system"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        fake.seed_instance(seed)
        
        self.merchant_categories = [
            "grocery", "gas", "restaurant", "online", "retail", "pharmacy",
            "entertainment", "travel", "utilities", "insurance", "gambling",
            "adult", "cryptocurrency", "money_transfer", "cash_advance"
        ]
        
        self.countries = [
            "US", "CA", "GB", "DE", "FR", "JP", "AU", "BR", "IN", "CN",
            "RU", "NG", "PK", "BD", "MM"  # Last few are higher risk
        ]
        
        self.high_risk_categories = ["gambling", "adult", "cryptocurrency", "money_transfer", "cash_advance"]
        self.high_risk_countries = ["RU", "NG", "PK", "BD", "MM"]
        
    def generate_normal_transaction(self, user_id: str = None) -> Dict[str, Any]:
        """Generate a normal (non-fraudulent) transaction"""
        if not user_id:
            user_id = f"user_{random.randint(1000, 9999)}"
        
        # Normal transaction patterns
        amount = max(5.0, np.random.lognormal(mean=3.5, sigma=1.2))  # Typical amounts $5-$500
        
        # Business hours more likely
        if random.random() < 0.8:
            hour = random.randint(8, 20)
        else:
            hour = random.randint(0, 23)
        
        # Create timestamp
        days_ago = random.randint(0, 30)
        base_time = datetime.utcnow() - timedelta(days=days_ago)
        timestamp = base_time.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))
        
        # Normal merchant categories
        merchant_category = random.choice([cat for cat in self.merchant_categories if cat not in self.high_risk_categories])
        
        # Domestic transactions more common
        country = "US" if random.random() < 0.85 else random.choice([c for c in self.countries if c not in self.high_risk_countries])
        
        transaction = {
            "transaction_id": str(uuid.uuid4()),
            "user_id": user_id,
            "amount": round(amount, 2),
            "currency": "USD",
            "transaction_type": random.choice(["purchase", "withdrawal"]),
            "merchant_id": f"merchant_{random.randint(1000, 9999)}",
            "merchant_category": merchant_category,
            "location": {
                "country": country,
                "city": fake.city(),
                "latitude": fake.latitude(),
                "longitude": fake.longitude()
            },
            "timestamp": timestamp.isoformat(),
            "device_info": {
                "type": random.choice(["mobile", "desktop", "tablet"]),
                "os": random.choice(["iOS", "Android", "Windows", "macOS"]),
                "browser": random.choice(["Chrome", "Safari", "Firefox", "Edge"]),
                "user_agent": fake.user_agent()
            },
            "ip_address": fake.ipv4(),
            "status": "pending",
            "metadata": {
                "is_synthetic": True,
                "fraud_label": False,
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
        return transaction
    
    def generate_fraudulent_transaction(self, user_id: str = None, fraud_type: str = "random") -> Dict[str, Any]:
        """Generate a fraudulent transaction with specific fraud patterns"""
        if not user_id:
            user_id = f"user_{random.randint(1000, 9999)}"
        
        # Start with normal transaction
        transaction = self.generate_normal_transaction(user_id)
        transaction["metadata"]["fraud_label"] = True
        
        # Apply fraud patterns based on type
        if fraud_type == "high_amount" or (fraud_type == "random" and random.random() < 0.3):
            # Unusually high amount
            transaction["amount"] = round(random.uniform(5000, 50000), 2)
        
        if fraud_type == "unusual_time" or (fraud_type == "random" and random.random() < 0.4):
            # Unusual time (late night/early morning)
            hour = random.choice([0, 1, 2, 3, 4, 5, 23])
            timestamp = datetime.fromisoformat(transaction["timestamp"])
            timestamp = timestamp.replace(hour=hour)
            transaction["timestamp"] = timestamp.isoformat()
        
        if fraud_type == "high_risk_location" or (fraud_type == "random" and random.random() < 0.3):
            # High risk country
            country = random.choice(self.high_risk_countries)
            transaction["location"]["country"] = country
            transaction["location"]["city"] = fake.city()
        
        if fraud_type == "high_risk_merchant" or (fraud_type == "random" and random.random() < 0.25):
            # High risk merchant category
            transaction["merchant_category"] = random.choice(self.high_risk_categories)
        
        if fraud_type == "no_device_info" or (fraud_type == "random" and random.random() < 0.3):
            # Missing device information (suspicious)
            transaction["device_info"] = None
        
        if fraud_type == "round_amount" or (fraud_type == "random" and random.random() < 0.2):
            # Suspiciously round amounts
            base_amount = random.choice([1000, 2000, 5000, 10000])
            transaction["amount"] = float(base_amount)
        
        if fraud_type == "weekend_fraud" or (fraud_type == "random" and random.random() < 0.2):
            # Weekend fraud pattern
            timestamp = datetime.fromisoformat(transaction["timestamp"])
            days_to_add = (5 - timestamp.weekday()) % 7  # Make it Saturday
            if days_to_add == 0:
                days_to_add = 1  # Make it Sunday if already Saturday
            timestamp = timestamp + timedelta(days=days_to_add)
            transaction["timestamp"] = timestamp.isoformat()
        
        # Add fraud-specific metadata
        transaction["metadata"]["fraud_type"] = fraud_type
        transaction["metadata"]["fraud_indicators"] = self._identify_fraud_indicators(transaction)
        
        return transaction
    
    def generate_high_velocity_fraud(self, user_id: str, count: int = 5) -> List[Dict[str, Any]]:
        """Generate multiple transactions for velocity-based fraud"""
        transactions = []
        base_time = datetime.utcnow()
        
        for i in range(count):
            # Transactions within a short time window
            timestamp = base_time + timedelta(minutes=random.randint(0, 30))
            
            transaction = self.generate_fraudulent_transaction(user_id, "high_velocity")
            transaction["timestamp"] = timestamp.isoformat()
            transaction["amount"] = round(random.uniform(500, 2000), 2)
            transaction["metadata"]["fraud_type"] = "high_velocity"
            transaction["metadata"]["velocity_group"] = f"velocity_group_{uuid.uuid4().hex[:8]}"
            
            transactions.append(transaction)
        
        return transactions
    
    def generate_user_transaction_history(self, user_id: str, days: int = 30, fraud_rate: float = 0.1) -> List[Dict[str, Any]]:
        """Generate transaction history for a user over specified days"""
        transactions = []
        
        # Determine user's typical behavior
        user_profile = self._generate_user_profile(user_id)
        
        # Generate transactions over time period
        total_transactions = random.randint(10, 100)
        
        for _ in range(total_transactions):
            is_fraud = random.random() < fraud_rate
            
            if is_fraud:
                transaction = self.generate_fraudulent_transaction(user_id)
            else:
                transaction = self._generate_user_normal_transaction(user_id, user_profile)
            
            # Randomize timestamp within the period
            days_ago = random.randint(0, days)
            timestamp = datetime.utcnow() - timedelta(days=days_ago, 
                                                    hours=random.randint(0, 23),
                                                    minutes=random.randint(0, 59))
            transaction["timestamp"] = timestamp.isoformat()
            
            transactions.append(transaction)
        
        # Sort by timestamp
        transactions.sort(key=lambda x: x["timestamp"])
        
        return transactions
    
    def _generate_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Generate a consistent user profile for transaction generation"""
        random.seed(hash(user_id))  # Consistent profile per user
        
        profile = {
            "avg_amount": random.uniform(50, 500),
            "preferred_categories": random.sample(
                [cat for cat in self.merchant_categories if cat not in self.high_risk_categories], 
                random.randint(2, 4)
            ),
            "typical_hours": list(range(random.randint(7, 10), random.randint(18, 22))),
            "home_country": random.choice(["US", "CA", "GB", "DE", "FR"]),
            "device_consistency": random.random() > 0.2  # 80% have consistent device info
        }
        
        random.seed()  # Reset seed
        return profile
    
    def _generate_user_normal_transaction(self, user_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a normal transaction consistent with user profile"""
        transaction = self.generate_normal_transaction(user_id)
        
        # Apply user profile patterns
        transaction["amount"] = max(5.0, random.normalvariate(profile["avg_amount"], profile["avg_amount"] * 0.3))
        transaction["amount"] = round(transaction["amount"], 2)
        
        # Use preferred categories
        if random.random() < 0.7:
            transaction["merchant_category"] = random.choice(profile["preferred_categories"])
        
        # Use typical hours
        if random.random() < 0.8:
            hour = random.choice(profile["typical_hours"])
            timestamp = datetime.fromisoformat(transaction["timestamp"])
            timestamp = timestamp.replace(hour=hour)
            transaction["timestamp"] = timestamp.isoformat()
        
        # Use home country
        if random.random() < 0.9:
            transaction["location"]["country"] = profile["home_country"]
        
        # Device consistency
        if not profile["device_consistency"] and random.random() < 0.3:
            transaction["device_info"] = None
        
        return transaction
    
    def _identify_fraud_indicators(self, transaction: Dict[str, Any]) -> List[str]:
        """Identify fraud indicators in a transaction"""
        indicators = []
        
        if transaction["amount"] > 5000:
            indicators.append("high_amount")
        
        timestamp = datetime.fromisoformat(transaction["timestamp"])
        if timestamp.hour < 6 or timestamp.hour > 22:
            indicators.append("unusual_time")
        
        if transaction["location"]["country"] in self.high_risk_countries:
            indicators.append("high_risk_location")
        
        if transaction["merchant_category"] in self.high_risk_categories:
            indicators.append("high_risk_merchant")
        
        if not transaction["device_info"]:
            indicators.append("no_device_info")
        
        if transaction["amount"] % 100 == 0 and transaction["amount"] > 1000:
            indicators.append("round_amount")
        
        if timestamp.weekday() >= 5:  # Weekend
            indicators.append("weekend_transaction")
        
        return indicators
    
    def generate_test_dataset(self, size: int = 1000, fraud_rate: float = 0.15) -> List[Dict[str, Any]]:
        """Generate a complete test dataset with mixed normal and fraudulent transactions"""
        transactions = []
        fraud_count = int(size * fraud_rate)
        normal_count = size - fraud_count
        
        # Generate normal transactions
        for _ in range(normal_count):
            transactions.append(self.generate_normal_transaction())
        
        # Generate fraudulent transactions
        fraud_types = ["high_amount", "unusual_time", "high_risk_location", "high_risk_merchant", "random"]
        for _ in range(fraud_count):
            fraud_type = random.choice(fraud_types)
            transactions.append(self.generate_fraudulent_transaction(fraud_type=fraud_type))
        
        # Shuffle the dataset
        random.shuffle(transactions)
        
        return transactions
    
    def generate_real_time_stream(self, duration_minutes: int = 60, tps: float = 10.0, fraud_rate: float = 0.1):
        """Generate a real-time stream of transactions (generator)"""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        transaction_interval = 1.0 / tps  # Seconds between transactions
        next_transaction_time = start_time
        
        while datetime.utcnow() < end_time:
            if datetime.utcnow() >= next_transaction_time:
                is_fraud = random.random() < fraud_rate
                
                if is_fraud:
                    transaction = self.generate_fraudulent_transaction()
                else:
                    transaction = self.generate_normal_transaction()
                
                # Set current timestamp
                transaction["timestamp"] = datetime.utcnow().isoformat()
                
                yield transaction
                
                # Schedule next transaction
                next_transaction_time += timedelta(seconds=transaction_interval)
            
            # Small sleep to avoid busy waiting
            import asyncio
            asyncio.sleep(0.01)

# Convenience functions
def generate_sample_transaction(fraud: bool = False) -> Dict[str, Any]:
    """Generate a single sample transaction"""
    generator = SyntheticDataGenerator()
    if fraud:
        return generator.generate_fraudulent_transaction()
    else:
        return generator.generate_normal_transaction()

def generate_test_batch(size: int = 100, fraud_rate: float = 0.15) -> List[Dict[str, Any]]:
    """Generate a batch of test transactions"""
    generator = SyntheticDataGenerator()
    return generator.generate_test_dataset(size, fraud_rate)