import asyncio
import csv
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import random
import numpy as np
from pathlib import Path

from fraud_detection_system import FraudDetectionSystem
from core.transaction_processor import ProcessingResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationTestCase:
    """Represents a single validation test case."""
    test_id: str
    transaction_data: Dict[str, Any]
    expected_decision: str  # approve, decline, review
    expected_risk_range: Tuple[float, float]  # min, max risk score
    fraud_label: bool  # ground truth
    test_category: str  # legitimate, card_testing, account_takeover, etc.
    description: str
    priority: str = "normal"


@dataclass
class ValidationResult:
    """Results from validating a single test case."""
    test_id: str
    transaction_id: str
    
    # Expected vs Actual
    expected_decision: str
    actual_decision: str
    expected_risk_range: Tuple[float, float]
    actual_risk_score: float
    fraud_label: bool
    
    # Performance Metrics
    processing_time_ms: float
    confidence_score: float
    
    # Agent Analysis
    agents_participated: int
    pattern_agent_decision: Optional[str]
    risk_agent_decision: Optional[str]
    coordination_decision: Optional[str]
    
    # Validation Results
    decision_correct: bool
    risk_score_in_range: bool
    false_positive: bool
    false_negative: bool
    
    # Additional Metadata
    test_category: str
    reasoning: List[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        return {
            'test_id': self.test_id,
            'transaction_id': self.transaction_id,
            'test_category': self.test_category,
            'fraud_label': self.fraud_label,
            
            # Expected vs Actual
            'expected_decision': self.expected_decision,
            'actual_decision': self.actual_decision,
            'expected_risk_min': self.expected_risk_range[0],
            'expected_risk_max': self.expected_risk_range[1],
            'actual_risk_score': self.actual_risk_score,
            
            # Performance
            'processing_time_ms': self.processing_time_ms,
            'confidence_score': self.confidence_score,
            'agents_participated': self.agents_participated,
            
            # Individual Agent Decisions
            'pattern_agent_decision': self.pattern_agent_decision or 'N/A',
            'risk_agent_decision': self.risk_agent_decision or 'N/A',
            'coordination_decision': self.coordination_decision or 'N/A',
            
            # Validation Results
            'decision_correct': self.decision_correct,
            'risk_score_in_range': self.risk_score_in_range,
            'false_positive': self.false_positive,
            'false_negative': self.false_negative,
            
            # Overall Pass/Fail
            'test_passed': self.decision_correct and self.risk_score_in_range and not self.false_positive and not self.false_negative,
            
            # Metadata
            'reasoning_summary': '; '.join(self.reasoning[:3]) if self.reasoning else 'N/A',
            'timestamp': self.timestamp.isoformat()
        }


class TestDataGenerator:
    """Generates diverse test cases for validation."""
    
    def __init__(self):
        self.countries = ['US', 'GB', 'CA', 'FR', 'DE', 'AU', 'JP', 'BR', 'IN', 'CN']
        self.payment_methods = ['credit_card', 'debit_card', 'bank_transfer', 'paypal', 'apple_pay']
        self.device_types = ['desktop', 'mobile', 'tablet']
        self.currencies = ['USD', 'EUR', 'GBP', 'CAD', 'AUD']
        
    def generate_test_cases(self, num_cases: int = 100) -> List[ValidationTestCase]:
        """Generate comprehensive test cases."""
        
        test_cases = []
        
        # 40% legitimate transactions
        legitimate_cases = int(num_cases * 0.4)
        test_cases.extend(self._generate_legitimate_transactions(legitimate_cases))
        
        # 20% card testing fraud
        card_testing_cases = int(num_cases * 0.2)
        test_cases.extend(self._generate_card_testing_fraud(card_testing_cases))
        
        # 15% account takeover fraud
        takeover_cases = int(num_cases * 0.15)
        test_cases.extend(self._generate_account_takeover_fraud(takeover_cases))
        
        # 10% high-value fraud
        high_value_cases = int(num_cases * 0.1)
        test_cases.extend(self._generate_high_value_fraud(high_value_cases))
        
        # 10% velocity fraud
        velocity_cases = int(num_cases * 0.1)
        test_cases.extend(self._generate_velocity_fraud(velocity_cases))
        
        # 5% edge cases
        edge_cases = num_cases - len(test_cases)
        test_cases.extend(self._generate_edge_cases(edge_cases))
        
        # Shuffle to randomize order
        random.shuffle(test_cases)
        
        return test_cases
    
    def _generate_legitimate_transactions(self, count: int) -> List[ValidationTestCase]:
        """Generate legitimate transaction test cases."""
        
        cases = []
        for i in range(count):
            transaction_data = {
                "transaction_id": f"legit_{i:04d}",
                "user_id": f"user_legit_{random.randint(1000, 9999)}",
                "amount": round(random.uniform(10, 500), 2),
                "currency": random.choice(self.currencies),
                "transaction_type": "purchase",
                "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 60))).isoformat(),
                "country": random.choice(self.countries[:5]),  # Low-risk countries
                "payment_method": random.choice(self.payment_methods[:3]),  # Safe payment methods
                "device_id": f"device_legit_{random.randint(100, 999)}",
                "device_type": random.choice(self.device_types),
                "ip_address": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                "is_mobile": random.choice([True, False]),
                "velocity_1h": random.randint(0, 3),
                "velocity_24h": random.randint(1, 8),
                "account_age_days": random.randint(30, 1000),
                "is_first_transaction": False,
                "session_duration": random.randint(120, 1800),  # 2-30 minutes
                "pages_visited": random.randint(3, 15)
            }
            
            cases.append(ValidationTestCase(
                test_id=f"VAL_LEGIT_{i:04d}",
                transaction_data=transaction_data,
                expected_decision="approve",
                expected_risk_range=(0.0, 0.3),
                fraud_label=False,
                test_category="legitimate",
                description="Normal legitimate transaction"
            ))
        
        return cases
    
    def _generate_card_testing_fraud(self, count: int) -> List[ValidationTestCase]:
        """Generate card testing fraud test cases."""
        
        cases = []
        for i in range(count):
            transaction_data = {
                "transaction_id": f"card_test_{i:04d}",
                "user_id": f"user_test_{random.randint(1000, 9999)}",
                "amount": round(random.uniform(0.01, 5.0), 2),  # Very small amounts
                "currency": "USD",
                "transaction_type": "purchase",
                "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 10))).isoformat(),
                "country": random.choice(self.countries),
                "payment_method": "credit_card",
                "device_id": f"device_test_{random.randint(100, 999)}",
                "device_type": random.choice(self.device_types),
                "ip_address": f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}",
                "is_mobile": random.choice([True, False]),
                "velocity_1h": random.randint(5, 20),  # High velocity
                "velocity_24h": random.randint(20, 100),  # Very high velocity
                "account_age_days": random.randint(0, 7),  # New account
                "is_first_transaction": random.choice([True, False]),
                "session_duration": random.randint(10, 60),  # Very short sessions
                "pages_visited": random.randint(1, 3)  # Minimal browsing
            }
            
            cases.append(ValidationTestCase(
                test_id=f"VAL_CARD_TEST_{i:04d}",
                transaction_data=transaction_data,
                expected_decision="decline",
                expected_risk_range=(0.7, 1.0),
                fraud_label=True,
                test_category="card_testing",
                description="Card testing with small amounts and high velocity"
            ))
        
        return cases
    
    def _generate_account_takeover_fraud(self, count: int) -> List[ValidationTestCase]:
        """Generate account takeover fraud test cases."""
        
        cases = []
        for i in range(count):
            transaction_data = {
                "transaction_id": f"takeover_{i:04d}",
                "user_id": f"user_takeover_{random.randint(1000, 9999)}",
                "amount": round(random.uniform(100, 2000), 2),
                "currency": random.choice(self.currencies),
                "transaction_type": random.choice(["purchase", "transfer"]),
                "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 30))).isoformat(),
                "country": random.choice(self.countries[5:]),  # Different country
                "payment_method": random.choice(self.payment_methods),
                "device_id": f"device_new_{random.randint(100, 999)}",  # New device
                "device_type": random.choice(self.device_types),
                "ip_address": f"203.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
                "is_mobile": random.choice([True, False]),
                "velocity_1h": random.randint(1, 5),
                "velocity_24h": random.randint(3, 15),
                "account_age_days": random.randint(90, 500),  # Established account
                "is_first_transaction": False,
                "session_duration": random.randint(30, 180),  # Quick session
                "pages_visited": random.randint(1, 5)  # Direct to target
            }
            
            cases.append(ValidationTestCase(
                test_id=f"VAL_TAKEOVER_{i:04d}",
                transaction_data=transaction_data,
                expected_decision="decline",
                expected_risk_range=(0.6, 0.9),
                fraud_label=True,
                test_category="account_takeover",
                description="Account takeover with new device and location"
            ))
        
        return cases
    
    def _generate_high_value_fraud(self, count: int) -> List[ValidationTestCase]:
        """Generate high-value fraud test cases."""
        
        cases = []
        for i in range(count):
            transaction_data = {
                "transaction_id": f"high_value_{i:04d}",
                "user_id": f"user_high_{random.randint(1000, 9999)}",
                "amount": round(random.uniform(5000, 50000), 2),  # Very high amounts
                "currency": random.choice(self.currencies),
                "transaction_type": random.choice(["transfer", "purchase"]),
                "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 120))).isoformat(),
                "country": random.choice(self.countries),
                "payment_method": random.choice(["bank_transfer", "wire_transfer"]),
                "device_id": f"device_high_{random.randint(100, 999)}",
                "device_type": random.choice(self.device_types),
                "ip_address": f"172.{random.randint(16, 31)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
                "is_mobile": random.choice([True, False]),
                "velocity_1h": random.randint(0, 2),
                "velocity_24h": random.randint(1, 5),
                "account_age_days": random.randint(1, 30),  # Relatively new account
                "is_first_transaction": random.choice([True, False]),
                "session_duration": random.randint(60, 300),
                "pages_visited": random.randint(2, 8)
            }
            
            cases.append(ValidationTestCase(
                test_id=f"VAL_HIGH_VALUE_{i:04d}",
                transaction_data=transaction_data,
                expected_decision="review",  # High value should trigger review
                expected_risk_range=(0.4, 0.8),
                fraud_label=True,
                test_category="high_value_fraud",
                description="High-value transaction requiring review"
            ))
        
        return cases
    
    def _generate_velocity_fraud(self, count: int) -> List[ValidationTestCase]:
        """Generate velocity-based fraud test cases."""
        
        cases = []
        for i in range(count):
            transaction_data = {
                "transaction_id": f"velocity_{i:04d}",
                "user_id": f"user_velocity_{random.randint(1000, 9999)}",
                "amount": round(random.uniform(50, 1000), 2),
                "currency": random.choice(self.currencies),
                "transaction_type": "purchase",
                "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 5))).isoformat(),
                "country": random.choice(self.countries),
                "payment_method": random.choice(self.payment_methods),
                "device_id": f"device_velocity_{random.randint(100, 999)}",
                "device_type": random.choice(self.device_types),
                "ip_address": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                "is_mobile": random.choice([True, False]),
                "velocity_1h": random.randint(15, 50),  # Extremely high velocity
                "velocity_24h": random.randint(50, 200),
                "account_age_days": random.randint(30, 200),
                "is_first_transaction": False,
                "session_duration": random.randint(20, 120),
                "pages_visited": random.randint(1, 4)
            }
            
            cases.append(ValidationTestCase(
                test_id=f"VAL_VELOCITY_{i:04d}",
                transaction_data=transaction_data,
                expected_decision="decline",
                expected_risk_range=(0.8, 1.0),
                fraud_label=True,
                test_category="velocity_fraud",
                description="High-velocity fraud pattern"
            ))
        
        return cases
    
    def _generate_edge_cases(self, count: int) -> List[ValidationTestCase]:
        """Generate edge case test scenarios."""
        
        cases = []
        for i in range(count):
            # Random edge case type
            edge_type = random.choice(['missing_data', 'boundary_values', 'unusual_patterns'])
            
            if edge_type == 'missing_data':
                transaction_data = {
                    "transaction_id": f"edge_missing_{i:04d}",
                    "user_id": f"user_edge_{random.randint(1000, 9999)}",
                    "amount": round(random.uniform(10, 500), 2),
                    "currency": "USD",
                    "transaction_type": "purchase",
                    "timestamp": datetime.now().isoformat(),
                    "country": "US",
                    "payment_method": "credit_card",
                    # Missing optional fields to test robustness
                }
                expected_decision = "review"
                expected_risk = (0.3, 0.7)
                
            elif edge_type == 'boundary_values':
                transaction_data = {
                    "transaction_id": f"edge_boundary_{i:04d}",
                    "user_id": f"user_edge_{random.randint(1000, 9999)}",
                    "amount": 0.01,  # Minimum amount
                    "currency": "USD",
                    "transaction_type": "purchase",
                    "timestamp": datetime.now().isoformat(),
                    "country": "US",
                    "payment_method": "credit_card",
                    "device_id": f"device_edge_{i}",
                    "velocity_1h": 0,
                    "velocity_24h": 0,
                    "account_age_days": 0,  # Brand new account
                    "is_first_transaction": True
                }
                expected_decision = "review"
                expected_risk = (0.4, 0.8)
                
            else:  # unusual_patterns
                transaction_data = {
                    "transaction_id": f"edge_unusual_{i:04d}",
                    "user_id": f"user_edge_{random.randint(1000, 9999)}",
                    "amount": 999.99,  # Round amount
                    "currency": "EUR",
                    "transaction_type": "refund",  # Unusual type
                    "timestamp": (datetime.now().replace(hour=3)).isoformat(),  # 3 AM
                    "country": "XX",  # Unknown country
                    "payment_method": "cryptocurrency",
                    "device_id": f"device_edge_{i}",
                    "velocity_1h": 1,
                    "velocity_24h": 1,
                    "account_age_days": 1,
                    "is_first_transaction": False
                }
                expected_decision = "review"
                expected_risk = (0.5, 0.9)
            
            cases.append(ValidationTestCase(
                test_id=f"VAL_EDGE_{edge_type.upper()}_{i:04d}",
                transaction_data=transaction_data,
                expected_decision=expected_decision,
                expected_risk_range=expected_risk,
                fraud_label=edge_type != 'missing_data',  # Most edge cases are suspicious
                test_category=f"edge_case_{edge_type}",
                description=f"Edge case: {edge_type.replace('_', ' ')}"
            ))
        
        return cases


class ValidationRunner:
    """Runs validation tests and generates results."""
    
    def __init__(self, fraud_system: FraudDetectionSystem):
        self.fraud_system = fraud_system
        self.results: List[ValidationResult] = []
        
    async def run_validation_suite(
        self, 
        test_cases: List[ValidationTestCase],
        concurrent_limit: int = 10
    ) -> List[ValidationResult]:
        """Run all validation test cases."""
        
        logger.info(f"Starting validation suite with {len(test_cases)} test cases")
        
        # Process test cases in batches to avoid overwhelming the system
        batch_size = concurrent_limit
        total_batches = (len(test_cases) + batch_size - 1) // batch_size
        
        self.results = []
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(test_cases))
            batch_cases = test_cases[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_cases)} cases)")
            
            # Process batch concurrently
            batch_tasks = [
                self._run_single_test_case(test_case) 
                for test_case in batch_cases
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Test case {batch_cases[i].test_id} failed: {result}")
                    # Create failed result
                    failed_result = ValidationResult(
                        test_id=batch_cases[i].test_id,
                        transaction_id="FAILED",
                        expected_decision=batch_cases[i].expected_decision,
                        actual_decision="ERROR",
                        expected_risk_range=batch_cases[i].expected_risk_range,
                        actual_risk_score=0.0,
                        fraud_label=batch_cases[i].fraud_label,
                        processing_time_ms=0.0,
                        confidence_score=0.0,
                        agents_participated=0,
                        pattern_agent_decision=None,
                        risk_agent_decision=None,
                        coordination_decision=None,
                        decision_correct=False,
                        risk_score_in_range=False,
                        false_positive=False,
                        false_negative=False,
                        test_category=batch_cases[i].test_category,
                        reasoning=[f"Test execution failed: {str(result)}"],
                        timestamp=datetime.now()
                    )
                    self.results.append(failed_result)
                else:
                    self.results.append(result)
            
            # Brief pause between batches
            if batch_idx < total_batches - 1:
                await asyncio.sleep(1)
        
        logger.info(f"Validation suite completed. {len(self.results)} results generated.")
        return self.results
    
    async def _run_single_test_case(self, test_case: ValidationTestCase) -> ValidationResult:
        """Run a single validation test case."""
        
        start_time = time.time()
        
        try:
            # Process transaction through fraud detection system
            transaction_id = await self.fraud_system.process_transaction(
                test_case.transaction_data, 
                test_case.priority
            )
            
            # Get processing result
            processing_result = await self.fraud_system.get_transaction_result(
                transaction_id, 
                timeout=15.0
            )
            
            if not processing_result:
                raise Exception("No processing result received within timeout")
            
            processing_time = time.time() - start_time
            
            # Extract agent decisions from processing result
            pattern_decision = None
            risk_decision = None
            coordination_decision = None
            agents_participated = len(processing_result.agent_decisions)
            
            for agent_decision in processing_result.agent_decisions:
                if 'pattern' in agent_decision.agent_id.lower():
                    pattern_decision = agent_decision.decision_type
                elif 'risk' in agent_decision.agent_id.lower():
                    risk_decision = agent_decision.decision_type
                elif 'coordination' in agent_decision.agent_id.lower():
                    coordination_decision = agent_decision.decision_type
            
            # Validate results
            decision_correct = (processing_result.decision.lower() == test_case.expected_decision.lower())
            
            risk_in_range = (
                test_case.expected_risk_range[0] <= processing_result.risk_score <= test_case.expected_risk_range[1]
            )
            
            # Determine false positives/negatives
            actual_fraud_prediction = processing_result.decision.lower() in ['decline', 'review']
            false_positive = not test_case.fraud_label and actual_fraud_prediction
            false_negative = test_case.fraud_label and not actual_fraud_prediction
            
            # Extract reasoning
            reasoning = []
            for agent_decision in processing_result.agent_decisions:
                reasoning.extend(agent_decision.reasoning[:2])  # Top 2 reasons per agent
            
            return ValidationResult(
                test_id=test_case.test_id,
                transaction_id=transaction_id,
                expected_decision=test_case.expected_decision,
                actual_decision=processing_result.decision,
                expected_risk_range=test_case.expected_risk_range,
                actual_risk_score=processing_result.risk_score,
                fraud_label=test_case.fraud_label,
                processing_time_ms=processing_time * 1000,
                confidence_score=processing_result.confidence,
                agents_participated=agents_participated,
                pattern_agent_decision=pattern_decision,
                risk_agent_decision=risk_decision,
                coordination_decision=coordination_decision,
                decision_correct=decision_correct,
                risk_score_in_range=risk_in_range,
                false_positive=false_positive,
                false_negative=false_negative,
                test_category=test_case.test_category,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error processing test case {test_case.test_id}: {e}")
            raise
    
    def export_results_to_csv(self, output_path: str) -> str:
        """Export validation results to CSV file."""
        
        if not self.results:
            raise ValueError("No validation results to export")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Get all field names from the first result
        fieldnames = list(self.results[0].to_dict().keys())
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                writer.writerow(result.to_dict())
        
        logger.info(f"Validation results exported to {output_file}")
        return str(output_file)
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary statistics from validation results."""
        
        if not self.results:
            return {"error": "No validation results available"}
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.decision_correct and r.risk_score_in_range)
        
        # Calculate metrics by category
        category_stats = {}
        for result in self.results:
            category = result.test_category
            if category not in category_stats:
                category_stats[category] = {
                    'total': 0, 'passed': 0, 'decision_correct': 0, 
                    'risk_in_range': 0, 'false_positives': 0, 'false_negatives': 0
                }
            
            stats = category_stats[category]
            stats['total'] += 1
            if result.decision_correct and result.risk_score_in_range:
                stats['passed'] += 1
            if result.decision_correct:
                stats['decision_correct'] += 1
            if result.risk_score_in_range:
                stats['risk_in_range'] += 1
            if result.false_positive:
                stats['false_positives'] += 1
            if result.false_negative:
                stats['false_negatives'] += 1
        
        # Calculate overall metrics
        total_fp = sum(1 for r in self.results if r.false_positive)
        total_fn = sum(1 for r in self.results if r.false_negative)
        avg_processing_time = np.mean([r.processing_time_ms for r in self.results])
        avg_confidence = np.mean([r.confidence_score for r in self.results])
        
        return {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'decision_accuracy': sum(1 for r in self.results if r.decision_correct) / total_tests,
                'risk_score_accuracy': sum(1 for r in self.results if r.risk_score_in_range) / total_tests,
                'false_positive_rate': total_fp / total_tests,
                'false_negative_rate': total_fn / total_tests,
                'avg_processing_time_ms': avg_processing_time,
                'avg_confidence_score': avg_confidence
            },
            'category_breakdown': {
                category: {
                    'total': stats['total'],
                    'pass_rate': stats['passed'] / stats['total'],
                    'decision_accuracy': stats['decision_correct'] / stats['total'],
                    'risk_accuracy': stats['risk_in_range'] / stats['total'],
                    'false_positive_rate': stats['false_positives'] / stats['total'],
                    'false_negative_rate': stats['false_negatives'] / stats['total']
                }
                for category, stats in category_stats.items()
            },
            'performance_stats': {
                'min_processing_time_ms': min(r.processing_time_ms for r in self.results),
                'max_processing_time_ms': max(r.processing_time_ms for r in self.results),
                'p95_processing_time_ms': np.percentile([r.processing_time_ms for r in self.results], 95),
                'avg_agents_participated': np.mean([r.agents_participated for r in self.results])
            }
        }