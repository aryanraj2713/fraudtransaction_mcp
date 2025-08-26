from __future__ import annotations

import asyncio
import csv
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional


@dataclass
class ValidationTestCase:
    test_id: str
    transaction_data: Dict[str, Any]
    expected_decision: str
    expected_risk_range: Tuple[float, float]
    fraud_label: bool
    test_category: str = "generated"
    description: str = ""
    priority: str = "normal"


class TestDataGenerator:
    def __init__(self) -> None:
        self.countries = ["US", "CA", "UK", "DE", "FR", "JP", "AU", "BR", "IN", "MX", "XX"]
        self.payment_methods = ["credit_card", "debit_card", "paypal", "bank_transfer", "crypto"]
        self.transaction_types = ["purchase", "refund", "transfer", "withdrawal", "deposit"]

    def _base_transaction(self, idx: int) -> Dict[str, Any]:
        amount = round(random.uniform(5.0, 2000.0), 2)
        now = datetime.utcnow()
        tx = {
            "transaction_id": f"val_tx_{idx:05d}",
            "user_id": f"user_{random.randint(10000, 99999)}",
            "amount": amount,
            "currency": "USD",
            "transaction_type": random.choice(self.transaction_types),
            "timestamp": (now - timedelta(minutes=random.randint(0, 120))).isoformat(),
            "country": random.choice(self.countries),
            "payment_method": random.choice(self.payment_methods),
            "device_id": f"device_{random.randint(100, 999)}",
            "ip_address": f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}",
            "velocity_1h": random.randint(0, 5),
            "velocity_24h": random.randint(0, 20),
            "account_age_days": random.randint(1, 3650),
            "is_first_transaction": random.random() < 0.1,
            # Optional fields used by some parts of the system
            "merchant_category": random.choice(["retail", "food", "travel", "electronics", "services"]),
            "device_type": random.choice(["mobile", "desktop"]),
            "is_mobile": random.choice([True, False]),
            "session_duration": random.randint(10, 600),
            "pages_visited": random.randint(1, 15),
            # Add required nested structures expected by validators
            "geographic_data": {
                "ip_country": random.choice(self.countries),
                "billing_country": random.choice(self.countries),
                "shipping_country": random.choice(self.countries),
                "distance_km": random.uniform(1, 8000),
            },
            "device_data": {
                "device_id": f"device_{random.randint(100, 999)}",
                "device_fingerprint": f"fp_{random.randint(100000, 999999)}",
                "is_new_device": random.random() < 0.2,
                "os": random.choice(["iOS", "Android", "Windows", "macOS", "Linux"]),
                "browser": random.choice(["Safari", "Chrome", "Firefox", "Edge"]),
            },
        }
        return tx

    def generate_test_cases(self, num_tests: int = 100) -> List[ValidationTestCase]:
        tests: List[ValidationTestCase] = []
        for i in range(num_tests):
            tx = self._base_transaction(i + 1)

            # Create a mix of legitimate and fraud-like transactions
            if i % 5 == 0:
                # Fraud-like pattern strengthened
                tx.update({
                    "amount": round(random.uniform(4000, 16000), 2),
                    "velocity_1h": random.randint(15, 60),
                    "velocity_24h": random.randint(40, 120),
                    "country": random.choice(["XX", "YY", "ZZ"]),
                    "is_first_transaction": True,
                    "account_age_days": random.randint(1, 7),
                })
                # Ensure nested fields reflect risk
                tx["device_data"]["is_new_device"] = True
                tx["geographic_data"]["ip_country"] = tx["country"]

                expected_decision = "decline"
                expected_risk_range = (0.6, 1.0)
                fraud_label = True
                category = "generated_fraud"
            else:
                expected_decision = "approve"
                expected_risk_range = (0.0, 0.5)
                fraud_label = False
                category = "generated_legitimate"

            tests.append(
                ValidationTestCase(
                    test_id=f"GEN_{i+1:05d}",
                    transaction_data=tx,
                    expected_decision=expected_decision,
                    expected_risk_range=expected_risk_range,
                    fraud_label=fraud_label,
                    test_category=category,
                    description="Auto-generated validation test",
                    priority="high" if fraud_label else "normal",
                )
            )
        return tests


class ValidationRunner:
    def __init__(self, fraud_system) -> None:
        self.fraud_system = fraud_system
        self.results: List[Dict[str, Any]] = []

    async def _run_single_test(self, test: ValidationTestCase) -> Dict[str, Any]:
        tx_id = await self.fraud_system.process_transaction(test.transaction_data)
        result = await self.fraud_system.get_transaction_result(tx_id, timeout=10.0)

        passed_decision = False
        passed_risk = False
        actual_decision = None
        actual_risk = None
        processing_time_ms: Optional[float] = None
        confidence: Optional[float] = None

        if result is not None:
            actual_decision = result.decision
            actual_risk = float(result.risk_score)
            processing_time_ms = float(result.processing_time_ms)
            confidence = float(result.confidence)
            passed_decision = (actual_decision == test.expected_decision)
            passed_risk = (test.expected_risk_range[0] <= actual_risk <= test.expected_risk_range[1])

        outcome = {
            "test_id": test.test_id,
            "category": test.test_category,
            "expected_decision": test.expected_decision,
            "actual_decision": actual_decision,
            "expected_risk_min": test.expected_risk_range[0],
            "expected_risk_max": test.expected_risk_range[1],
            "actual_risk": actual_risk,
            "passed_decision": passed_decision,
            "passed_risk": passed_risk,
            "passed": bool(passed_decision and passed_risk),
            "processing_time_ms": processing_time_ms,
            "confidence": confidence,
        }
        return outcome

    async def run_validation_suite(self, tests: List[ValidationTestCase], concurrent_limit: int = 10) -> List[Dict[str, Any]]:
        # Sequential or limited concurrency to respect agent busy state
        semaphore = asyncio.Semaphore(max(1, concurrent_limit))

        async def guarded(test: ValidationTestCase):
            async with semaphore:
                try:
                    return await self._run_single_test(test)
                except Exception as e:
                    return {
                        "test_id": test.test_id,
                        "category": test.test_category,
                        "error": str(e),
                        "passed": False,
                    }

        self.results = []
        for test in tests:
            self.results.append(await guarded(test))
        return self.results

    def generate_summary_report(self) -> Dict[str, Any]:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get("passed"))
        pass_rate = (passed / total) if total else 0.0

        decisions_ok = sum(1 for r in self.results if r.get("passed_decision"))
        risk_ok = sum(1 for r in self.results if r.get("passed_risk"))

        processing_times = [r.get("processing_time_ms") for r in self.results if isinstance(r.get("processing_time_ms"), (int, float))]
        confidences = [r.get("confidence") for r in self.results if isinstance(r.get("confidence"), (int, float))]

        by_category: Dict[str, Dict[str, Any]] = {}
        for r in self.results:
            cat = r.get("category") or "unknown"
            b = by_category.setdefault(cat, {"total": 0, "passed": 0, "decision_ok": 0, "risk_ok": 0})
            b["total"] += 1
            b["passed"] += 1 if r.get("passed") else 0
            b["decision_ok"] += 1 if r.get("passed_decision") else 0
            b["risk_ok"] += 1 if r.get("passed_risk") else 0

        category_breakdown = {
            c: {
                "total": stats["total"],
                "pass_rate": (stats["passed"] / stats["total"]) if stats["total"] else 0.0,
                "decision_accuracy": (stats["decision_ok"] / stats["total"]) if stats["total"] else 0.0,
                "false_positive_rate": 0.0,  # placeholder without labels mapping
                "false_negative_rate": 0.0,  # placeholder without labels mapping
            }
            for c, stats in by_category.items()
        }

        summary = {
            "summary": {
                "total_tests": total,
                "passed_tests": passed,
                "pass_rate": pass_rate,
                "decision_accuracy": (decisions_ok / total) if total else 0.0,
                "risk_score_accuracy": (risk_ok / total) if total else 0.0,
                "false_positive_rate": 0.0,
                "false_negative_rate": 0.0,
                "avg_processing_time_ms": (sum(processing_times) / len(processing_times)) if processing_times else 0.0,
                "avg_confidence_score": (sum(confidences) / len(confidences)) if confidences else 0.0,
            },
            "performance_stats": {
                "min_processing_time_ms": min(processing_times) if processing_times else 0.0,
                "max_processing_time_ms": max(processing_times) if processing_times else 0.0,
                "p95_processing_time_ms": sorted(processing_times)[int(0.95 * len(processing_times)) - 1] if processing_times else 0.0,
                "avg_agents_participated": 3.0,  # placeholder without deeper introspection
            },
            "category_breakdown": category_breakdown,
        }
        return summary

    def export_results_to_csv(self, output_path: str) -> str:
        fieldnames = [
            "test_id",
            "category",
            "expected_decision",
            "actual_decision",
            "expected_risk_min",
            "expected_risk_max",
            "actual_risk",
            "passed_decision",
            "passed_risk",
            "passed",
            "processing_time_ms",
            "confidence",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow({k: r.get(k) for k in fieldnames})
        return output_path
