#!/usr/bin/env python3
"""
Fraud Detection System Validation Script

This script runs comprehensive validation tests against the fraud detection system
to compare expected vs actual outputs across multiple test scenarios.

Usage:
    python run_validation.py --num-tests 100 --output validation_results.csv
    python run_validation.py --config validation_config.json --concurrent 20
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from validation.validation_framework import TestDataGenerator, ValidationRunner, ValidationTestCase
from fraud_detection_system import FraudDetectionSystem


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def load_custom_test_cases(config_file: str) -> list:
    """Load custom test cases from configuration file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        test_cases = []
        for test_config in config.get('test_cases', []):
            test_case = ValidationTestCase(
                test_id=test_config['test_id'],
                transaction_data=test_config['transaction_data'],
                expected_decision=test_config['expected_decision'],
                expected_risk_range=tuple(test_config['expected_risk_range']),
                fraud_label=test_config['fraud_label'],
                test_category=test_config.get('test_category', 'custom'),
                description=test_config.get('description', 'Custom test case'),
                priority=test_config.get('priority', 'normal')
            )
            test_cases.append(test_case)
        
        return test_cases
    
    except Exception as e:
        logging.error(f"Error loading custom test cases: {e}")
        return []


async def run_validation_suite(args):
    """Main validation suite runner."""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Fraud Detection System Validation Suite")
    
    # Initialize test data generator
    test_generator = TestDataGenerator()
    
    # Generate or load test cases
    if args.config:
        logger.info(f"Loading custom test cases from {args.config}")
        test_cases = load_custom_test_cases(args.config)
        if not test_cases:
            logger.error("No valid test cases loaded from config file")
            return 1
    else:
        logger.info(f"Generating {args.num_tests} test cases")
        test_cases = test_generator.generate_test_cases(args.num_tests)
    
    logger.info(f"Total test cases to execute: {len(test_cases)}")
    
    # Print test case distribution
    category_counts = {}
    for test_case in test_cases:
        category_counts[test_case.test_category] = category_counts.get(test_case.test_category, 0) + 1
    
    logger.info("Test case distribution:")
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count} cases")
    
    # Initialize fraud detection system
    logger.info("Initializing Fraud Detection System...")
    
    try:
        fraud_system = FraudDetectionSystem()
        await fraud_system.start()
        
        # Wait for system to fully initialize
        await asyncio.sleep(5)
        
        # Verify system is running
        system_status = await fraud_system.get_system_status()
        if system_status['system_status'] != 'running':
            logger.error("Fraud detection system failed to start properly")
            return 1
        
        logger.info(f"System started successfully with {system_status['system_stats']['agents_active']} agents")
        
    except Exception as e:
        logger.error(f"Failed to start fraud detection system: {e}")
        return 1
    
    # Run validation tests
    try:
        logger.info("Starting validation test execution...")
        validation_runner = ValidationRunner(fraud_system)
        
        start_time = time.time()
        results = await validation_runner.run_validation_suite(
            test_cases, 
            concurrent_limit=args.concurrent
        )
        execution_time = time.time() - start_time
        
        logger.info(f"Validation completed in {execution_time:.2f} seconds")
        logger.info(f"Average time per test: {(execution_time / len(results)):.3f} seconds")
        
        # Generate summary report
        summary = validation_runner.generate_summary_report()
        
        # Print summary to console
        print("\n" + "="*80)
        print("VALIDATION SUMMARY REPORT")
        print("="*80)
        
        summary_stats = summary['summary']
        print(f"Total Tests: {summary_stats['total_tests']}")
        print(f"Passed Tests: {summary_stats['passed_tests']}")
        print(f"Pass Rate: {summary_stats['pass_rate']:.2%}")
        print(f"Decision Accuracy: {summary_stats['decision_accuracy']:.2%}")
        print(f"Risk Score Accuracy: {summary_stats['risk_score_accuracy']:.2%}")
        print(f"False Positive Rate: {summary_stats['false_positive_rate']:.2%}")
        print(f"False Negative Rate: {summary_stats['false_negative_rate']:.2%}")
        print(f"Avg Processing Time: {summary_stats['avg_processing_time_ms']:.1f}ms")
        print(f"Avg Confidence Score: {summary_stats['avg_confidence_score']:.3f}")
        
        print("\nPERFORMANCE METRICS:")
        perf_stats = summary['performance_stats']
        print(f"Min Processing Time: {perf_stats['min_processing_time_ms']:.1f}ms")
        print(f"Max Processing Time: {perf_stats['max_processing_time_ms']:.1f}ms")
        print(f"95th Percentile Time: {perf_stats['p95_processing_time_ms']:.1f}ms")
        print(f"Avg Agents Participated: {perf_stats['avg_agents_participated']:.1f}")
        
        print("\nCATEGORY BREAKDOWN:")
        for category, stats in summary['category_breakdown'].items():
            print(f"  {category.upper()}:")
            print(f"    Tests: {stats['total']}")
            print(f"    Pass Rate: {stats['pass_rate']:.2%}")
            print(f"    Decision Accuracy: {stats['decision_accuracy']:.2%}")
            print(f"    FP Rate: {stats['false_positive_rate']:.2%}")
            print(f"    FN Rate: {stats['false_negative_rate']:.2%}")
        
        # Export results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output:
            output_file = args.output
        else:
            output_file = f"validation_results_{timestamp}.csv"
        
        csv_file = validation_runner.export_results_to_csv(output_file)
        print(f"\nDetailed results exported to: {csv_file}")
        
        # Export summary to JSON
        summary_file = output_file.replace('.csv', '_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary report exported to: {summary_file}")
        
        # Determine exit code based on results
        if summary_stats['pass_rate'] >= 0.95:  # 95% pass rate required
            print(f"\n✅ VALIDATION PASSED - Pass rate: {summary_stats['pass_rate']:.2%}")
            exit_code = 0
        else:
            print(f"\n❌ VALIDATION FAILED - Pass rate: {summary_stats['pass_rate']:.2%} (minimum: 95%)")
            exit_code = 1
        
        # Additional failure conditions
        if summary_stats['false_positive_rate'] > 0.02:  # Max 2% FP rate
            print(f"❌ False positive rate too high: {summary_stats['false_positive_rate']:.2%}")
            exit_code = 1
        
        if summary_stats['false_negative_rate'] > 0.005:  # Max 0.5% FN rate
            print(f"❌ False negative rate too high: {summary_stats['false_negative_rate']:.2%}")
            exit_code = 1
        
        if summary_stats['avg_processing_time_ms'] > 150:  # Max 150ms average
            print(f"❌ Average processing time too high: {summary_stats['avg_processing_time_ms']:.1f}ms")
            exit_code = 1
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Validation execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        try:
            await fraud_system.stop()
            logger.info("Fraud detection system stopped")
        except Exception as e:
            logger.error(f"Error stopping fraud detection system: {e}")


def create_sample_config():
    """Create a sample validation configuration file."""
    
    sample_config = {
        "description": "Sample validation configuration for fraud detection system",
        "test_cases": [
            {
                "test_id": "CUSTOM_LEGIT_001",
                "transaction_data": {
                    "transaction_id": "tx_custom_001",
                    "user_id": "user_123456",
                    "amount": 299.99,
                    "currency": "USD",
                    "transaction_type": "purchase",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "country": "US",
                    "payment_method": "credit_card",
                    "device_id": "device_abc123",
                    "device_type": "desktop",
                    "ip_address": "192.168.1.100",
                    "is_mobile": False,
                    "velocity_1h": 1,
                    "velocity_24h": 3,
                    "account_age_days": 180,
                    "is_first_transaction": False,
                    "session_duration": 300,
                    "pages_visited": 5
                },
                "expected_decision": "approve",
                "expected_risk_range": [0.0, 0.3],
                "fraud_label": False,
                "test_category": "custom_legitimate",
                "description": "Custom legitimate transaction test",
                "priority": "normal"
            },
            {
                "test_id": "CUSTOM_FRAUD_001",
                "transaction_data": {
                    "transaction_id": "tx_custom_002",
                    "user_id": "user_789012",
                    "amount": 2.99,
                    "currency": "USD",
                    "transaction_type": "purchase",
                    "timestamp": "2024-01-15T03:15:00Z",
                    "country": "XX",
                    "payment_method": "credit_card",
                    "device_id": "device_suspicious_001",
                    "device_type": "mobile",
                    "ip_address": "10.0.0.1",
                    "is_mobile": True,
                    "velocity_1h": 15,
                    "velocity_24h": 45,
                    "account_age_days": 2,
                    "is_first_transaction": True,
                    "session_duration": 30,
                    "pages_visited": 1
                },
                "expected_decision": "decline",
                "expected_risk_range": [0.7, 1.0],
                "fraud_label": True,
                "test_category": "custom_fraud",
                "description": "Custom fraud pattern test",
                "priority": "high"
            }
        ]
    }
    
    with open('validation_config_sample.json', 'w') as f:
        json.dump(sample_config, f, indent=2)
    
    print("Sample configuration created: validation_config_sample.json")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Fraud Detection System Validation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 100 generated test cases
  python run_validation.py --num-tests 100

  # Use custom test configuration
  python run_validation.py --config my_tests.json

  # Export to specific file with higher concurrency
  python run_validation.py --num-tests 200 --output results.csv --concurrent 20

  # Create sample configuration file
  python run_validation.py --create-sample-config
        """
    )
    
    parser.add_argument(
        '--num-tests', 
        type=int, 
        default=100, 
        help='Number of test cases to generate (default: 100)'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to custom test configuration JSON file'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        help='Output CSV file path (default: validation_results_TIMESTAMP.csv)'
    )
    
    parser.add_argument(
        '--concurrent', 
        type=int, 
        default=10, 
        help='Number of concurrent test executions (default: 10)'
    )
    
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--create-sample-config', 
        action='store_true',
        help='Create a sample validation configuration file'
    )
    
    args = parser.parse_args()
    
    # Handle sample config creation
    if args.create_sample_config:
        create_sample_config()
        return 0
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Run validation suite
    try:
        exit_code = asyncio.run(run_validation_suite(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()