import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

from core.mcp_client import MCPClient
from agents.base_agent import BaseFraudDetectionAgent, AgentDecision
from agents.coordination_agent import CoordinationAgent
from schemas.transaction_schema import Transaction, FraudScore, FraudAlert

logger = logging.getLogger(__name__)


class ProcessingStatus(str, Enum):
    RECEIVED = "received"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ProcessingResult:
    transaction_id: str
    status: ProcessingStatus
    risk_score: float
    decision: str
    confidence: float
    processing_time_ms: float
    agent_decisions: List[AgentDecision]
    mcp_server_calls: Dict[str, Any]
    timestamp: datetime
    error_message: Optional[str] = None


@dataclass
class PerformanceMetrics:
    total_processed: int = 0
    total_processing_time_ms: float = 0.0
    successful_transactions: int = 0
    failed_transactions: int = 0
    timeout_transactions: int = 0
    avg_processing_time_ms: float = 0.0
    p95_processing_time_ms: float = 0.0
    p99_processing_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    
    def update_metrics(self, processing_time_ms: float, success: bool):
        self.total_processed += 1
        self.total_processing_time_ms += processing_time_ms
        
        if success:
            self.successful_transactions += 1
        else:
            self.failed_transactions += 1
        
        self.avg_processing_time_ms = self.total_processing_time_ms / self.total_processed


class CircuitBreaker:
    """Circuit breaker for handling agent failures and overload."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time < self.recovery_timeout:
                    raise Exception("Circuit breaker is OPEN")
                else:
                    self.state = "HALF_OPEN"
        
        try:
            result = func(*args, **kwargs)
            
            with self.lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
            
            return result
            
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
            
            raise e


class LoadBalancer:
    """Load balancer for distributing work across agent instances."""
    
    def __init__(self):
        self.agents: Dict[str, List[BaseFraudDetectionAgent]] = defaultdict(list)
        self.agent_loads: Dict[str, int] = defaultdict(int)
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
    
    def register_agent(self, agent: BaseFraudDetectionAgent):
        """Register an agent instance for load balancing."""
        
        with self.lock:
            self.agents[agent.specialization].append(agent)
            self.agent_loads[agent.agent_id] = 0
        
        logger.info(f"Registered agent {agent.agent_id} ({agent.specialization}) for load balancing")
    
    def get_agent(self, specialization: str) -> Optional[BaseFraudDetectionAgent]:
        """Get the least loaded agent of specified specialization."""
        
        with self.lock:
            available_agents = self.agents.get(specialization, [])
            
            if not available_agents:
                return None
            
            # Find agent with minimum load
            min_load_agent = min(available_agents, key=lambda a: self.agent_loads[a.agent_id])
            self.agent_loads[min_load_agent.agent_id] += 1
            
            return min_load_agent
    
    def release_agent(self, agent: BaseFraudDetectionAgent):
        """Release agent after processing completion."""
        
        with self.lock:
            if agent.agent_id in self.agent_loads:
                self.agent_loads[agent.agent_id] = max(0, self.agent_loads[agent.agent_id] - 1)
    
    def get_load_status(self) -> Dict[str, Any]:
        """Get current load balancer status."""
        
        with self.lock:
            return {
                "total_agents": sum(len(agents) for agents in self.agents.values()),
                "specializations": list(self.agents.keys()),
                "agent_loads": dict(self.agent_loads),
                "avg_load_per_specialization": {
                    spec: sum(self.agent_loads[agent.agent_id] for agent in agents) / len(agents)
                    for spec, agents in self.agents.items() if agents
                }
            }


class TransactionQueue:
    """High-performance transaction queue with priority handling."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queues = {
            "critical": deque(),
            "high": deque(),
            "normal": deque(),
            "low": deque()
        }
        self.queue_sizes = {priority: 0 for priority in self.queues.keys()}
        self.lock = asyncio.Lock()
        self.not_empty = asyncio.Condition(self.lock)
        self.not_full = asyncio.Condition(self.lock)
        self.total_size = 0
    
    async def put(self, transaction: Dict[str, Any], priority: str = "normal") -> bool:
        """Add transaction to queue with specified priority."""
        
        async with self.not_full:
            while self.total_size >= self.max_size:
                await self.not_full.wait()
            
            if priority not in self.queues:
                priority = "normal"
            
            self.queues[priority].append({
                "transaction": transaction,
                "enqueued_at": datetime.now(),
                "priority": priority
            })
            
            self.queue_sizes[priority] += 1
            self.total_size += 1
            
            self.not_empty.notify()
            return True
    
    async def get(self) -> Optional[Dict[str, Any]]:
        """Get highest priority transaction from queue."""
        
        async with self.not_empty:
            while self.total_size == 0:
                await self.not_empty.wait()
            
            # Get from highest priority queue first
            for priority in ["critical", "high", "normal", "low"]:
                if self.queues[priority]:
                    item = self.queues[priority].popleft()
                    self.queue_sizes[priority] -= 1
                    self.total_size -= 1
                    
                    self.not_full.notify()
                    return item
            
            return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        
        return {
            "total_size": self.total_size,
            "max_size": self.max_size,
            "queue_sizes": dict(self.queue_sizes),
            "utilization": self.total_size / self.max_size
        }


class RealTimeFraudProcessor:
    """High-performance real-time fraud detection processor."""
    
    def __init__(
        self, 
        mcp_client: MCPClient,
        max_concurrent_transactions: int = 1000,
        processing_timeout_seconds: int = 30
    ):
        self.mcp_client = mcp_client
        self.max_concurrent_transactions = max_concurrent_transactions
        self.processing_timeout = processing_timeout_seconds
        
        # Core components
        self.transaction_queue = TransactionQueue(max_size=50000)
        self.load_balancer = LoadBalancer()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.processing_times = deque(maxlen=10000)  # For percentile calculations
        self.active_transactions: Dict[str, Dict[str, Any]] = {}
        
        # Worker management
        self.worker_pool = ThreadPoolExecutor(max_workers=100)
        self.processing_workers: List[asyncio.Task] = []
        self.is_running = False
        
        # Auto-scaling
        self.target_latency_ms = 100
        self.scale_up_threshold = 0.8  # 80% capacity
        self.scale_down_threshold = 0.3  # 30% capacity
        
        logger.info(f"Initialized RealTimeFraudProcessor with max_concurrent={max_concurrent_transactions}")
    
    async def start(self, num_workers: int = 10):
        """Start the real-time processing system."""
        
        self.is_running = True
        
        # Start processing workers
        for i in range(num_workers):
            worker = asyncio.create_task(self._processing_worker(f"worker_{i}"))
            self.processing_workers.append(worker)
        
        # Start monitoring and auto-scaling
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.processing_workers.append(monitoring_task)
        
        logger.info(f"Started RealTimeFraudProcessor with {num_workers} workers")
    
    async def stop(self):
        """Stop the processing system gracefully."""
        
        self.is_running = False
        
        # Cancel all workers
        for worker in self.processing_workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.processing_workers, return_exceptions=True)
        
        # Shutdown thread pool
        self.worker_pool.shutdown(wait=True)
        
        logger.info("RealTimeFraudProcessor stopped gracefully")
    
    def register_agent(self, agent: BaseFraudDetectionAgent):
        """Register an agent for processing."""
        
        self.load_balancer.register_agent(agent)
        self.circuit_breakers[agent.agent_id] = CircuitBreaker()
        
        logger.info(f"Registered agent {agent.agent_id} for real-time processing")
    
    async def process_transaction_stream(
        self, 
        transaction: Dict[str, Any], 
        priority: str = "normal"
    ) -> str:
        """Add transaction to processing queue."""
        
        transaction_id = transaction.get("transaction_id", f"tx_{uuid.uuid4().hex[:8]}")
        
        # Add processing metadata
        transaction.update({
            "received_at": datetime.now().isoformat(),
            "processing_priority": priority
        })
        
        # Queue transaction for processing
        success = await self.transaction_queue.put(transaction, priority)
        
        if success:
            logger.debug(f"Queued transaction {transaction_id} with priority {priority}")
            return transaction_id
        else:
            logger.error(f"Failed to queue transaction {transaction_id} - queue full")
            raise Exception("Transaction queue is full")
    
    async def get_processing_result(self, transaction_id: str, timeout: float = 30.0) -> Optional[ProcessingResult]:
        """Get processing result for a transaction."""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if transaction_id in self.active_transactions:
                tx_info = self.active_transactions[transaction_id]
                
                if tx_info.get("status") in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
                    result = tx_info.get("result")
                    
                    # Clean up
                    del self.active_transactions[transaction_id]
                    
                    return result
            
            await asyncio.sleep(0.1)  # Check every 100ms
        
        # Timeout
        if transaction_id in self.active_transactions:
            self.active_transactions[transaction_id]["status"] = ProcessingStatus.TIMEOUT
        
        return None
    
    async def _processing_worker(self, worker_id: str):
        """Main processing worker loop."""
        
        logger.info(f"Processing worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get transaction from queue
                queue_item = await self.transaction_queue.get()
                if not queue_item:
                    continue
                
                transaction = queue_item["transaction"]
                transaction_id = transaction.get("transaction_id", "unknown")
                
                # Track active transaction
                self.active_transactions[transaction_id] = {
                    "status": ProcessingStatus.PROCESSING,
                    "start_time": datetime.now(),
                    "worker_id": worker_id
                }
                
                # Process transaction with timeout
                try:
                    result = await asyncio.wait_for(
                        self._process_single_transaction(transaction),
                        timeout=self.processing_timeout
                    )
                    
                    self.active_transactions[transaction_id].update({
                        "status": ProcessingStatus.COMPLETED,
                        "result": result,
                        "end_time": datetime.now()
                    })
                    
                    # Update performance metrics
                    self.performance_metrics.update_metrics(result.processing_time_ms, True)
                    self.processing_times.append(result.processing_time_ms)
                    
                    logger.debug(f"Worker {worker_id} completed transaction {transaction_id} in {result.processing_time_ms:.1f}ms")
                    
                except asyncio.TimeoutError:
                    error_result = ProcessingResult(
                        transaction_id=transaction_id,
                        status=ProcessingStatus.TIMEOUT,
                        risk_score=0.5,
                        decision="review",
                        confidence=0.0,
                        processing_time_ms=self.processing_timeout * 1000,
                        agent_decisions=[],
                        mcp_server_calls={},
                        timestamp=datetime.now(),
                        error_message="Processing timeout"
                    )
                    
                    self.active_transactions[transaction_id].update({
                        "status": ProcessingStatus.TIMEOUT,
                        "result": error_result,
                        "end_time": datetime.now()
                    })
                    
                    self.performance_metrics.timeout_transactions += 1
                    logger.warning(f"Worker {worker_id} timeout on transaction {transaction_id}")
                
                except Exception as e:
                    error_result = ProcessingResult(
                        transaction_id=transaction_id,
                        status=ProcessingStatus.FAILED,
                        risk_score=0.5,
                        decision="review",
                        confidence=0.0,
                        processing_time_ms=0.0,
                        agent_decisions=[],
                        mcp_server_calls={},
                        timestamp=datetime.now(),
                        error_message=str(e)
                    )
                    
                    self.active_transactions[transaction_id].update({
                        "status": ProcessingStatus.FAILED,
                        "result": error_result,
                        "end_time": datetime.now()
                    })
                    
                    self.performance_metrics.update_metrics(0.0, False)
                    logger.error(f"Worker {worker_id} error on transaction {transaction_id}: {e}")
                
            except Exception as e:
                logger.error(f"Worker {worker_id} unexpected error: {e}")
                await asyncio.sleep(1)  # Brief pause on error
        
        logger.info(f"Processing worker {worker_id} stopped")
    
    async def _process_single_transaction(self, transaction: Dict[str, Any]) -> ProcessingResult:
        """Process a single transaction through the fraud detection pipeline."""
        
        start_time = time.time()
        transaction_id = transaction.get("transaction_id", "unknown")
        
        agent_decisions = []
        mcp_server_calls = {}
        
        try:
            # Step 1: Data Intelligence - Ingest and validate transaction
            data_intelligence_result = await self._call_mcp_server(
                "data_intelligence",
                "ingest_transaction",
                {
                    "transaction_data": transaction,
                    "source": "real_time_stream",
                    "timestamp": datetime.now().isoformat()
                }
            )
            mcp_server_calls["data_intelligence"] = data_intelligence_result
            
            # Step 2: Feature Engineering
            features_result = await self._call_mcp_server(
                "data_intelligence",
                "engineer_features",
                {
                    "raw_data": transaction,
                    "feature_config": {"advanced_patterns": True}
                }
            )
            
            # Step 3: Get coordination agent (main orchestrator)
            coordinator = self.load_balancer.get_agent("coordination")
            
            if coordinator:
                try:
                    # Process through coordination agent (which manages other agents)
                    circuit_breaker = self.circuit_breakers[coordinator.agent_id]
                    decision = await circuit_breaker.call(
                        coordinator.process_transaction,
                        transaction
                    )
                    agent_decisions.append(decision)
                    
                finally:
                    self.load_balancer.release_agent(coordinator)
            
            # Step 4: Make final decision using Decision Engine
            if agent_decisions:
                best_decision = max(agent_decisions, key=lambda d: d.confidence)
                
                decision_result = await self._call_mcp_server(
                    "decision_engine",
                    "make_decision",
                    {
                        "risk_scores": {
                            "overall_risk_score": best_decision.risk_score,
                            "confidence_score": best_decision.confidence,
                            "risk_factors": best_decision.reasoning,
                            "transaction_id": transaction_id
                        }
                    }
                )
                mcp_server_calls["decision_engine"] = decision_result
                
                final_decision = decision_result.get("decision", {})
                risk_score = final_decision.get("risk_score", best_decision.risk_score)
                decision_type = final_decision.get("decision", best_decision.decision_type)
                confidence = final_decision.get("confidence", best_decision.confidence)
            else:
                # Fallback decision
                risk_score = 0.5
                decision_type = "review"
                confidence = 0.3
            
            # Step 5: Log performance metrics
            await self._call_mcp_server(
                "monitoring",
                "log_performance_metric",
                {
                    "metric_name": "transaction_processing_time_ms",
                    "metric_value": (time.time() - start_time) * 1000,
                    "component": "real_time_processor"
                }
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return ProcessingResult(
                transaction_id=transaction_id,
                status=ProcessingStatus.COMPLETED,
                risk_score=risk_score,
                decision=decision_type,
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                agent_decisions=agent_decisions,
                mcp_server_calls=mcp_server_calls,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Transaction processing error for {transaction_id}: {e}")
            raise
    
    async def _call_mcp_server(self, server_id: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP server with error handling and retries."""
        
        max_retries = 3
        retry_delay = 0.1  # 100ms
        
        for attempt in range(max_retries):
            try:
                result = await self.mcp_client.call_tool(server_id, tool_name, parameters)
                return result
            
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    logger.warning(f"MCP call retry {attempt + 1} for {server_id}.{tool_name}: {e}")
                else:
                    logger.error(f"MCP call failed after {max_retries} attempts for {server_id}.{tool_name}: {e}")
                    return {"status": "error", "error": str(e)}
    
    async def _monitoring_loop(self):
        """Monitoring and auto-scaling loop."""
        
        logger.info("Started monitoring and auto-scaling loop")
        
        while self.is_running:
            try:
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Check for auto-scaling opportunities
                await self._check_auto_scaling()
                
                # Log system status
                if self.performance_metrics.total_processed % 1000 == 0:
                    logger.info(
                        f"Processing stats: {self.performance_metrics.total_processed} total, "
                        f"{self.performance_metrics.avg_processing_time_ms:.1f}ms avg, "
                        f"{self.performance_metrics.throughput_per_second:.1f} tx/sec"
                    )
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
        
        logger.info("Monitoring loop stopped")
    
    async def _update_performance_metrics(self):
        """Update performance metrics and percentiles."""
        
        if len(self.processing_times) > 10:
            sorted_times = sorted(self.processing_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            
            self.performance_metrics.p95_processing_time_ms = sorted_times[p95_index]
            self.performance_metrics.p99_processing_time_ms = sorted_times[p99_index]
        
        # Calculate throughput (transactions per second)
        if self.performance_metrics.total_processed > 0 and self.performance_metrics.avg_processing_time_ms > 0:
            self.performance_metrics.throughput_per_second = min(
                1000 / self.performance_metrics.avg_processing_time_ms,
                len(self.processing_workers) * 1000 / max(self.performance_metrics.avg_processing_time_ms, 1)
            )
    
    async def _check_auto_scaling(self):
        """Check if auto-scaling is needed."""
        
        queue_status = self.transaction_queue.get_queue_status()
        load_status = self.load_balancer.get_load_status()
        
        utilization = queue_status["utilization"]
        avg_processing_time = self.performance_metrics.avg_processing_time_ms
        
        # Scale up if queue is getting full or processing time is high
        should_scale_up = (
            utilization > self.scale_up_threshold or 
            avg_processing_time > self.target_latency_ms * 2
        )
        
        # Scale down if queue is mostly empty and processing time is low
        should_scale_down = (
            utilization < self.scale_down_threshold and 
            avg_processing_time < self.target_latency_ms * 0.5 and
            len(self.processing_workers) > 5  # Minimum workers
        )
        
        if should_scale_up and len(self.processing_workers) < 50:  # Max workers
            await self._scale_up()
        elif should_scale_down:
            await self._scale_down()
    
    async def _scale_up(self):
        """Add more processing workers."""
        
        new_worker_id = f"worker_{len(self.processing_workers)}"
        worker = asyncio.create_task(self._processing_worker(new_worker_id))
        self.processing_workers.append(worker)
        
        logger.info(f"Scaled up: Added worker {new_worker_id} (total: {len(self.processing_workers)})")
    
    async def _scale_down(self):
        """Remove processing workers."""
        
        if len(self.processing_workers) > 5:  # Keep minimum workers
            worker_to_remove = self.processing_workers.pop()
            worker_to_remove.cancel()
            
            logger.info(f"Scaled down: Removed worker (total: {len(self.processing_workers)})")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        return {
            "is_running": self.is_running,
            "performance_metrics": asdict(self.performance_metrics),
            "queue_status": self.transaction_queue.get_queue_status(),
            "load_balancer_status": self.load_balancer.get_load_status(),
            "active_transactions": len(self.active_transactions),
            "processing_workers": len(self.processing_workers),
            "circuit_breaker_status": {
                agent_id: cb.state for agent_id, cb in self.circuit_breakers.items()
            },
            "system_health": "healthy" if self.performance_metrics.avg_processing_time_ms < self.target_latency_ms * 2 else "degraded"
        }