import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import aiohttp
from pydantic import BaseModel
import time

logger = logging.getLogger(__name__)


class MCPServerConnection(BaseModel):
    server_id: str
    endpoint: str
    capabilities: List[str]
    tools: List[str]
    status: str = "disconnected"
    last_heartbeat: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 5


class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: str
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise Exception("Circuit breaker is OPEN")
            else:
                self.state = "HALF_OPEN"
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise e


class MCPClient:
    def __init__(self, connection_pool_size: int = 10):
        self.servers: Dict[str, MCPServerConnection] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.connection_pool_size = connection_pool_size
        self.request_timeout = 30.0
        self.retry_delays = [1, 2, 4, 8, 16]  # exponential backoff
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        connector = aiohttp.TCPConnector(limit=self.connection_pool_size)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={"Content-Type": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def register_server(
        self, 
        server_id: str, 
        endpoint: str, 
        capabilities: List[str],
        tools: List[str]
    ):
        """Register an MCP server with the client."""
        connection = MCPServerConnection(
            server_id=server_id,
            endpoint=endpoint,
            capabilities=capabilities,
            tools=tools
        )
        
        self.servers[server_id] = connection
        self.circuit_breakers[server_id] = CircuitBreaker()
        
        # Test initial connection
        try:
            await self._test_connection(server_id)
            connection.status = "connected"
            logger.info(f"Successfully registered MCP server: {server_id}")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {server_id}: {e}")
            connection.status = "failed"
    
    async def _test_connection(self, server_id: str):
        """Test connection to an MCP server."""
        request = MCPRequest(
            id=f"test_{server_id}_{int(time.time())}",
            method="capabilities",
            params={}
        )
        
        response = await self._send_request(server_id, request)
        if response.error:
            raise Exception(f"Server error: {response.error}")
    
    async def call_tool(
        self, 
        server_id: str, 
        tool_name: str, 
        parameters: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Call a tool on a specific MCP server."""
        if server_id not in self.servers:
            raise ValueError(f"Unknown server: {server_id}")
        
        server = self.servers[server_id]
        if server.status != "connected":
            await self._attempt_reconnection(server_id)
        
        request = MCPRequest(
            id=f"{tool_name}_{server_id}_{int(time.time() * 1000)}",
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": parameters
            }
        )
        
        circuit_breaker = self.circuit_breakers[server_id]
        
        try:
            response = await circuit_breaker.call(
                self._send_request_with_retry,
                server_id,
                request,
                timeout or self.request_timeout
            )
            
            if response.error:
                logger.error(f"Tool call error on {server_id}.{tool_name}: {response.error}")
                raise Exception(f"Tool error: {response.error}")
            
            return response.result
            
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} on server {server_id}: {e}")
            server.status = "error"
            raise
    
    async def _send_request_with_retry(
        self, 
        server_id: str, 
        request: MCPRequest, 
        timeout: float
    ) -> MCPResponse:
        """Send request with exponential backoff retry logic."""
        server = self.servers[server_id]
        last_exception = None
        
        for attempt in range(len(self.retry_delays) + 1):
            try:
                return await self._send_request(server_id, request, timeout)
                
            except Exception as e:
                last_exception = e
                server.retry_count += 1
                
                if attempt < len(self.retry_delays):
                    delay = self.retry_delays[attempt]
                    logger.warning(
                        f"Request to {server_id} failed (attempt {attempt + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed for {server_id}")
                    break
        
        raise last_exception
    
    async def _send_request(
        self, 
        server_id: str, 
        request: MCPRequest, 
        timeout: Optional[float] = None
    ) -> MCPResponse:
        """Send a single request to an MCP server."""
        if not self.session:
            raise RuntimeError("MCPClient session not initialized")
        
        server = self.servers[server_id]
        
        try:
            async with self.session.post(
                server.endpoint,
                json=request.dict(),
                timeout=aiohttp.ClientTimeout(total=timeout or self.request_timeout)
            ) as response:
                
                if response.status != 200:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status
                    )
                
                response_data = await response.json()
                mcp_response = MCPResponse(**response_data)
                
                # Update server status
                server.last_heartbeat = datetime.now()
                server.retry_count = 0
                
                return mcp_response
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout calling {server_id}")
            raise Exception(f"Request timeout to server {server_id}")
        
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error calling {server_id}: {e}")
            raise Exception(f"HTTP error: {e}")
    
    async def _attempt_reconnection(self, server_id: str):
        """Attempt to reconnect to a failed server."""
        server = self.servers[server_id]
        
        if server.retry_count >= server.max_retries:
            raise Exception(f"Max retries exceeded for server {server_id}")
        
        try:
            await self._test_connection(server_id)
            server.status = "connected"
            server.retry_count = 0
            logger.info(f"Successfully reconnected to server {server_id}")
            
        except Exception as e:
            server.retry_count += 1
            logger.error(f"Reconnection failed for {server_id}: {e}")
            raise
    
    async def broadcast_to_servers(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any],
        server_filter: Optional[Callable[[str], bool]] = None
    ) -> Dict[str, Any]:
        """Broadcast a tool call to multiple servers."""
        target_servers = []
        
        for server_id, server in self.servers.items():
            if server.status == "connected" and tool_name in server.tools:
                if not server_filter or server_filter(server_id):
                    target_servers.append(server_id)
        
        if not target_servers:
            raise Exception(f"No available servers for tool: {tool_name}")
        
        # Execute calls concurrently
        tasks = []
        for server_id in target_servers:
            task = asyncio.create_task(
                self.call_tool(server_id, tool_name, parameters)
            )
            tasks.append((server_id, task))
        
        results = {}
        errors = {}
        
        for server_id, task in tasks:
            try:
                result = await task
                results[server_id] = result
            except Exception as e:
                errors[server_id] = str(e)
                logger.error(f"Broadcast failed for {server_id}: {e}")
        
        return {
            "results": results,
            "errors": errors,
            "success_count": len(results),
            "error_count": len(errors)
        }
    
    async def get_server_health(self, server_id: str) -> Dict[str, Any]:
        """Get health status of a specific server."""
        if server_id not in self.servers:
            raise ValueError(f"Unknown server: {server_id}")
        
        server = self.servers[server_id]
        circuit_breaker = self.circuit_breakers[server_id]
        
        return {
            "server_id": server_id,
            "status": server.status,
            "last_heartbeat": server.last_heartbeat.isoformat() if server.last_heartbeat else None,
            "retry_count": server.retry_count,
            "circuit_breaker_state": circuit_breaker.state,
            "circuit_breaker_failures": circuit_breaker.failure_count,
            "capabilities": server.capabilities,
            "tools": server.tools
        }
    
    async def get_all_servers_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all registered servers."""
        health_status = {}
        
        for server_id in self.servers:
            try:
                health_status[server_id] = await self.get_server_health(server_id)
            except Exception as e:
                health_status[server_id] = {
                    "server_id": server_id,
                    "status": "error",
                    "error": str(e)
                }
        
        return health_status
    
    def get_available_tools(self) -> Dict[str, List[str]]:
        """Get all available tools grouped by server."""
        tools_by_server = {}
        
        for server_id, server in self.servers.items():
            if server.status == "connected":
                tools_by_server[server_id] = server.tools
        
        return tools_by_server