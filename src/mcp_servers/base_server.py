import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import traceback
import time
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import uvicorn

from schemas.mcp_tools_schema import MCPTool

logger = logging.getLogger(__name__)


class ServerStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


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


class MCPError(BaseModel):
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    
    @property
    def average_response_time(self) -> float:
        return self.total_response_time / max(self.request_count, 1)
    
    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.request_count, 1)


class BaseMCPServer(ABC):
    def __init__(
        self, 
        server_name: str, 
        version: str = "1.0.0",
        host: str = "0.0.0.0",
        port: int = 8000
    ):
        self.server_name = server_name
        self.version = version
        self.host = host
        self.port = port
        self.status = ServerStatus.STOPPED
        self.app = FastAPI(title=f"{server_name} MCP Server", version=version)
        self.tools: Dict[str, MCPTool] = {}
        self.tool_handlers: Dict[str, Callable] = {}
        self.capabilities: List[str] = []
        self.metrics = PerformanceMetrics()
        self.start_time = None
        self.middleware_handlers: List[Callable] = []
        
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_routes(self):
        """Set up FastAPI routes for MCP protocol."""
        
        @self.app.post("/")
        async def handle_mcp_request(request: Request):
            try:
                body = await request.json()
                mcp_request = MCPRequest(**body)
                return await self._process_request(mcp_request)
            except ValidationError as e:
                logger.error(f"Invalid MCP request: {e}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "jsonrpc": "2.0",
                        "id": body.get("id", "unknown"),
                        "error": {
                            "code": -32600,
                            "message": "Invalid Request",
                            "data": {"validation_errors": e.errors()}
                        }
                    }
                )
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                return JSONResponse(
                    status_code=500,
                    content={
                        "jsonrpc": "2.0",
                        "id": "unknown",
                        "error": {
                            "code": -32603,
                            "message": "Internal error",
                            "data": {"error": str(e)}
                        }
                    }
                )
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": self.status.value,
                "server_name": self.server_name,
                "version": self.version,
                "uptime_seconds": time.time() - self.start_time if self.start_time else 0,
                "metrics": {
                    "requests": self.metrics.request_count,
                    "success_rate": self.metrics.success_rate,
                    "avg_response_time_ms": self.metrics.average_response_time * 1000,
                    "min_response_time_ms": self.metrics.min_response_time * 1000,
                    "max_response_time_ms": self.metrics.max_response_time * 1000
                }
            }
        
        @self.app.get("/capabilities")
        async def get_capabilities():
            return {
                "capabilities": self.capabilities,
                "tools": [tool.dict() for tool in self.tools.values()],
                "server_info": {
                    "name": self.server_name,
                    "version": self.version
                }
            }
    
    def _setup_middleware(self):
        """Set up middleware for logging and metrics."""
        
        @self.app.middleware("http")
        async def logging_middleware(request: Request, call_next):
            start_time = time.time()
            
            # Execute custom middleware
            for handler in self.middleware_handlers:
                try:
                    await handler(request)
                except Exception as e:
                    logger.error(f"Middleware error: {e}")
            
            try:
                response = await call_next(request)
                process_time = time.time() - start_time
                
                # Update metrics
                self.metrics.request_count += 1
                self.metrics.total_response_time += process_time
                self.metrics.min_response_time = min(self.metrics.min_response_time, process_time)
                self.metrics.max_response_time = max(self.metrics.max_response_time, process_time)
                
                if response.status_code < 400:
                    self.metrics.success_count += 1
                else:
                    self.metrics.error_count += 1
                
                logger.info(
                    f"{request.method} {request.url.path} - "
                    f"Status: {response.status_code} - "
                    f"Time: {process_time:.3f}s"
                )
                
                return response
                
            except Exception as e:
                self.metrics.error_count += 1
                logger.error(f"Request processing error: {e}")
                raise
    
    async def _process_request(self, request: MCPRequest) -> JSONResponse:
        """Process incoming MCP request."""
        start_time = time.time()
        
        try:
            if request.method == "capabilities":
                result = await self._handle_capabilities()
            
            elif request.method == "tools/list":
                result = await self._handle_tools_list()
            
            elif request.method == "tools/call":
                if not request.params:
                    raise ValueError("Missing parameters for tool call")
                result = await self._handle_tool_call(request.params)
            
            else:
                # Allow subclasses to handle custom methods
                result = await self._handle_custom_method(request.method, request.params)
            
            response = MCPResponse(id=request.id, result=result)
            
        except Exception as e:
            logger.error(f"Error processing method '{request.method}': {e}")
            logger.error(traceback.format_exc())
            
            error = MCPError(
                code=-32603,
                message="Internal error",
                data={"error": str(e), "method": request.method}
            )
            response = MCPResponse(id=request.id, error=error.dict())
        
        return JSONResponse(content=response.dict())
    
    async def _handle_capabilities(self) -> Dict[str, Any]:
        """Handle capabilities request."""
        return {
            "capabilities": self.capabilities,
            "server_info": {
                "name": self.server_name,
                "version": self.version
            }
        }
    
    async def _handle_tools_list(self) -> Dict[str, Any]:
        """Handle tools list request."""
        return {
            "tools": [tool.dict() for tool in self.tools.values()]
        }
    
    async def _handle_tool_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if not tool_name:
            raise ValueError("Missing tool name")
        
        if tool_name not in self.tool_handlers:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        # Validate tool arguments against schema
        tool = self.tools[tool_name]
        try:
            # Basic schema validation (could be enhanced)
            if "required" in tool.inputSchema:
                for required_field in tool.inputSchema["required"]:
                    if required_field not in arguments:
                        raise ValueError(f"Missing required argument: {required_field}")
        except Exception as e:
            raise ValueError(f"Invalid arguments for tool {tool_name}: {e}")
        
        # Call the tool handler
        handler = self.tool_handlers[tool_name]
        result = await handler(**arguments)
        
        return {
            "tool": tool_name,
            "result": result,
            "metadata": {
                "execution_time": time.time(),
                "server": self.server_name
            }
        }
    
    async def _handle_custom_method(self, method: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle custom methods - to be overridden by subclasses."""
        raise ValueError(f"Unknown method: {method}")
    
    def register_tool(self, tool: MCPTool, handler: Callable):
        """Register a tool with its handler."""
        self.tools[tool.name] = tool
        self.tool_handlers[tool.name] = handler
        logger.info(f"Registered tool: {tool.name}")
    
    def add_capability(self, capability: str):
        """Add a server capability."""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
            logger.info(f"Added capability: {capability}")
    
    def add_middleware(self, handler: Callable):
        """Add custom middleware handler."""
        self.middleware_handlers.append(handler)
    
    @abstractmethod
    async def initialize(self):
        """Initialize server-specific components."""
        pass
    
    @abstractmethod
    async def shutdown(self):
        """Clean up server-specific components."""
        pass
    
    async def start(self):
        """Start the MCP server."""
        try:
            self.status = ServerStatus.STARTING
            self.start_time = time.time()
            
            logger.info(f"Starting {self.server_name} MCP Server on {self.host}:{self.port}")
            
            # Initialize server-specific components
            await self.initialize()
            
            self.status = ServerStatus.RUNNING
            logger.info(f"{self.server_name} MCP Server started successfully")
            
            # Start the FastAPI server
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            self.status = ServerStatus.ERROR
            logger.error(f"Error starting server: {e}")
            raise
    
    async def stop(self):
        """Stop the MCP server."""
        try:
            self.status = ServerStatus.STOPPING
            logger.info(f"Stopping {self.server_name} MCP Server")
            
            # Clean up server-specific components
            await self.shutdown()
            
            self.status = ServerStatus.STOPPED
            logger.info(f"{self.server_name} MCP Server stopped successfully")
            
        except Exception as e:
            self.status = ServerStatus.ERROR
            logger.error(f"Error stopping server: {e}")
            raise


class ServerManager:
    """Manages multiple MCP servers."""
    
    def __init__(self):
        self.servers: Dict[str, BaseMCPServer] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
    
    def register_server(self, server: BaseMCPServer):
        """Register an MCP server."""
        self.servers[server.server_name] = server
        logger.info(f"Registered server: {server.server_name}")
    
    async def start_server(self, server_name: str):
        """Start a specific server."""
        if server_name not in self.servers:
            raise ValueError(f"Unknown server: {server_name}")
        
        server = self.servers[server_name]
        task = asyncio.create_task(server.start())
        self.running_tasks[server_name] = task
    
    async def stop_server(self, server_name: str):
        """Stop a specific server."""
        if server_name in self.running_tasks:
            task = self.running_tasks[server_name]
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            del self.running_tasks[server_name]
        
        if server_name in self.servers:
            await self.servers[server_name].stop()
    
    async def start_all_servers(self):
        """Start all registered servers."""
        tasks = []
        for server_name in self.servers:
            task = asyncio.create_task(self.start_server(server_name))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_all_servers(self):
        """Stop all running servers."""
        tasks = []
        for server_name in list(self.running_tasks.keys()):
            task = asyncio.create_task(self.stop_server(server_name))
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_server_status(self, server_name: str) -> Dict[str, Any]:
        """Get status of a specific server."""
        if server_name not in self.servers:
            return {"error": f"Unknown server: {server_name}"}
        
        server = self.servers[server_name]
        return {
            "name": server_name,
            "status": server.status.value,
            "version": server.version,
            "uptime": time.time() - server.start_time if server.start_time else 0,
            "metrics": {
                "requests": server.metrics.request_count,
                "success_rate": server.metrics.success_rate,
                "avg_response_time": server.metrics.average_response_time
            },
            "capabilities": server.capabilities,
            "tools": list(server.tools.keys())
        }
    
    def get_all_servers_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all servers."""
        return {
            name: self.get_server_status(name) 
            for name in self.servers.keys()
        }