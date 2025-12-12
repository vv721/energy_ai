"""MCP 客户端"""

from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator
import httpx
import json

class MCPClientInterface(ABC):
    """MCP 客户端接口"""

    @abstractmethod
    async def connect(self, server_config: Dict[str, Any]) -> bool:
        #连接到MCP服务器
        pass

    @abstractmethod
    async def call_tool(self, tool_name: str, 
                        arguments: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        #调用MCP工具
        pass

    @abstractmethod
    async def close(self):
        #关闭连接
        pass


class AliyunMCPClient(MCPClientInterface):
    #阿里云MCP客户端

    def __init__(self):
        self.client = None
        self.connected = False

    async def connect(self, server_config: Dict[str, Any]) -> bool:
        #连接到MCP服务器
        try:
            self.client = httpx.AsyncClient(
                timeout=10.0,
                headers=server_config.get("headers", {})
            )
            self.connected = True
            return True
        except Exception as e:
            print(f"连接MCP服务器失败: {e}")
            return False
        
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        #调用工具，使用SSE流式相应
        if not self.connected or not self.client:
            yield {"error": "MCP客户端未连接"}
            return
        
        try:
            #构建SSE请求
            sse_data = {
                "tool": tool_name,
                "arguments": arguments
            }

            async with self.client.stream(
                'POST',
                self.server_config['baseUrl'],
                json=sse_data
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        data_line = line[6:]
                        if data_line.strip():
                            try:
                                yield json.loads(data_line)
                            except json.JSONDecodeError:
                                yield {"content": data_line}
        except Exception as e:
            yield {"error": f"调用工具失败: {e}"}
            
    async def close(self):
        #关闭连接
        if self.client:
            await self.client.aclose()
            self.connected = False

            