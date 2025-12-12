"""MCP工具服务层"""

from typing import Dict, Any, List
from .config import MCPConfigManager
from .client import AliyunMCPClient


class MCPService:
    #服务基类
    def __init__(self, service_name: str):
        self.config_manager = MCPConfigManager()
        self.client = AliyunMCPClient()
        self.service_name = service_name
        self.server_config = self.config_manager.get_server_config(service_name)

    async def initialize(self) -> bool:
        #初始化服务
        if not self.config_manager.validate_config():
            return False
        return await self.client.connect(self.server_config)
    
    async def execute_tool(self, tool_name: str, **kwargs) -> List[Dict[str, Any]]:
        #执行工具
        result = []
        async for resp in self.client.call_tool(tool_name, **kwargs):
            result.append(resp)
        return result
    
    async def close(self):
        #关闭服务
        await self.client.close()


class WeatherService(MCPService):
    #天气服务
    def __init__(self):
        super().__init__("weather")

    async def query_weather(self, location: str) -> Dict[str, Any]:
        #查询天气
        resps = await self.execute_tool("query_weather", location=location)
        return self._process_weather_response(resps)
    
    def _process_weather_resp(self, resps: List[Dict[str, Any]]) -> Dict[str, Any]:
        #处理天气响应
        if not resps:
            return {"success": False, "error": "未获取到天气信息"}
        
        content_parts = []
        for resp in resps:
            if "content" in resp:
                if "content" in resp:
                    content_parts.append(resp["content"])
                elif "error" in resp:
                    return {"success": False, "error": resp["error"]}
                
        if content_parts:
            return {"success": True, 
                    "content": "".join(content_parts),
                    "location": "未知位置"}
        else:
            return {"success": False, "error": "响应内容为空"}
        

class AmapMapsService(MCPService):
    """高德地图服务"""
    
    def __init__(self):
        super().__init__("amap-maps")
    
    async def query_weather(self, city: str) -> Dict[str, Any]:
        """查询城市天气（使用高德地图）"""
        responses = await self.execute_tool("weather_query", city=city)
        return self._process_amap_responses(responses)
    
    async def geocode(self, address: str) -> Dict[str, Any]:
        """地理编码"""
        responses = await self.execute_tool("geocode", address=address)
        return self._process_amap_responses(responses)
    
    def _process_amap_responses(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理高德地图响应"""
        if not responses:
            return {"success": False, "error": "未收到响应"}
        
        content_parts = []
        for response in responses:
            if "content" in response:
                content_parts.append(response["content"])
            elif "error" in response:
                return {"success": False, "error": response["error"]}
        
        if content_parts:
            return {
                "success": True,
                "content": "".join(content_parts)
            }
        else:
            return {"success": False, "error": "响应内容为空"}