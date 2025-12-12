"""智能体管理器"""

from typing import Dict, Any, List
from .services import WeatherService, AmapMapsService

class AgentManager:
    """智能体管理器"""
    
    def __init__(self):
        self.weather_service = None
        self.amap_service = None
        self.initialized = False
    
    async def initialize(self) -> bool:
        """初始化所有服务"""
        try:
            self.weather_service = WeatherService()
            self.amap_service = AmapMapsService()
            
            weather_ok = await self.weather_service.initialize()
            amap_ok = await self.amap_service.initialize()
            
            self.initialized = weather_ok and amap_ok
            return self.initialized
        except Exception as e:
            print(f"智能体初始化失败: {e}")
            return False
    
    async def query_weather(self, location: str, service_type: str = "weather") -> Dict[str, Any]:
        """查询天气"""
        if not self.initialized:
            return {"success": False, "error": "智能体未初始化"}
        
        try:
            if service_type == "weather":
                return await self.weather_service.query_weather(location)
            elif service_type == "amap":
                return await self.amap_service.query_weather(location)
            else:
                return {"success": False, "error": f"不支持的天气服务类型: {service_type}"}
        except Exception as e:
            return {"success": False, "error": f"天气查询失败: {str(e)}"}
    
    async def geocode_address(self, address: str) -> Dict[str, Any]:
        """地理编码"""
        if not self.initialized:
            return {"success": False, "error": "智能体未初始化"}
        
        try:
            return await self.amap_service.geocode(address)
        except Exception as e:
            return {"success": False, "error": f"地理编码失败: {str(e)}"}
    
    async def close(self):
        """关闭所有服务"""
        if self.weather_service:
            await self.weather_service.close()
        if self.amap_service:
            await self.amap_service.close()
        self.initialized = False