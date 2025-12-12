"""MCP 配置管理"""

import os
from typing import Dict, Any

class MCPConfigManager:
    #MCP配置管理器

    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self._servers_config = self._load_servers_config()

    def _load_servers_config(self) -> Dict[str, Dict[str, Any]]:
        #加载服务器配置
        return{
            "amap-maps":{
                "type": "sse",
                "description": "高德地图MCP Server，提供地理编码、天气查询、路径规划等服务",
                "isActive": True,
                "name": "高德地图MCP Server",
                "baseUrl": "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse",
                "headers":{
                    "Authorization": f"Bearer {self.api_key}"
                }
            },
            "weather":{
                "type": "sse",
                "description": "极简的天气查询工具，一句话即可查看全球天气",
                "isActive": True,
                "name": "Weather MCP Server",
                "baseUrl": "https://dashscope.aliyuncs.com/api/v1/mcps/weather/sse",
                "headers":{
                    "Authorization": f"Bearer {self.api_key}"
                }
            }
        }
    
    def get_server_config(self, server_name: str) -> Dict[str, Any]:
        #获取指定服务器配置
        return self._servers_config.get(server_name, {})
    
    def get_all_servers(self) -> Dict[str, Dict[str, Any]]:
        #获取所有服务器配置
        return self._servers_config.copy()
    
    def validate_config(self) -> bool:
        #验证配置是否有效
        if not self.api_key:
            return False
        return True
