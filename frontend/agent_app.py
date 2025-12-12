"""Agent功能前端界面"""

import streamlit as st
import asyncio
import sys
import os

# 添加backend路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backend.mcp.agent_manager import AgentManager

class AgentApp:
    """Agent应用类"""
    
    def __init__(self):
        self.agent_manager = None
        self.initialized = False
    
    async def initialize(self):
        """初始化应用"""
        if not self.initialized:
            self.agent_manager = AgentManager()
            self.initialized = await self.agent_manager.initialize()
        return self.initialized
    
    def render_sidebar(self):
        """渲染侧边栏"""
        st.sidebar.title("🔧 Agent功能")
        st.sidebar.markdown("使用阿里云百炼MCP组件")
        
        # 服务选择
        service_type = st.sidebar.selectbox(
            "选择天气服务",
            ["weather", "amap"],
            format_func=lambda x: "阿里云天气" if x == "weather" else "高德地图天气"
        )
        
        # 示例查询
        st.sidebar.markdown("### 示例查询")
        if st.sidebar.button("查询杭州天气"):
            st.session_state.location = "杭州"
            st.session_state.service_type = service_type
        
        if st.sidebar.button("查询北京天气"):
            st.session_state.location = "北京" 
            st.session_state.service_type = service_type
    
    def render_main_content(self):
        """渲染主内容区域"""
        st.title("🤖 Agent智能体")
        st.markdown("集成阿里云百炼MCP组件，提供天气查询和地理信息服务")
        
        # 输入区域
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            location = st.text_input(
                "请输入地点:",
                value=st.session_state.get('location', ''),
                placeholder="例如：杭州、北京、上海"
            )
        with col2:
            service_type = st.selectbox(
                "服务类型",
                ["weather", "amap"],
                index=0 if st.session_state.get('service_type') != "amap" else 1,
                format_func=lambda x: "阿里云天气" if x == "weather" else "高德地图天气"
            )
        with col3:
            query_button = st.button("查询", type="primary", use_container_width=True)
        
        # 查询结果
        if query_button and location:
            asyncio.run(self.execute_query(location, service_type))
    
    async def execute_query(self, location: str, service_type: str):
        """执行查询"""
        if not await self.initialize():
            st.error("❌ Agent初始化失败，请检查DASHSCOPE_API_KEY环境变量")
            return
        
        with st.spinner(f"正在查询{location}的天气信息..."):
            result = await self.agent_manager.query_weather(location, service_type)
            self.display_result(result, location, service_type)
    
    def display_result(self, result: dict, location: str, service_type: str):
        """显示查询结果"""
        if result["success"]:
            st.success(f"✅ {location}天气查询成功")
            
            # 显示服务信息
            service_name = "阿里云天气" if service_type == "weather" else "高德地图天气"
            st.info(f"**数据来源:** {service_name}")
            
            # 显示内容
            st.subheader("查询结果")
            st.markdown(result["content"])
        else:
            st.error(f"❌ 查询失败: {result['error']}")
    
    async def close(self):
        """关闭应用"""
        if self.agent_manager:
            await self.agent_manager.close()

def main():
    """主函数"""
    st.set_page_config(
        page_title="Agent智能体",
        page_icon="🤖",
        layout="wide"
    )
    
    # 初始化session state
    if 'location' not in st.session_state:
        st.session_state.location = ''
    if 'service_type' not in st.session_state:
        st.session_state.service_type = 'weather'
    
    # 创建应用实例
    app = AgentApp()
    
    try:
        # 渲染界面
        app.render_sidebar()
        app.render_main_content()
    except Exception as e:
        st.error(f"应用运行错误: {e}")
    finally:
        # 确保资源清理
        asyncio.run(app.close())

if __name__ == "__main__":
    main()