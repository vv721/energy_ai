"""前端配置模块，管理前端应用的全局配置和常量。"""

import os
import sys

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 确保项目根目录在sys.path中，以便可以导入backend模块
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 向量存储路径
VECTORSTORE_PATH = os.path.join(PROJECT_ROOT, "vectorstore", "energy_docs")

# 集合名称
COLLECTION_NAME = "energy_docs"

# 页面配置
PAGE_CONFIG = {
    "page_title": "能源AI系统",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# 页面导航配置
PAGES = [
    {"id": 0, "name": "能源AI助手", "module": "app"},
    {"id": 1, "name": "功能二", "module": None, "coming_soon": True},
    {"id": 2, "name": "功能三", "module": None, "coming_soon": True},
    {"id": 3, "name": "RAG manager", "module": "rag_manager"}
]

# UI样式配置
UI_CONFIG = {
    "primary_color": "#FF6B6B",
    "background_color": "#F8F9FA",
    "text_color": "#212529",
    "border_radius": "0px",  # 直角设计
    "button_style": {
        "use_container_width": True,
        "type": "primary"
    }
}