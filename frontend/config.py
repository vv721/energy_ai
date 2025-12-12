"""前端配置模块，管理前端应用的全局配置和常量。"""

import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 确保项目根目录在sys.path中，以便可以导入backend模块
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 向量存储路径
VECTORSTORE_PATH = os.path.join(PROJECT_ROOT, "vectorstore", "energy_docs")

# 集合名称
COLLECTION_NAME = "energy_docs"

# 模型配置（从环境变量读取）
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "aliyun")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen-turbo")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

# 删除以下Streamlit特定配置：
# PAGE_CONFIG = {...}
# PAGES = [...]
# UI_CONFIG = {...}