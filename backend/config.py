"""配置模块，集中管理应用配置信息。"""

import os
from typing import Dict, Any

# 项目基础配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_PATH = os.path.join(PROJECT_ROOT, "vectorstore", "energy_docs")

# LLM配置
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_PROVIDER", "aliyun")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "qwen-turbo")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1000))

# API配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY")
ALIYUN_BASE_URL = os.getenv("ALIYUN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 文档处理配置
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
SUPPORTED_DOCUMENT_EXTENSIONS = [".pdf", ".txt", ".doc", ".docx"]

# RAG配置
DEFAULT_RETRIEVAL_K = 3
DEFAULT_COLLECTION_NAME = "energy_docs"

# 系统提示词
ENERGY_SYSTEM_PROMPT = """你是一个专业的能源AI助手，专注于回答与能源相关的问题，包括但不限于：
- 能源生产（煤炭、石油、天然气、风电、光伏、水电等）
- 能源消耗与效率
- 可再生能源技术与应用
- 能源政策与法规
- 能源经济与市场
- 能源存储与电网

请针对用户的问题提供准确、专业的回答。如果问题与能源无关，请礼貌地说明你主要专注于能源领域的问题。"""

# RAG提示词
RAG_PROMPT_TEMPLATE = """使用以下上下文来回答用户的问题。
如果你不知道答案，请诚实地说你不知道，不要编造答案。
回答应该简明扼要，并且基于提供的上下文。

上下文：
{context}

问题：{question}

回答："""

def get_llm_config(provider: str = None) -> Dict[str, Any]:
    """获取LLM配置"""
    provider = provider or DEFAULT_LLM_PROVIDER
    
    if provider == "openai":
        return {
            "provider": "openai",
            "model_name": os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
            "api_key": OPENAI_API_KEY,
            "api_base": OPENAI_BASE_URL,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS
        }
    elif provider == "aliyun":
        return {
            "provider": "aliyun",
            "model_name": os.getenv("MODEL_NAME", "qwen-turbo"),
            "api_key": ALIYUN_API_KEY,
            "api_base": ALIYUN_BASE_URL,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS
        }
    else:
        raise ValueError(f"不支持的provider: {provider}")