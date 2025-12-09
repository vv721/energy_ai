"""异常类模块。"""


class EnergyAIBaseException(Exception):
    """能源AI系统基础异常类"""
    pass


class LLMConfigError(EnergyAIBaseException):
    """LLM配置错误"""
    pass


class DocumentProcessingError(EnergyAIBaseException):
    """文档处理错误"""
    pass


class VectorStoreError(EnergyAIBaseException):
    """向量存储错误"""
    pass


class RAGChainError(EnergyAIBaseException):
    """RAG链错误"""
    pass


class APIConnectionError(EnergyAIBaseException):
    """API连接错误"""
    pass