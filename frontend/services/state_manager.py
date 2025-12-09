"""前端管理模块，管理Streamlit会话状态"""

import streamlit as st
from typing import Dict, Any, List, Optional

class StateManager:
    """状态管理类，封装Streamlit会话状态"""

    @staticmethod
    def init_state(key: str, default_val: Any) -> Any:
        """初始化会话状态"""
        if key not in st.session_state:
            st.session_state[key] = default_val
        return st.session_state[key]
    
    @staticmethod
    def get_state(key: str, default_val: Any = None) -> Any:
        """获取会话状态"""
        return st.session_state.get(key, default_val)
    
    @staticmethod
    def set_state(key: str, val: Any) -> None:
        """设置会话状态"""
        st.session_state[key] = val

    @staticmethod
    def del_state(key: str) -> None:
        """删除会话状态"""
        if key in st.session_state:
            del st.session_state[key]

    @staticmethod
    def get_chat_msg() -> List[Dict[str, str]]:
        #获取聊天消息
        return StateManager.get_state("chat_messages", [])
    
    @staticmethod
    def add_chat_msg(role: str, content: str) -> None:
        #添加聊天消息
        msgs = StateManager.get_chat_msg()
        msgs.append({"role": role, "content": content})
        StateManager.set_state("chat_messages", msgs)

    @staticmethod
    def clear_chat_msg() -> None:
        #清空聊天消息
        StateManager.set_state("chat_messages", [])

    @staticmethod
    def get_rag_components() -> Dict[str, Any]:
        #获取RAG组件状态
        return StateManager.get_state("rag_components", {})
    
    @staticmethod
    def set_rag_components(components: Dict[str, Any]) -> None:
        #设置RAG组件状态
        StateManager.set_state("rag_components", components)

    @staticmethod
    def get_vector_store_loaded() -> bool:
        #获取向量存储加载状态
        return StateManager.get_state("vector_store_loaded", False)
    
    @staticmethod
    def set_vector_store_loaded(loaded: bool) -> None:
        #设置向量存储加载状态
        StateManager.set_state("vector_store_loaded", loaded)

    @staticmethod
    def get_rag_initialized() -> bool:
        #获取RAG初始化状态
        return StateManager.get_state("rag_initialized", False)
    
    @staticmethod
    def set_rag_initialized(initialized: bool) -> None:
        #设置RAG初始化状态
        StateManager.set_state("rag_initialized", initialized)

    @staticmethod
    def get_llm_config() -> Dict[str, Any]:
        #获取LLM配置
        return StateManager.get_state("llm_config", {})
        
    @staticmethod
    def set_llm_config(config: Dict[str, Any]) -> None:
        #设置LLM配置
        StateManager.set_state("llm_config", config)
