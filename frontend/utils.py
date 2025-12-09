import os
import sys
import time
import gc
from typing import List, Dict, Any


def get_project_root():
    """获取项目根目录路径，不依赖任何相对导入"""
    current_file = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(current_file))

def set_proj_root():
    """设置项目根目录到sys.path"""
    project_root = get_project_root()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        return True
    return False


def format_file_size(size_bytes: int) -> str:
    #格式化文件大小为可读格式
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1

    return f"{size_bytes:.2f} {size_names[i]}"


def format_docs_prvw(content: str, max_len: int = 100) -> str:
    #格式化文档预览，截断长文本
    if len(content) <= max_len:
        return content
    return content[:max_len] + "..."


def safe_del_res(obj: Any) -> None:
    #安全删除对象资源
    try:
        if hasattr(obj, 'vector_store') and obj.vector_store:
            if hasattr(obj.vector_store, '_client'):
                obj.vector_store._client.close()
            obj.vector_store = None
    except:
        pass


def force_gc_coll() -> None:
    gc.collect()
    time.sleep(0.2)


def display_stat_msg(status: str, message: str) -> None:
    #显示状态消息
    import streamlit as st

    if status == "success":
        st.success(message)
    elif status == "warning":
        st.warning(message)
    elif status == "error":
        st.error(message)
    elif status == "info":
        st.info(message)


def handle_exc(e: Exception, context: str = "") -> None:
    #处理异常，显示错误消息
    import streamlit as st

    error_msg = f"{context}: {str(e)}" if context else str(e)
    st.error(error_msg)


def reset_app_state() -> None:
    #重置应用状态
    import streamlit as st
    
    st.cache_resource.clear()

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.rerun()


def format_result_msg(result: Dict[str, Any]) -> None:
    #格式化查询结果消息
    import streamlit as st

    if not isinstance(result, dict):
        return

    msgs = result.get("msgs", [])
    for msg in msgs:
        if "失败" in msg or "错误" in msg:
            st.error(msg)
        elif "警告" in msg:
            st.warning(msg)
        else:
            st.success(msg)

    errors = result.get("errors", [])
    for err in errors:
        st.error(f"- {err}")
