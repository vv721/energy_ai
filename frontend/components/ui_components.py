import streamlit as st
from typing import Dict, Any, List, Optional, Generator

from ..config import UI_CONFIG


def create_navi_menu(pages: List[Dict[str, Any]],
                     selected_page_key: str = "selected_page") -> None:
    #创建导航菜单
    st.markdown("### 功能导航")

    for page in pages:
        if page.get("coming_soon", False):
            st.button(page["name"], disabled=True, use_container_width=True)
        else:
            is_selected = st.session_state.get(selected_page_key, 0) == page["id"]
            button_type = "primary" if is_selected else "secondary"
            
            if st.button(page["name"], use_container_width=True, type=button_type):
                st.session_state[selected_page_key] = page["id"]


def create_docs_expander(doc_id: int, content: str, max_prvw_len: int = 100) -> None:
    #创建文档展开器
    from ..utils import format_docs_prvw
    preview = format_docs_prvw(content, max_prvw_len)
    with st.expander(f"文档 {doc_id} "):
        st.markdown(preview)


def create_act_btns(buttons: List[Dict[str, Any]]) -> None:
    #创建操作按钮
    cols = st.columns(len(buttons))

    for i, btn in enumerate(buttons):
        with cols[i]:
            btn_type = btn.get("type", "secondary")
            callback = btn.get("callback", None)
            args = btn.get("args", {})

            if st.button(btn['name'], type=btn_type, use_container_width=True):
                if callback:
                    callback(**args)


def create_stat_indicator(status: str, msg: str) -> None:
    #创建状态指示器
    if status == "success":
        st.success(msg)
    elif status == "warning":
        st.warning(msg)
    elif status == "error":
        st.error(msg)
    elif status == "info":
        st.info(msg)


def create_loading_spinner(text: str = "加载中...") -> Generator[None, None, None]:
    #创建加载动画
    with st.spinner(text):
        yield


def create_progress_bar(progress: int, text: str = "") -> None:
    #创建进度条
    st.progress(progress / 100)
    if text:
        st.write(text)


def create_file_uploader(label: str, accept_mul_files: bool = False,
                         file_type: Optional[List[str]] = None,
                         help_text: Optional[str] = None) -> Any:
    #文件上传组件
    return st.file_uploader(
        label = label,
        accept_multiple_files = accept_mul_files,
        type = file_type,
        help = help_text
    )


def create_text_input(label: str, key: str, default_val: str = "",
                      help_text: Optional[str] = None,
                      max_chars: Optional[int] = None) -> str:
    #文本输入组件
    return st.text_input(
        label = label,
        value=default_val,
        key=key,
        max_chars=max_chars,
        help = help_text
    )


def create_selectbox(label: str, options: List[str], key: str,
                     index: int = 0, help_text: Optional[str] = None) -> str:
    #选择框组件
    return st.selectbox(
        label = label,
        options = options,
        index = index,
        key = key,
        help = help_text
    )


def create_slider(label: str, min_val: float, max_val: float,
                  value: float = None, step: float = None,
                  key: str = None, help_text: Optional[str] = None) -> float:
    #创建滑块
    return st.slider(
        label = label,
        min_value = min_val,
        max_value = max_val,
        value = value,
        step = step,
        key = key,
        help = help_text
    )
