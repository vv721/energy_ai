import streamlit as st
import sys
import os

# 确保项目根目录在sys.path中，以便可以导入backend模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 设置页面配置
st.set_page_config(page_title="能源AI系统", layout="wide")

# 初始化会话状态，用于管理当前选中的功能页
if "selected_page" not in st.session_state:
    st.session_state.selected_page = 0

# 创建左右分隔的布局
left_col, right_col = st.columns([1, 6], gap="small")

# 左侧选项标签区域 - 实现硬朗直角设计
with left_col:
    # 使用更直接的方式实现直角容器
    st.markdown("## 功能导航")
    
    # 创建四个直角按钮，使用紧凑布局
    if st.button("能源AI助手", use_container_width=True, type="primary"):
        st.session_state.selected_page = 0
    
    # 使用最小间距
    st.text("")
    
    # 其他三个空白选项
    if st.button("功能二", use_container_width=True):
        st.session_state.selected_page = 1
    
    st.text("")
    
    if st.button("功能三", use_container_width=True):
        st.session_state.selected_page = 2
    
    st.text("")
    
    if st.button("RAG manager", use_container_width=True):
        st.session_state.selected_page = 3

# 右侧内容区域
with right_col:
    if st.session_state.selected_page ==0:
        try:
            from frontend import app
            app.main()
        except Exception as e:
            st.error(f"加载能源AI助手页面时出错: {e}")
    
    elif st.session_state.selected_page == 1:
        st.markdown("### 功能二")
        st.info("此功能正在开发中...")
    
    elif st.session_state.selected_page == 2:
        st.markdown("### 功能三")
        st.info("此功能正在开发中...")
    
    elif st.session_state.selected_page == 3:
        try:
            from frontend import rag_manager
            rag_manager.main()
        except Exception as e:
            st.error(f"加载RAG manager页面时出错: {e}")