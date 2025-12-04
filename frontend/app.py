"""Streamlit 前端：简洁页面，允许选择 provider/model，输入 prompt 并展示回复。"""
import streamlit as st
from dotenv import load_dotenv
import os
import sys

from backend.rag import RAGChain, VectorStoreManager, DocumentProcessor
import tempfile

# Ensure project root is on sys.path so `from backend...` works when running
# the app via Streamlit or other runners whose CWD may differ.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from backend.llm.llm_factory import get_llm

st.set_page_config(page_title="能源AI助手", layout="wide", initial_sidebar_state="collapsed")

#initizlize RAG session state
if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False
    st.session_state.rag_components = None
    st.session_state.use_rag = False
    st.session_state.vector_store_loaded = False

st.markdown("""
    <style>
    /* 隐藏主菜单、页脚、部署按钮和所有可能的滚动条 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .reportview-container {background: white;}
    .stDeployButton {visibility: hidden;}
    
    /* 移除页面整体滚动条 */
    body {overflow: hidden;}
    .main {overflow: hidden;}
    
    /* 调整容器内边距，使其更紧凑 */
    .block-container {padding-top: 0rem; padding-bottom: 0rem;}
    
    /* 调整组件样式 */
    .stTextInput > label {font-size: 0.9rem;}
    .stButton > button {padding: 0.5rem 1rem;}
    
    /* 隐藏可能出现的额外空白元素 */
    .css-1lcbmhc {display: none;}
    .css-145kmo2 {display: none;}
    </style>
    """, unsafe_allow_html=True)

def initialize_rag():
    if not st.session_state.rag_initialized:
        vector_store_manager = VectorStoreManager(persist_directory="./vectorstore/energy_docs")
        vector_store = vector_store_manager.load_vector_store(collection_name="energy_docs")
        st.session_state.vector_store_loaded = vector_store is not None

        #create RAG components
        st.session_state.rag_components = {
            "doc_processor": DocumentProcessor(chunk_size=1000, chunk_overlap=200),
            "vector_store_manager": vector_store_manager,
            "rag_chain": RAGChain(vector_store_manager=vector_store_manager)
        }

        st.session_state.rag_initialized = True

    return st.session_state.rag_components

main_container = st.empty()

with main_container.container():
    st.markdown("## 能源AI助手", unsafe_allow_html=True)

    col1, col2 = st.columns([1,3], gap="small")

    with col1:
        with st.container(height=700):

            st.subheader("模型配置", divider="gray")
            provider = st.selectbox("模型提供者", 
                                    options=["OpenAI", "Aliyun"], 
                                    index=1,
                                    key="provider_select")
            
            if provider == "OpenAI":
                model_name = st.selectbox("模型名称", 
                                            options=["gpt-4", "gpt-3.5-turbo"], 
                                            index=1,
                                            key="model_name_select")
            if provider == "Aliyun":
                model_name = st.selectbox("模型名称", 
                                            options=["Qwen3", "qwen-turbo"], 
                                            index=1,
                                            key="model_name_select")

            st.subheader("参数设置", divider="gray")
            temperature = st.slider("temperature", 
                                    min_value=0.0, 
                                    max_value=1.0, 
                                    value=0.1, 
                                    step=0.1)
            max_tokens = st.number_input("max_tokens", 
                                        min_value=64, 
                                        max_value=4096, 
                                        value=int(os.getenv("MAX_TOKENS", 1000)), 
                                        step=64)
            
            st.subheader("RAG设置", divider="gray")
            use_rag = st.checkbox("启用RAG", value=st.session_state.use_rag)
            st.session_state.use_rag = use_rag
            
            if use_rag:
                rag_components = initialize_rag()
                if st.session_state.vector_store_loaded:
                    st.success("向量存储已加载")
                else:
                    st.warning("未加载向量存储")

                st.markdown("上传文档")
                uploaded_files = st.file_uploader(
                    "选择文档",
                    type=["pdf", "txt", "doc", "docx"],
                    accept_multiple_files=True,
                )

                if uploaded_files and st.button("处理文档"):
                    with st.spinner("..."):
                        try:
                            all_docs = []
                            # 存储临时文件路径，用于后续删除
                            temp_file_paths = []
                            
                            for uploaded_file in uploaded_files:
                                # 1. 创建临时文件（注意with语句结束后文件会自动关闭）
                                tmp_path = None
                                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    tmp_path = tmp_file.name
                                    temp_file_paths.append(tmp_path)
                                
                                # 2. 文件已关闭后再处理文档
                                if tmp_path:
                                    docs = rag_components["doc_processor"].process_single_docu(tmp_path)
                                    all_docs.extend(docs)
                            
                            # 3. 处理完所有文档后再创建向量存储
                            rag_components["vector_store_manager"].create_vector_store(all_docs, collection_name="energy_docs")
                            st.session_state.vector_store_loaded = True
                            st.success(f"成功处理{len(uploaded_files)}个文件")
                            
                            # 4. 最后删除所有临时文件
                            for tmp_path in temp_file_paths:
                                try:
                                    if os.path.exists(tmp_path):
                                        os.unlink(tmp_path)
                                except Exception as e:
                                    # 记录删除失败但不影响主流程
                                    print(f"删除临时文件失败: {e}")
                        except Exception as e:
                            # 清理临时文件
                            for tmp_path in temp_file_paths:
                                try:
                                    if os.path.exists(tmp_path):
                                        os.unlink(tmp_path)
                                except:
                                    pass
                            st.error(f"处理文档时出错: {e}")
            

    with col2:
        chat_container = st.container(height=570)
        with chat_container:
            st.subheader("对话历史", divider="gray")

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

        # message_container = st.container()
        # with message_container:
            if not st.session_state.chat_history:
                st.info("👋 您好！我是能源AI助手，有什么可以帮助您的吗？")

            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"**您：** {message['content']}")
                else:
                    st.markdown(f"**能源AI助手：** {message['content']}")

        if "prompt" not in st.session_state:
            st.session_state.prompt = ""
        
        # 使用表单确保输入框和按钮在同一区域
        with st.form(key="chat_form", clear_on_submit=True, height=100):
            # 使用列布局将输入框和按钮放在同一行
            input_col, button_col = st.columns([10, 1])
            
            # 文本输入框
            with input_col:
                prompt_input = st.text_input(
                    "",
                    value=st.session_state.prompt,
                    placeholder="请输入您的问题...",
                    label_visibility="collapsed",
                    key="prompt_input"
                )
                
            # 发送按钮 - type="primary" 使其为蓝色
            with button_col:
                submit_button = st.form_submit_button(
                    "发送",
                    type="primary",
                    use_container_width=True  # 按钮占据整个列宽，与输入框高度一致
                )
        
        # 处理表单提交
        if submit_button and prompt_input.strip():
            # 添加用户消息到历史
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt_input
            })
            
            try:
                # 创建LLM
                llm = get_llm(
                    provider=provider,
                    model_name=model_name, 
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # 调用LLM获取回复
                with st.spinner("正在生成回复..."):
                    if st.session_state.use_rag and st.session_state.rag_components and st.session_state.vector_store_loaded:
                        rag_chain = st.session_state.rag_components["rag_chain"]
                        #rag_setup
                        rag_chain.setup_qa_chain(
                            llm_provider = "langchain",
                            model_name = model_name,
                            temperature = temperature,
                            max_tokens = max_tokens,
                            k =3
                        )

                        result = rag_chain.answer_question(prompt_input)
                        resp = result["answer"]
                    else:
                        resp = llm.chat(prompt_input)
                    
                    # 添加AI回复到历史
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": resp
                    })
                
                # 清空输入框
                st.session_state.prompt = ""
                
                # 刷新页面以显示新消息
                st.rerun()
                
            except Exception as e:
                st.error(f"调用出错: {e}")