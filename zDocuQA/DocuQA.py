import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置页面配置
st.set_page_config(page_title="Document_Q&A", layout="wide")
st.title("Document_Q&A")

# 文件上传组件
uploaded_files = st.file_uploader(
    label="Upload a document (PDF or TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True)

if not uploaded_files:
    st.info("Please upload at least one document to proceed.")
    st.stop()

# 配置检索器函数，使用缓存优化性能
@st.cache_data(ttl="1h")
def configure_retriever(upload_files):
    all_texts = []
    temp_dir = tempfile.TemporaryDirectory(dir=".")
    
    for file in upload_files:
        temp_path = os.path.join(temp_dir.name, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        
        if file.name.endswith(".txt"):
            # 多编码尝试机制
            encodings_to_try = ["utf-8", "gbk", "gb2312", "latin-1"]
            document_loaded = False
            
            for encoding in encodings_to_try:
                try:
                    loader = TextLoader(temp_path, encoding=encoding)
                    loaded_docs = loader.load()
                    all_texts.extend(loaded_docs)
                    document_loaded = True
                    break
                except UnicodeDecodeError:
                    continue
            
            if not document_loaded:
                # 降级处理
                loader = TextLoader(temp_path, encoding="utf-8", errors="replace")
                loaded_docs = loader.load()
                all_texts.extend(loaded_docs)
                st.warning(f"Warning: File {file.name} contains non-UTF-8 characters that were replaced.")
        else:
            # PDF文件处理
            loader = PyPDFLoader(temp_path)
            all_texts.extend(loader.load())

    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_texts)

    # 创建向量数据库
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(splits, embeddings)

    # 返回检索器
    retriever = vectordb.as_retriever()
    return retriever

# 初始化检索器
retriever = configure_retriever(uploaded_files)

# 会话状态管理
if "messages" not in st.session_state or st.sidebar.button("Clear Chat History"):
    st.session_state["messages"] = [{"role": "system", "content": "I am a helpful assistant that answers questions based on the uploaded documents."}]

# 显示聊天历史
for msg in st.session_state["messages"][1:]:  # 跳过系统消息
    st.chat_message(msg["role"]).write(msg["content"])

# 创建提示模板
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based solely on the provided context.
If you don't know the answer from the context, say "I don't have enough information to answer that question from the uploaded documents."

Context:
{context}

Question: {question}

Answer: """)

# 初始化LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 创建RAG链
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 用户输入处理
user_query = st.chat_input("Enter your question:")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        config = {"callbacks": [st_cb]}
        try:
            # 直接使用RAG链回答问题
            response = rag_chain.invoke(user_query, config=config)
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.write(response)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state["messages"].append({"role": "assistant", "content": error_msg})