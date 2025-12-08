import streamlit as st
import os
import sys
import pathlib as Path

#navigate to root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.rag import VectorStoreManager, DocumentProcessor
from langchain_core.documents import Document

@st.cache_resource
def init_vector_store_manager():
    vectorstore_path = os.path.join(PROJECT_ROOT, "vectorstore", "energy_docs")
    return VectorStoreManager(persist_directory=str(vectorstore_path))

def get_all_docs(vector_store_manager):
    vector_store = vector_store_manager.load_vector_store(collection_name="energy_docs")
    
    if vector_store is None:
        return []
    return []

def del_collection(vector_store_manager):
    # 清除当前会话状态中的RAG组件
    if "rag_components" in st.session_state:
        # 显式清理RAG组件中的资源
        if st.session_state.rag_components:
            try:
                # 关闭向量存储连接
                if "vector_store_manager" in st.session_state.rag_components:
                    vs_manager = st.session_state.rag_components["vector_store_manager"]
                    if hasattr(vs_manager, 'vector_store') and vs_manager.vector_store:
                        try:
                            if hasattr(vs_manager.vector_store, '_client'):
                                vs_manager.vector_store._client.close()
                        except:
                            pass
                        vs_manager.vector_store = None
            except:
                pass
        st.session_state.rag_components = None
    
    if "vector_store_loaded" in st.session_state:
        st.session_state.vector_store_loaded = False
    if "rag_initialized" in st.session_state:
        st.session_state.rag_initialized = False

    # 强制垃圾回收
    import gc
    gc.collect()
    
    # 等待一小段时间让资源释放
    import time
    time.sleep(0.2)

    # 调用后端删除方法并返回结果
    result = vector_store_manager.del_collection(collection_name="energy_docs")
    return result

def main():
    st.markdown("#RAG docs manager", unsafe_allow_html=True)

    vector_store_manager = init_vector_store_manager()
    vector_store = vector_store_manager.load_vector_store(collection_name="energy_docs")
    if vector_store is None:
        st.warning("No vector store found.")
    else:
        st.success("Vector store loaded successfully.")

    st.divider()

    st.subheader("docs list")

    if vector_store is not None:
        st.info(f"loaded vector store")

        try:
            sample_docs = vector_store_manager.similar_search("test", k=5)
            st.write(f"sample docs number: {len(sample_docs)}")

            for i, doc in enumerate(sample_docs):
                with st.expander(f"doc {i}"):
                    st.write(doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content)
        except Exception as e:
            st.error(f"Error loading docs: {e}")
    else:
        st.warning("No docs found.")

    st.divider()

    st.subheader("docs manager")

    if st.button("DELETE collection", type="primary"):
        if vector_store is not None:
            try:
                # 在删除前尝试清理可能占用资源的对象
                if 'vector_store_manager' in locals():
                    try:
                        if vector_store_manager.vector_store:
                            vector_store_manager.vector_store = None
                    except:
                        pass
                
                # 强制垃圾回收
                import gc
                gc.collect()
                gc.collect()
                
                # 等待一小段时间让资源释放
                import time
                time.sleep(0.5)
                
                # 调用删除函数并获取详细结果
                result = del_collection(vector_store_manager)
                
                # 显示删除过程的详细信息
                st.subheader("删除过程详情:")
                if isinstance(result, dict):  # 确保是字典格式
                    for message in result.get("messages", []):
                        if "失败" in message or "错误" in message:
                            st.error(message)
                        elif "警告" in message:
                            st.warning(message)
                        else:
                            st.success(message)
                    
                    if result.get("success", False):
                        st.success("Collection deleted successfully.")
                        # 清除所有缓存并重新初始化
                        st.cache_resource.clear()
                        init_vector_store_manager.clear()
                        # 清除会话状态
                        st.session_state.clear()
                        # 强制重新运行应用
                        st.rerun()  # 使用新的 rerun 方法替代 experimental_rerun
                    else:
                        st.error("Failed to delete collection.")
                        # 显示错误详情
                        st.error("删除过程中出现以下错误:")
                        for error in result.get("errors", []):
                            st.error(f"- {error}")
                else:  # 处理旧版本返回布尔值的情况
                    if result:
                        st.success("Collection deleted successfully.")
                        # 清除所有缓存并重新初始化
                        st.cache_resource.clear()
                        init_vector_store_manager.clear()
                        # 清除会话状态
                        st.session_state.clear()
                        # 强制重新运行应用
                        st.rerun()
                    else:
                        st.error("Failed to delete collection.")
            except Exception as e:
                st.error(f"Error deleting collection: {e}")

    if st.button("REFRESH", type="primary"):
        vector_store = vector_store_manager.load_vector_store(collection_name="energy_docs")
        if vector_store is not None:
            st.success("Vector store loaded successfully.")
        else:
            st.warning("No vector store found.")

        #update session state
        st.session_state.vector_store_loaded = vector_store is not None
        st.rerun()


if __name__ == "__main__":
    main()
