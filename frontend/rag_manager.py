import streamlit as st

from .config import VECTORSTORE_PATH, COLLECTION_NAME
from .components import create_docs_expander, create_act_btns, create_stat_indicator
from .services import RAGService, StateManager
from .utils import format_result_msg, reset_app_state


@st.cache_resource
def get_rag_service():
    #获取RAG服务实例（缓存）
    return RAGService()


def main():

    st.markdown("#RAG docs manager", unsafe_allow_html=True)

    rag_service = get_rag_service()

    vector_store = rag_service.load_vector_store()

    if vector_store is None:
        create_stat_indicator("warning", "未找到向量存储")
    else:
        create_stat_indicator("success", "向量存储加载成功")

    st.divider()

    st.subheader("文档列表")

    if vector_store is not None:
        st.info("已加载向量存储")

        try:
            sample_docs = rag_service.get_sample_docs()
            st.write(f"示例文档数量: {len(sample_docs)}")

            for i, doc in enumerate(sample_docs):
                create_docs_expander(i, doc.page_content)
        except Exception as e:
            st.error(f"加载文档时出错 {e}")
    else:
        create_stat_indicator("warning", "未找到文档")

    st.divider()

    st.subheader("文档管理")

    def del_coll(rag_service):
        #删除集合
        vector_store = rag_service.load_vector_store()
        if vector_store is None:
            create_stat_indicator("warning", "向量存储未初始化")
            return
        
        #清楚当前会话状态的RAG组件
        if "rag_components" in st.session_state:
            StateManager.set_rag_components({})

        #更新状态
        StateManager.set_vector_store_loaded(False)
        StateManager.set_rag_initialized(False)

        result = rag_service.del_coll()

        st.subheader("删除过程详情：")
        format_result_msg(result)

        if result.get("success", False):
            create_stat_indicator("success", "集合删除成功")
            #重置应用状态
            reset_app_state()
        else:
            create_stat_indicator("error", "集合删除失败")

    def refresh_vector_store(rag_service):
        #刷新向量存储
        vector_store = rag_service.load_vector_store()

        if vector_store is not None:
            create_stat_indicator("success", "向量存储刷新成功")
        else:
            create_stat_indicator("error", "向量存储刷新失败")

        #更新状态
        StateManager.set_vector_store_loaded(vector_store is not None)
        st.rerun()


    buttons = [
        {
            "name": "删除集合",
            "type": "primary",
            "callback": del_coll,
            "args": {"rag_service": rag_service}
        },
        {
            "name": "刷新",
            "type": "primary",
            "callback": refresh_vector_store,
            "args": {"rag_service": rag_service}
        }
    ]

    create_act_btns(buttons)

if __name__ == "__main__":
    main()
