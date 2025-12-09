import os
from typing import Dict, Any, List, Optional

from ..config import VECTORSTORE_PATH, COLLECTION_NAME
from ..utils import safe_del_res, force_gc_coll, handle_exc


class RAGService:
    #封装与后端RAG组件的交互逻辑
    def __init__(self):
        self.vector_store_manager = None
        self.docs_processor = None
        self.rag_chain = None

    def init_vector_store_manager(self):
        try:
            from backend.rag import VectorStoreManager
            self.vector_store_manager = VectorStoreManager(persist_directory=VECTORSTORE_PATH)
            return self.vector_store_manager
        except Exception as e:
            handle_exc(e, "初始化向量存储管理器失败")
            return None
        
    def init_docs_processor(self):
        try:
            from backend.rag import DocumentProcessor
            self.docs_processor = DocumentProcessor()
            return self.docs_processor
        except Exception as e:
            handle_exc(e, "初始化文档处理器失败")
            return None
        
    def init_rag_chain(self):
        try:
            from backend.rag import RAGChain
            if not self.vector_store_manager:
                self.init_vector_store_manager()

            self.rag_chain = RAGChain(self.vector_store_manager)
            return self.rag_chain
        except Exception as e:
            handle_exc(e, "初始化RAG链失败")
            return None
        
    def load_vector_store(self):
        if not self.vector_store_manager:
            self.init_vector_store_manager()

        try:
            return self.vector_store_manager.load_vector_store(COLLECTION_NAME)
        except Exception as e:
            handle_exc(e, "加载向量存储失败")
            return None
        
    def get_sample_docs(self, query: str = "test", k: int = 5) ->List:
        if not self.vector_store_manager:
            self.init_vector_store_manager()
        try:
            return self.vector_store_manager.similar_search(query, k)
        except Exception as e:
            handle_exc(e, "获取样本文档失败")
            return None
        
    def del_coll(self) -> Dict[str, Any]:
        if not self.vector_store_manager:
            self.init_vector_store_manager()

        try:
            #clear resources
            safe_del_res(self.vector_store_manager)
            force_gc_coll()

            result = self.vector_store_manager.del_collection(COLLECTION_NAME)
            force_gc_coll()

            return result
        except Exception as e:
            handle_exc(e, "删除集合失败")
            return {"success": False, "error": str(e)}
        
    def add_docs(self, file_paths: List[str]) -> Dict[str, Any]:
        if not self.docs_processor:
            self.init_docs_processor()

        if not self.vector_store_manager:
            self.init_vector_store_manager()

        try:
            #处理文档
            docs = []
            for file_path in file_paths:
                doc = self.docs_processor.load_document(file_path)
                docs.extend(doc)

            chunks = self.docs_processor.split_documents(docs)

            result = self.vector_store_manager.add_documents(chunks)

            return result
        except Exception as e:
            handle_exc(e, "添加文档失败")
            return {"success": False, "error": str(e)}
        
    def answer_question(self, question: str, llm_provider: str = "openai",
                        model_name: str = None, temperature: float = 0.1,
                        max_tokens: int = 1024) -> Dict[str, Any]:
        if not self.rag_chain:
            self.init_rag_chain()

        try:
            #设置QA链
            self.rag_chain.setup_qa_chain(llm_provider=llm_provider,
                                          model_name=model_name,
                                          temperature=temperature,
                                          max_tokens=max_tokens)
            
            result = self.rag_chain.answer_question(question)
            return result
        except Exception as e:
            handle_exc(e, "回答问题失败")
            return {"answer": f"回答问题失败: {str(e)}", "source_docs": []}
