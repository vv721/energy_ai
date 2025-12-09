"""RAG链模块，将检索到的文档与LLM结合生成回答。使用 LCEL (LangChain Expression Language) 实现。

相比旧的 PebbloRetrievalQA，LCEL 实现的优势：
- 支持所有向量存储（包括 Chroma）
- 更易自定义提示词和文档处理
- 更易返回源文档和中间结果
- 支持流式处理和异步操作
"""
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from backend.llm.llm_factory import get_llm
from .vector_store import VectorStoreManager

from ..config import RAG_PROMPT_TEMPLATE, DEFAULT_RETRIEVAL_K, get_llm_config
from ..exceptions import RAGChainError
from ..utils import format_docs
from ..llm.llm_factory import LLMFactory
from .vector_store import VectorStoreManager

class RAGChain:
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.retriever = None
        self.qa_chain = None
        self.llm = None

    def setup_qa_chain(
            self,
            llm_provider: str = None,
            model_name: str = None,
            temperature: float = None,
            max_tokens: int = None,
            k: int = DEFAULT_RETRIEVAL_K
        ) -> bool:
        """设置 QA 链（使用 LCEL 实现）。
        
        Args:
            llm_provider: LLM 提供者 ("openai", "langchain" 等)
            k: 检索时返回的文档数
            
        Returns:
            bool: 是否成功设置
        """
        # 获取 LLM 实例
        try:
            self.llm = get_llm(
                provider=llm_provider,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # 检查向量存储是否已初始化
            if self.vector_store_manager.vector_store is None:
                raise RAGChainError("请先创建或加载向量存储")
            
            # 创建检索器
            self.retriever = self.vector_store_manager.vector_store.as_retriever(
                search_kwargs={"k": k}
            )

            rag_prompt = PromptTemplate(
                template=RAG_PROMPT_TEMPLATE,
                input_variables=["context", "question"]
            )

            #构建 LCEL 链
            setup_and_retrieval = RunnableParallel(
            {"context": self.retriever | (lambda docs: format_docs(docs)), "question": RunnablePassthrough()}
            )
            
            def invoke_llm(inputs):
                prompt = rag_prompt.format(**inputs)
                return self.llm.chat(prompt)

            self.qa_chain = setup_and_retrieval | invoke_llm

            return True
        except Exception as e:
            raise RAGChainError(f"设置QA链时出错: {e}")       
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Args:
            question: 用户问题
            
        Returns:
            dict: 包含 'answer' 和 'source_documents' 的字典
        """
        if not self.qa_chain:
            raise RAGChainError("请先设置QA链")
        
        try:
            # 使用 invoke 调用链，传入问题
            answer = self.qa_chain.invoke({"question": question})
            
            # 获取源文档
            source_documents = self.get_relevant_documents(question)
            
            return {
                "answer": answer,
                "source_documents": source_documents
            }
        except Exception as e:
            raise RAGChainError(f"回答问题时出错: {e}")
        
    def get_relevant_documents(self, query: str, k: int = DEFAULT_RETRIEVAL_K) -> List[Document]:
        """获取与查询相关的文档。
        Args:
            query: 查询文本
            k: 返回的文档数
        Returns:
            list: 相关文档列表
        """
        try:
            if self.retriever:
                return self.retriever.invoke(query)
            else:
                return self.vector_store_manager.similar_search(query, k=k)
        except Exception as e:
            raise RAGChainError(f"获取相关文档时出错: {e}")