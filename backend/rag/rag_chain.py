"""RAG 链, 结合向量数据库和 LLM"""
from typing import List, Optional, Union
from langchain_classic.schema.document import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.chains import RetrievalQA

from backend.llm.llm_factory import get_llm
from .vector_store import VectorStoreManager

class RAGChain:
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.retriever = None
        self.qa_chain = None
        self.llm = None

    def setup_qa_chain(
            self,
            llm_provider: str = "langchain",
            model_name: str = "qwen-turbo",
            temperature: float = 0.1,
            max_tokens: int = 1000,
            k: int = 3
    ):
        #setup qusetion answering chain
        self.llm = get_llm(
            provider=llm_provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if self.vector_store_manager.vector_store is None:
            print("请先创建或加载向量存储")
            return False
        
        self.retriever = self.vector_store_manager.vector_store.as_retriever(search_kwargs={"k": k})

        prompt_template = """
        使用以下上下文来回答用户的问题。如果你不知道答案，请诚实地说你不知道，不要编造答案。
        回答应该简明扼要，并且基于提供的上下文。

        上下文:
        {context}

        问题:
        {question}

        回答:
        """
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm = self.llm,
            chain_type = "stuff",
            retriever = self.retriever,
            chain_type_kwargs = {"prompt": PROMPT},
            return_source_documents = True
        )

        return True
    
    def answer_question(self, question: str) -> dict:
        if not self.qa_chain:
            print("请先设置 QA 链")
            return {"answer": "QA 链未设置", "source_documents": []}
        
        try:
            result = self.qa_chain.invoke({"query": question})
            return {
                "answer": result.get("result", ""),
                "source_documents": result.get("source_documents", [])
            }
        except Exception as e:
            print(f"回答问题时出错: {e}")
            return {"answer": "回答问题时出错", "source_documents": []}
        
    def get_relevant_documents(self, query: str, k: int=3) -> List[Document]:
        #get and query relevan documents
        if self.retriever:
            return self.retriever.invoke(query)
        else:
            return self.vector_store_manager.similar_search(query, k=k)