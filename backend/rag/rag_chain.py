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

class RAGChain:
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.retriever = None
        self.qa_chain = None
        self.llm = None

    def setup_qa_chain(
            self,
            llm_provider: str = "openai",
            model_name: str = "gpt-3.5-turbo",
            temperature: float = 0.1,
            max_tokens: int = 1000,
            k: int = 3
    ) -> bool:
        """设置 QA 链（使用 LCEL 实现）。
        
        Args:
            llm_provider: LLM 提供者 ("openai", "langchain" 等)
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大生成 tokens
            k: 检索时返回的文档数
            
        Returns:
            bool: 是否成功设置
        """
        # 获取 LLM 实例
        self.llm = get_llm(
            provider=llm_provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # 检查向量存储是否已初始化
        if self.vector_store_manager.vector_store is None:
            print("请先创建或加载向量存储")
            return False
        
        # 创建检索器
        self.retriever = self.vector_store_manager.vector_store.as_retriever(
            search_kwargs={"k": k}
        )

        # 定义 RAG 提示词
        rag_prompt_template = """使用以下上下文来回答用户的问题。
如果你不知道答案，请诚实地说你不知道，不要编造答案。
回答应该简明扼要，并且基于提供的上下文。

上下文：
{context}

问题：{question}

回答："""
        
        rag_prompt = PromptTemplate(
            template=rag_prompt_template,
            input_variables=["context", "question"]
        )

        # 使用 LCEL 构建 RAG 链
        # 格式化函数：将检索到的文档列表转换为字符串
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        # 使用 RunnablePassthrough 构建简单的 RAG 链
        # 输入：{"question": "用户问题"}
        # 输出：生成的回答
        
        # 构建 LCEL 链：
        # chain_input = {"question": question_text}
        # retriever.invoke(question) → 返回 List[Document]
        # format_docs() → 返回字符串
        # 最终传给提示词的是 {"context": docs_string, "question": question_text}
        
        setup_and_retrieval = RunnableParallel(
            {"context": self.retriever | (lambda docs: format_docs(docs)), "question": RunnablePassthrough()}
        )
        
        # self.qa_chain = setup_and_retrieval | rag_prompt | self.llm | StrOutputParser()
        def invoke_llm(inputs):
            prompt = rag_prompt.format(**inputs)
            return self.llm.chat(prompt)

        self.qa_chain = setup_and_retrieval | invoke_llm

        return True
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """回答问题。
        
        Args:
            question: 用户问题
            
        Returns:
            dict: 包含 'answer' 和 'source_documents' 的字典
        """
        if not self.qa_chain:
            print("请先设置 QA 链")
            return {"answer": "QA 链未设置", "source_documents": []}
        
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
            print(f"回答问题时出错: {e}")
            import traceback
            traceback.print_exc()
            return {"answer": f"回答问题时出错: {str(e)}", "source_documents": []}
        
    def get_relevant_documents(self, query: str, k: int = 3) -> List[Document]:
        """获取与查询相关的文档。
        
        Args:
            query: 查询文本
            k: 返回的文档数
            
        Returns:
            list: 相关文档列表
        """
        if self.retriever:
            try:
                return self.retriever.invoke(query)
            except:
                pass
        
        return self.vector_store_manager.similar_search(query, k=k)