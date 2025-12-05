import os
import dashscope
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from http import HTTPStatus
import hashlib
import numpy as np


class LocalEmbeddings(Embeddings):
    """本地简单嵌入模型（哈希+随机）- 用于演示和测试，不依赖外部服务。
    
    警告：这不是真正的语义嵌入，仅用于演示。生产环境应使用 OpenAI 或 DashScope。
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed
    
    def _hash_to_embedding(self, text: str, dim: int = 384) -> List[float]:
        """将文本哈希转换为嵌入向量（用于确定性）。"""
        text_hash = hashlib.md5(text.encode()).digest()
        np.random.seed(int.from_bytes(text_hash[:4], 'big'))
        return np.random.randn(dim).tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表。"""
        return [self._hash_to_embedding(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询。"""
        return self._hash_to_embedding(text)





class DashScopeEmbeddings(Embeddings):
    #Packaging Aliyun DashScope Embeddings
    def __init__(self, model: str="text-embedding-v1", api_key: str=None):
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DashScope API Key 未配置")
        dashscope.api_key = self.api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表。使用 DashScope 优先，回退到本地模型。"""
        # 预处理文本：截断到 8192 字符
        inputs = [text[:8192] if isinstance(text, str) else str(text)[:8192] for text in texts]
        
        # 首先尝试 DashScope
        try:
            resp = dashscope.TextEmbedding.call(
                model=self.model,
                input=inputs
            )

            if resp.status_code == HTTPStatus.OK:
                embeddings = resp.output.get("embeddings", [])
                return [item["embedding"] for item in embeddings]
            else:
                error_msg = getattr(resp, "message", str(resp.status_code))
                raise Exception(f"DashScope 嵌入失败：{error_msg}")
        except Exception as e:
            print(f"[警告] DashScope 嵌入失败：{e}，尝试本地模型...")
            
            # 回退到本地模型
            try:
                from sentence_transformers import SentenceTransformer
                print("[信息] 正在加载本地模型...")
                model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                embeddings = model.encode(inputs, show_progress_bar=False)
                return embeddings.tolist()
            except Exception as fallback_e:
                print(f"[错误] 本地模型嵌入失败：{fallback_e}")
                raise RuntimeError(f"所有嵌入方法都失败了。DashScope: {e}, 本地模型: {fallback_e}")
            
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本。"""
        try:
            result = self.embed_documents([text])
            if result and len(result) > 0:
                return result[0]
            else:
                raise ValueError("嵌入返回为空")
        except Exception as e:
            print(f"[错误] 查询嵌入失败：{e}")
            raise


class VectorStoreManager:
    """向量存储管理器，支持 DashScope 或 OpenAI 嵌入。"""

    def __init__(self, persist_directory: str = "vectorstore"):
        self.persist_directory = persist_directory
        
        # 优先级：OpenAI > DashScope > 本地模型
        self.embeddings = self._init_embeddings()
        self.vector_store = None
    
    def _init_embeddings(self):
        """初始化嵌入模型，根据可用的 API Key 选择最佳选项。
        
        优先级：
        1. DashScope (Aliyun) - 如果配置了有效的 API Key
        2. OpenAI - 如果配置了有效的 API Key  
        3. 本地模型 - 作为后备方案
        """
        # 首选：DashScope (阿里云) - 优先级提高
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        # 检查 API Key 是否有效（不是 "your_key" 这样的默认值）
        if dashscope_key and dashscope_key.strip() and not dashscope_key.startswith("your"):
            try:
                print("[信息] 使用 DashScope 嵌入模型")
                return DashScopeEmbeddings(model="text-embedding-v1", api_key=dashscope_key)
            except Exception as e:
                print(f"[警告] DashScope 初始化失败: {e}")
        
        # 备选：OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        # 检查 API Key 是否有效（不是 "your_key" 这样的默认值）
        if openai_key and openai_key.strip() and not openai_key.startswith("your"):
            try:
                print("[信息] 使用 OpenAI 嵌入模型")
                return OpenAIEmbeddings(api_key=openai_key, model="text-embedding-3-small")
            except Exception as e:
                print(f"[警告] OpenAI 初始化失败: {e}")
        
        # 回退：本地模型（不下载，使用内存中的）
        print("[信息] 使用本地嵌入模型")
        return LocalEmbeddings()

    def create_vector_store(self, documents: List[Document], collection_name: str = "default_collection"):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=collection_name,
            persist_directory=self.persist_directory
        )
        return self.vector_store
    
    def load_vector_store(self, collection_name: str = "default_collection") -> Optional[Chroma]:
        if not os.path.exists(self.persist_directory):
            print(f"向量存储目录不存在： {self.persist_directory}")
            return None
        
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            return self.vector_store
        except Exception as e:
            print(f"加载向量存储失败：{e}")
            return None
    
    def add_documents(self, documents: List[Document]):
        if self.vector_store is None:
            print("请先创建或加载向量存储。")
            return False
        
        try:
            self.vector_store.add_documents(documents)
            return True
        except Exception as e:
            print(f"添加文档到向量存储失败：{e}")
            return False
        
    def similar_search(self, query: str, k: int = 3) -> List[Document]:
        if self.vector_store is None:
            print("请先创建或加载向量存储")
            return []
        
        return self.vector_store.similarity_search(query, k=k)
    
    def similar_search_score(self, query: str, k: int = 3) -> List[tuple[Document, float]]:
        if self.vector_store is None:
            print("请先创建或加载向量存储")
            return []
        
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def del_collection(self, collection_name: str = "default_collection"):
        if self.vector_store is None:
            self.load_vector_store(collection_name)

        if self.vector_store:
            try:
                self.vector_store.delete_collection()
                self.vector_store = None
                print(f"已删除集合: {collection_name}")
                return True
            except Exception as e:
                print(f"删除集合失败: {e}")
                return False
        return False