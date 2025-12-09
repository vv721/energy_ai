import os
import dashscope
import time
import shutil
from typing import List, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from http import HTTPStatus

from ..config import (
    DASHSCOPE_API_KEY, OPENAI_API_KEY,
    DEFAULT_COLLECTION_NAME, VECTORSTORE_PATH
)
from ..exceptions import VectorStoreError, APIConnectionError
from ..utils import ensure_dir_exists, safe_file_opn, cleanup_resources, hash_text

class LocalEmbeddings(Embeddings):
    """本地简单嵌入模型（哈希+随机）- 用于演示和测试，不依赖外部服务。
    
    警告：这不是真正的语义嵌入，仅用于演示。生产环境应使用 OpenAI 或 DashScope。
    """
    
    def __init__(self, seed: int = 42):
        import numpy as np
        np.random.seed(seed)
        self.seed = seed
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表。"""
        return [hash_text(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询。"""
        return hash_text(text)


class DashScopeEmbeddings(Embeddings):
    #Packaging Aliyun DashScope Embeddings
    def __init__(self, model: str="text-embedding-v1", api_key: str=None):
        self.model = model
        self.api_key = api_key or DASHSCOPE_API_KEY
        if not self.api_key:
            raise VectorStoreError("DashScope API Key 未配置")
        dashscope.api_key = self.api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表。使用 DashScope 优先，回退到本地模型。"""
        # 预处理文本：截断到 8192 字符
        inputs = [text[:8192] if isinstance(text, str) else str(text)[:8192] for text in texts]
        
        # 首先尝试 DashScope
        try:
            #DashScope limit 25 per request
            batch_size = 25
            all_embedding = []

            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i+batch_size]
                resp = dashscope.TextEmbedding.call(
                    model=self.model,
                    input=batch
            )
                if resp.status_code == HTTPStatus.OK:
                    embeddings = resp.output.get("embeddings", [])
                    all_embedding.extend([item["embedding"] for item in embeddings])
                else:
                    error_msg = getattr(resp, "message", str(resp.status_code))
                    raise Exception(f"DashScope 嵌入失败：{error_msg}")
            return all_embedding

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
            raise VectorStoreError(f"查询嵌入失败：{e}")


class EmbeddingFactory:
    #创建嵌入模型，根据可用APIkey选择最佳选项

    @staticmethod
    def create_embeddings() -> Embeddings:
        #DashScope
        if DASHSCOPE_API_KEY and DASHSCOPE_API_KEY.strip():
            try:
                print("[信息] 使用 DashScope 嵌入模型")
                return DashScopeEmbeddings(model="text-embedding-v1", api_key=DASHSCOPE_API_KEY)
            except Exception as e:
                print(f"[警告] DashScope 初始化失败: {e}")

        #OpenAI
        if OPENAI_API_KEY and OPENAI_API_KEY.strip():
            try:
                print("[信息] 使用 OpenAI 嵌入模型")
                return OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
            except Exception as e:
                print(f"[警告] OpenAI 初始化失败: {e}")

        print("[信息] use local embeddings")
        return LocalEmbeddings()

class VectorStoreManager:
    """向量存储管理器，支持多种嵌入模型和向量数据库"""

    def __init__(self, persist_directory: str = VECTORSTORE_PATH):

        self.persist_directory = persist_directory
        ensure_dir_exists(self.persist_directory)

        #init embedding model
        self.embeddings = EmbeddingFactory.create_embeddings()
        self.vector_store = None

    def create_vector_store(self, documents: List[Document], collection_name: str = DEFAULT_COLLECTION_NAME) -> Chroma:
        try:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=self.persist_directory
            )
            return self.vector_store
        except Exception as e:
            raise VectorStoreError(f"创建向量存储失败：{e}")
    
    def load_vector_store(self, collection_name: str = DEFAULT_COLLECTION_NAME) -> Optional[Chroma]:
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
            raise VectorStoreError(f"加载向量存储失败：{e}")
    
    def add_documents(self, documents: List[Document]) -> bool:
        if self.vector_store is None:
            print("请先创建或加载向量存储。")
        
        try:
            self.vector_store.add_documents(documents)
            return True
        except Exception as e:
            raise VectorStoreError(f"添加文档到向量存储失败：{e}")
        
    def similar_search(self, query: str, k: int = 3) -> List[Document]:
        if self.vector_store is None:
            raise VectorStoreError("请先创建或加载向量存储")
        
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            raise VectorStoreError(f"相似度搜索失败：{e}")
    
    def similar_search_score(self, query: str, k: int = 3) -> List[tuple[Document, float]]:
        if self.vector_store is None:
            raise VectorStoreError("请先创建或加载向量存储")
        
        try:
            return self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            raise VectorStoreError(f"相似度搜索失败：{e}")
    
    def del_collection(self, collection_name: str = "default_collection"):
        """删除向量存储集合，返回详细的状态信息"""
        result = {
            "success": False,
            "messages": [],
            "errors": []
        }
        
        try:
            # 首先尝试删除向量存储集合
            if self.vector_store is not None:
                try:
                    # 强制删除ChromaDB集合
                    self.vector_store.delete_collection()
                    result["messages"].append(f"已删除集合: {collection_name}")
                except Exception as e:
                    warning_msg = f"删除集合时出现警告: {e}"
                    result["messages"].append(warning_msg)
                    print(warning_msg)
                finally:
                    # 显式关闭 Chroma 客户端连接
                    if hasattr(self, 'vector_store') and self.vector_store is not None:
                        try:
                            #更彻底的关闭链接
                            if hasattr(self.vector_store, '_client'):
                                self.vector_store._client.close()
                                if hasattr(self.vector_store._client, '_http_client'):
                                    self.vector_store._client._http_client.close()
                        except Exception as e:
                            pass  # 忽略关闭过程中的异常
        
                        self.vector_store = None

            #强制垃圾回收释放文件句柄
            cleanup_resources()

            # 删除持久化目录
            if os.path.exists(self.persist_directory):
                def delete_dir(dir_path):
                    shutil.rmtree(dir_path)

                try:
                    # 尝试直接删除整个目录
                    safe_file_opn(delete_dir, self.persist_directory)
                    result["messages"].append(f"已删除持久化目录: {self.persist_directory}")
                except Exception as e:
                    warning_msg = f"删除持久化目录时出现警告: {e}"
                    result["messages"].append(warning_msg)
                    print(warning_msg)
                    # 强制删除：逐个删除文件和目录，增加重试机制
                    try:
                        max_retries = 10
                        for attempt in range(max_retries):
                            try:
                                cleanup_resources()

                                for root, dirs, files in os.walk(self.persist_directory, topdown=False):
                                    for name in files:
                                        file_path = os.path.join(root, name)
                                        try:
                                            #尝试先修改文件属性
                                            if os.name == 'nt':
                                                try:
                                                    import ctypes
                                                    from ctypes import wintypes

                                                    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
                                                    kernel32.SetFileAttributesW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
                                                    kernel32.SetFileAttributesW.restype = wintypes.BOOL
                                                    
                                                    # 清除只读、隐藏、系统属性
                                                    kernel32.SetFileAttributesW(file_path, 0x80)  # FILE_ATTRIBUTE_NORMAL
                                                except:
                                                    pass

                                            os.remove(file_path)
                                            result["messages"].append(f"已删除文件: {file_path}")
                                            
                                        except Exception as e3:
                                                # 如果文件被锁定，等待更长时间后重试
                                            if attempt < max_retries - 1:
                                                time.sleep(1.0)  # 每次重试等待1秒
                                                continue
                                            else:
                                                error_msg = f"删除文件失败 {file_path}: {e3}"
                                                result["messages"].append(error_msg)
                                                print(error_msg)

                                    for name in dirs:
                                        dir_path = os.path.join(root, name)
                                        try:
                                            os.rmdir(dir_path)
                                            result["messages"].append(f"已删除目录: {dir_path}")
                                        except Exception as e4:
                                            #目录非空，继续尝试删除内容
                                            if attempt < max_retries - 1:
                                                time.sleep(1.0)  # 每次重试等待1秒
                                                continue
                                            else:
                                                error_msg = f"删除目录失败 {dir_path}: {e4}"
                                                result["messages"].append(error_msg)
                                                print(error_msg)

                                # 最后尝试强制删除目录
                                os.rmdir(self.persist_directory)
                                result["messages"].append(f"已强制删除持久化目录: {self.persist_directory}")
                                break  # 成功删除，跳出重试循环

                            except Exception as retry_e:
                                if attempt < max_retries - 1:
                                    time.sleep(1.0)  # 等待后重试
                                    continue
                                else:
                                    raise retry_e
                    except Exception as e2:
                        error_msg = f"强制删除目录失败: {e2}"
                        result["errors"].append(error_msg)
                        result["messages"].append(error_msg)
                        print(error_msg)
                        return result
            
            # 重新创建空目录
            try:
                ensure_dir_exists(self.persist_directory)
                result["messages"].append(f"已重新创建空目录: {self.persist_directory}")
            except Exception as e:
                error_msg = f"重新创建目录失败: {e}"
                result["errors"].append(error_msg)
                result["messages"].append(error_msg)
                print(error_msg)
                return result
            
            result["success"] = True
            return result       
        except Exception as e:
            error_msg = f"删除集合失败: {e}"
            result["errors"].append(error_msg)
            result["messages"].append(error_msg)
            print(error_msg)
            return result