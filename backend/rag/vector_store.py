import os
from typing import List, Optional
from langchain_classic.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


class VectorStoreManager:
    #create, add, retrieve in vector store

    def __init__(self, persist_directory: str = "vectorstore"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("ALIYUN_API_KEY"))
        self.vector_store = None

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