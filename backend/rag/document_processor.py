"""文档处理器模块，加载，解析和处理文档"""

import os
from typing import List, Optional, Union
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.schema.document import Document


#load and split documents
class DocumentProcessor:
    def __init__(self, chunk_size: int=1000, chunk_overlap: int=200):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_document(self, file_path: str) -> Optional[List[Document]]:
        #加载文档
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None
        
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_ext in [".doc", ".docx"]:
                loader = UnstructuredWordLoader(file_path)
            else:
                print(f"不支持的文件格式: {file_ext}")
                return None
            
            return loader.load()
        except Exception as e:
            print(f"加载文档 {file_path} 时出错: {e}")
            return None
        
    def load_doc_from_dir(self, dir_path: str) -> List[Document]:
        #load all supported documents from a directory
        documents = []
        supported_exts = [".pdf", ".txt", ".doc", ".docx"]

        if not os.path.isdir(dir_path):
            print(f"目录不存在: {dir_path}")
            return documents
        
        for file_name in os.listdir(dir_path):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in supported_exts:
                file_path = os.path.join(dir_path, file_name)
                docs = self.load_document(file_path)
                if docs:
                    documents.extend(docs)
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        #split documents into chunks
        return self.text_splitter.split_documents(documents)
    
    def process_single_docu(self, file_path: str) -> List[Document]:
        #load and split a single document
        documents = self.load_document(file_path)
        if not documents:
            return []
        return self.split_documents(documents)
    
    def process_docu_dir(self, dir_path: str) -> List[Document]:
        #load and split documents in a directory
        documents = self.load_doc_from_dir(dir_path)
        return self.split_documents(documents)