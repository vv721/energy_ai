"""文档处理器模块，加载，解析和处理文档"""

import os
import dashscope
from typing import List, Optional, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from http import HTTPStatus

from ..config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, SUPPORTED_DOCUMENT_EXTENSIONS, DASHSCOPE_API_KEY
from ..exceptions import DocumentProcessingError, APIConnectionError
from ..utils import validate_file_ext

#load and split documents
class DocumentProcessor:
    def __init__(self, chunk_size: int=DEFAULT_CHUNK_SIZE, chunk_overlap: int=DEFAULT_CHUNK_OVERLAP):

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
            raise DocumentProcessingError(f"文件不存在: {file_path}")
        
        if not validate_file_ext(file_path, SUPPORTED_DOCUMENT_EXTENSIONS):
            raise DocumentProcessingError(f"不支持的文件格式: {os.path.splitext(file_path)[1].lower()}")
        
        file_ext = os.path.splitext(file_path)[1].lower()

        try:
            if file_ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_ext in [".doc", ".docx"]:
                loader = Docx2txtLoader(file_path)
            else:
                raise DocumentProcessingError(f"不支持的文件格式: {file_ext}")
            
            return loader.load()
        except Exception as e:
            raise DocumentProcessingError(f"加载文档 {file_path} 时出错: {e}")
        
    def load_doc_from_dir(self, dir_path: str) -> List[Document]:
        #load all supported documents from a directory
        documents = []

        if not os.path.isdir(dir_path):
            raise DocumentProcessingError(f"目录不存在: {dir_path}")
        
        for file_name in os.listdir(dir_path):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in SUPPORTED_DOCUMENT_EXTENSIONS:
                file_path = os.path.join(dir_path, file_name)
                try:
                    docs = self.load_document(file_path)
                    if docs:
                        documents.extend(docs)
                except DocumentProcessingError as e:
                    print(f"警告:{e}")
                    continue

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
    
    def process_img_embed(self, image_path: str) -> Optional[List[float]]:
        #process image embedding
        try:
            if not os.path.exists(image_path):
                raise DocumentProcessingError(f"图片文件不存在: {image_path}")
            
            #use Aliyun enbedding model
            if not DASHSCOPE_API_KEY:
                raise ValueError("DashScope API Key 未配置")
                
            dashscope.api_key = DASHSCOPE_API_KEY

            input_data = [{"image": image_path}]

            resp = dashscope.MultiModalEmbedding.call(
                model='tongyi-embedding-vision-plus',
                input=input_data
            )

            if resp.status_code == HTTPStatus.OK:
                return resp.output['embeddings'][0]['embedding']
            else:
                raise APIConnectionError(f"图片嵌入生成失败:{resp.message}")
            
        except Exception as e:
            raise DocumentProcessingError(f"处理图片 {image_path} 时出错: {e}")
        
        