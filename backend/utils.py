import os
import time
import gc
import hashlib
import numpy as np
from typing import List

def ensure_dir_exists(dir_path: str) -> None:
    #确保dir存在
    os.makedirs(dir_path, exist_ok=True)


def safe_file_opn(opn_func, file_path: str, max_retries: int = 3, retry_delay: float = 1.0):
    #safe file open with retry
    last_error = None
    for attempt in range(max_retries):
        try:
            return opn_func(file_path)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                gc.collect()
    raise last_error


def hash_text(text: str, dim: int = 384) -> List[float]:
    #hash text to vector
    text_hash = hashlib.md5(text.encode()).digest()
    np.random.seed(int.from_bytes(text_hash[:4], byteorder="big"))
    return np.random.rand(dim).tolist()


def validate_file_ext(file_path: str, allowed_extensions: List[str]) -> bool:
    #验证文件拓展名是否在列表
    file_ext = os.path.splitext(file_path)[1].lower()
    return file_ext in allowed_extensions


def truncate_text(text: str, max_len: int = 8192) -> str:
    #截断文本指定长度
    return text[:max_len] if isinstance(text, str) else (text)[:max_len]


def format_docs(docs: List) -> str:
    #格式化文档列表为字符串
    return "\n\n".join(doc.page_content for doc in docs)


def cleanup_resources() -> None:
    #清理资源
    gc.collect()
    time.sleep(0.2)