import os
from typing import Optional, Dict
from camel.storages import QdrantStorage
from camel.embeddings import SentenceTransformerEncoder
from camel.storages import VectorRecord

class QdrantDB:
    """Qdrant向量数据库操作类"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B"):
        """
        初始化Qdrant数据库
        
        任务：
        1. 设置数据存储路径
        2. 初始化embedding模型
        3. 创建QdrantStorage实例
        
        参数:
            model_name: huggingface模型名称
        """
        # 设置rootpath（数据存储根目录）
        rootpath = os.path.dirname(__file__)
        self.storage_path = os.path.join(rootpath, "qdrant_data")
        
        # 初始化SentenceTransformerEncoder 
        self.embedding_instance = SentenceTransformerEncoder(
            model_name=model_name,
            device='cpu'  
        )
        
        # 初始化QdrantStorage
        vector_dim = self.embedding_instance.get_output_dim()
        self.storage_instance = QdrantStorage(
            vector_dim=vector_dim,
            path=self.storage_path,
            collection_name="rag_collection"
        )
        
    def save_text(self, text: str, source_file: str = "unknown", payload_extra: Optional[Dict] = None):
        """
        保存单个文本到数据库
        
        任务：
        1. 将文本转换为向量
        2. 创建VectorRecord
        3. 保存到数据库
        
        参数:
            text: 要保存的文本
            source_file: 文本来源文件名
            payload_extra: 额外的载荷元数据（如页码、分块信息等）
        """
        # 跳过空文本
        if not text or not text.strip():
            return
            
        # 使用embedding_instance将文本转换为向量
        embeddings = self.embedding_instance.embed_list([text])
        vector = embeddings[0]
        
        # 创建payload字典，包含text和source_file信息
        payload = {
            "text": text,
            "source_file": source_file
        }
        if payload_extra:
            payload.update(payload_extra)
        
        # 创建VectorRecord对象
        record = VectorRecord(
            vector=vector,
            payload=payload
        )
        
        # 使用storage_instance.add()保存记录
        self.storage_instance.add([record])

