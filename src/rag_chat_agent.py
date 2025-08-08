"""
基于 CAMEL-AI 框架的智能RAG对话代理
"""

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.messages import BaseMessage
from camel.types import RoleType
from .vector_retriever import VecRetriever, KeywordRetriever, EnhancedHybridRetriever, PineconeCohereReranker
from .qdrant import QdrantDB
import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from .document_manager import DocumentManager
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

 


@dataclass
class RAGConfig:
    """RAG系统配置类"""
    
    # 模型配置
    model_name: str = "gpt-5"
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    reranker_model: str = "cohere-rerank-3.5"
    
    # 检索配置
    vector_weight: float = 0.5
    keyword_weight: float = 0.5
    rrf_k: int = 20

    
    # 数据源配置
    ocr_data_path: str = "data/small_ocr_content_list.json"
    
    # 重排配置
    enable_reranking: bool = True
    
    
    # 系统提示词配置
    system_message: str = """你是一位研究马克思《资本论》的专家。你的任务是基于提供的上下文信息来回答用户的问题。

请遵循以下规则：
1. 仔细阅读提供的上下文信息
2. 严格基于上下文内容回答用户问题
3. 如果上下文中没有相关信息，请明确说明"根据提供的信息，我无法回答这个问题"
4. 回答要准确、简洁、有条理
5. 可以引用上下文中的具体内容来支持你的回答
6. 使用中文回答

你现在准备好回答基于上下文的问题了。"""



class RetrievalManager:
    """检索管理器，封装和管理各种检索器"""
    
    def __init__(self, config: RAGConfig, document_manager: DocumentManager, 
                 qdrant_db, pinecone_api_key: Optional[str] = None):
        self.config = config
        self.document_manager = document_manager
        self.qdrant_db = qdrant_db
        
        # 初始化各个检索器
        self.vector_retriever = VecRetriever(qdrant_db)
        self.keyword_retriever = KeywordRetriever()
        
        # 初始化关键词检索器的文档数据
        self._initialize_keyword_retriever()
        
        # 创建外部重排器
        external_reranker = None
        if self.config.enable_reranking:
            pinecone_key = (pinecone_api_key or os.getenv("PINECONE_API_KEY", "")).strip()
            if pinecone_key:
                try:
                    external_reranker = PineconeCohereReranker(
                        api_key=pinecone_key, 
                        model=self.config.reranker_model
                    )
                except Exception:
                    external_reranker = None

        # 创建混合检索器
        self.hybrid_retriever = EnhancedHybridRetriever(
            vector_retriever=self.vector_retriever,
            keyword_retriever=self.keyword_retriever,
            rrf_k=self.config.rrf_k,
            vector_weight=self.config.vector_weight,
            keyword_weight=self.config.keyword_weight,
            external_reranker=external_reranker,
        )
    
    def _initialize_keyword_retriever(self):
        """初始化关键词检索器的文档数据"""
        try:
            documents = self.document_manager.load_documents_from_json()
            if documents:
                self.keyword_retriever.add_documents(documents)
        except Exception:
            pass
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """统一的检索接口（仅使用 top_k）"""
        return self.hybrid_retriever.search(
            query=query,
            top_k=top_k,
        )
    
    def search_with_scores(self, query: str, top_k: int = 5) -> Dict:
        """带详细分数信息的检索接口（仅使用 top_k）"""
        return self.hybrid_retriever.search_with_scores(
            query=query,
            top_k=top_k,
        )
    



class RAGChatAgent:
    """智能RAG对话代理"""
    
    def __init__(self, 
                 config: Optional[RAGConfig] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 pinecone_api_key: Optional[str] = None):
        """
        初始化RAG对话代理
        
        参数:
            config: RAG系统配置对象
            api_key: API密钥
            base_url: API基础URL（如果使用兼容的模型）
            pinecone_api_key: Pinecone API密钥（用于重排）
        """
        # 使用默认配置或传入的配置
        self.config = config or RAGConfig()
        
        # 初始化文档管理器
        self.document_manager = DocumentManager(self.config)
        
        # 1. 初始化向量数据库
        self.qdrant_db = QdrantDB(model_name=self.config.embedding_model)
        
        # 2. 初始化检索管理器
        self.retrieval_manager = RetrievalManager(
            config=self.config,
            document_manager=self.document_manager,
            qdrant_db=self.qdrant_db,
            pinecone_api_key=pinecone_api_key
        )
        
        # 3. 创建语言模型
        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=self.config.model_name,
            api_key=api_key or os.getenv('OPENAI_API_KEY', '')
        )
        
        
        # 4. 创建ChatAgent
        self.chat_agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name="RAG助手",
                content=self.config.system_message
            ),
            model=self.model
        )
    

    
    def query(self, question: str, top_k: int = 3) -> Dict[str, str]:
        """
        处理用户查询
        
        参数:
            question: 用户问题
            top_k: 检索的文档数量
            
        返回:
            包含回答和检索到的上下文的字典
        """
        try:
            # 1) 检索
            retrieved_docs = self.retrieval_manager.search(
                query=question,
                top_k=top_k
            )

            # 2) 构建上下文与来源
            context = self._build_context(retrieved_docs)
            sources = self._extract_sources(retrieved_docs)

            # 3) 生成提示词并向LLM提问
            user_prompt = self._create_user_prompt(context, question)
            answer = self._ask_llm(user_prompt)

            result: Dict[str, any] = {
                "answer": answer,
                "context": context,
                "sources": sources,
                "question": question,
            }
            # 如果检索结果包含相似度分数，附带展示仅存在的分数字段
            if retrieved_docs and isinstance(retrieved_docs[0], dict) and 'score' in retrieved_docs[0]:
                # 仅加入一个示例最高分
                top_score = max((d.get('score') for d in retrieved_docs if isinstance(d.get('score'), (int, float))), default=None)
                if top_score is not None:
                    result["vector_top_score"] = top_score
            return result

        except Exception as e:
            return {
                "answer": f"处理查询时出现错误: {str(e)}",
                "context": "",
                "sources": [],
                "question": question,
            }
    
    def chat(self, question: str, top_k: int = 3) -> str:
        """
        简化的聊天接口，只返回回答
        
        参数:
            question: 用户问题
            top_k: 检索的文档数量
            
        返回:
            AI回答
        """
        result = self.query(question, top_k)
        return result["answer"]
    
    def query_with_stats(self, question: str, top_k: int = 3) -> Dict:
        """
        带详细检索统计信息的查询
        
        参数:
            question: 用户问题
            top_k: 检索的文档数量
            
        返回:
            包含回答和详细检索统计的字典
        """
        try:
            detailed_results = self.retrieval_manager.search_with_scores(
                query=question,
                top_k=top_k
            )

            retrieved_docs = detailed_results["reranked_results"]
            context = self._build_context(retrieved_docs)
            user_prompt = self._create_user_prompt(context, question)
            answer = self._ask_llm(user_prompt)
            sources = self._extract_sources(retrieved_docs)

            result: Dict[str, any] = {
                "answer": answer,
                "context": context,
                "sources": sources,
                "question": question,
                "retrieval_stats": detailed_results.get("retrieval_stats", {}),
                "vector_results_count": detailed_results.get("vector_results_count", 0),
                "keyword_results_count": detailed_results.get("keyword_results_count", 0),
            }
            # 只展示已存在的分数字段
            if retrieved_docs and isinstance(retrieved_docs[0], dict):
                top_score = max((d.get('score') for d in retrieved_docs if isinstance(d.get('score'), (int, float))), default=None)
                if top_score is not None:
                    result["vector_top_score"] = top_score
            return result

        except Exception as e:
            return {
                "answer": f"处理查询时出现错误: {str(e)}",
                "context": "",
                "sources": [],
                "question": question,
                "retrieval_stats": {},
                "vector_results_count": 0,
                "keyword_results_count": 0,
            }

    def _format_score_info(self, doc: Dict) -> str:
        """返回文档的可读评分信息。"""
        if "weighted_rrf_score" in doc:
            return f" (RRF得分: {doc['weighted_rrf_score']:.3f})"
        if "rrf_score" in doc:
            return f" (RRF得分: {doc['rrf_score']:.3f})"
        return ""

    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """根据检索结果构建用于提示词的上下文字符串。"""
        context_parts: List[str] = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc.get("content", "")
            source = doc.get("file_name", "未知来源")
            score_info = self._format_score_info(doc)
            # 提取元数据中的页码/分块信息
            meta = doc.get("metadata") or {}
            page_idx = meta.get("page_idx")
            chunk_index = meta.get("chunk_index")
            chunk_count = meta.get("chunk_count")
            # Fallback: 从来源名解析（OCR_page_{page}_{type}_ch{i}of{n}）
            if not isinstance(page_idx, int) or not isinstance(chunk_index, int) or not isinstance(chunk_count, int):
                import re
                m = re.search(r"OCR_page_(\d+)_\w+(?:_ch(\d+)of(\d+))?", source)
                if m:
                    if page_idx is None:
                        try:
                            page_idx = int(m.group(1))
                        except Exception:
                            pass
                    if (chunk_index is None or chunk_count is None) and m.group(2) and m.group(3):
                        try:
                            chunk_index = int(m.group(2))
                            chunk_count = int(m.group(3))
                        except Exception:
                            pass
            meta_str_parts = []
            if isinstance(page_idx, int):
                meta_str_parts.append(f"页码: {page_idx}")
            if isinstance(chunk_index, int) and isinstance(chunk_count, int):
                meta_str_parts.append(f"分块: {chunk_index}/{chunk_count}")
            meta_str = (" | ".join(meta_str_parts)) if meta_str_parts else ""
            meta_suffix = f" | {meta_str}" if meta_str else ""
            context_parts.append(f"[文档{i}]{score_info} 来源: {source}{meta_suffix}\n内容: {content}")
        return "\n\n".join(context_parts)

    def _create_user_prompt(self, context: str, question: str) -> str:
        """构建用户提示词。若无上下文则给出无法回答的指引。"""
        if context.strip():
            return (
                f"""上下文信息：
{context}

用户问题：{question}

请基于以上上下文信息回答用户的问题。"""
            )
        return (
            f"""没有找到相关的上下文信息。

用户问题：{question}

请告诉用户你无法基于现有信息回答这个问题。"""
        )

    def _ask_llm(self, user_prompt: str) -> str:
        """向LLM发送消息并返回回答文本。"""
        user_message = BaseMessage.make_user_message(
            role_name="用户",
            content=user_prompt,
        )
        response = self.chat_agent.step(user_message)
        return response.msg.content

    def _reconstruct_source_info(self, content: str) -> Dict:
        """基于文本内容重建页码和分块信息"""
        return self.document_manager.reconstruct_source_info(content)

    def _extract_sources(self, retrieved_docs: List[Dict]) -> List[str]:
        """从检索结果中提取来源，并附带页码/分块等元数据。"""
        sources: List[str] = []
        for i, doc in enumerate(retrieved_docs, 1):
            source_preview = doc.get("file_name", "").strip()
            # 若来源无效，则用内容预览兜底，避免丢失来源条目
            if not source_preview or source_preview in [
                "unknown", "N/A", "无内容", "未知来源", "",
            ]:
                content_fallback = (doc.get("content", "") or "").strip()
                if content_fallback:
                    clean_cf = ' '.join(content_fallback.split())
                    source_preview = clean_cf[:80] + ("..." if len(clean_cf) > 80 else "")
                else:
                    source_preview = "未知来源"

            meta = doc.get("metadata") or {}
            page_idx = meta.get("page_idx")
            chunk_index = meta.get("chunk_index")
            chunk_count = meta.get("chunk_count")
            # Fallback 1: 从来源名解析
            if not isinstance(page_idx, int) or not isinstance(chunk_index, int) or not isinstance(chunk_count, int):
                import re
                m = re.search(r"OCR_page_(\d+)_\w+(?:_ch(\d+)of(\d+))?", source_preview)
                if m:
                    if page_idx is None:
                        try:
                            page_idx = int(m.group(1))
                        except Exception:
                            pass
                    if (chunk_index is None or chunk_count is None) and m.group(2) and m.group(3):
                        try:
                            chunk_index = int(m.group(2))
                            chunk_count = int(m.group(3))
                        except Exception:
                            pass
            
            # Fallback 2: 基于文本内容重建源信息
            if not isinstance(page_idx, int):
                content = doc.get("content", "")
                reconstructed_info = self._reconstruct_source_info(content)
                if reconstructed_info:
                    page_idx = reconstructed_info.get('page_idx')
                    # 如果是部分匹配，可能是分块的结果
                    if reconstructed_info.get('partial_match') and not isinstance(chunk_index, int):
                        chunk_index = 1  # 假设是第一个分块
                        chunk_count = 1   # 无法确定总分块数

            meta_parts: List[str] = []
            if isinstance(page_idx, int):
                meta_parts.append(f"页码: {page_idx}")
            if isinstance(chunk_index, int) and isinstance(chunk_count, int):
                meta_parts.append(f"分块: {chunk_index}/{chunk_count}")
            meta_suffix = f" | {' | '.join(meta_parts)}" if meta_parts else ""

            # 附加原文本内容（适度截断，避免过长输出）
            content_text = (doc.get("content", "") or "").strip()
            if content_text:
                clean = ' '.join(content_text.split())
                preview = clean[:160] + ("..." if len(clean) > 160 else "")
                content_suffix = f" | 内容: {preview}"
            else:
                content_suffix = ""

            sources.append(f"[文档{i}] 来源: {source_preview}{meta_suffix}{content_suffix}")
        return sources
    