# 从Qdrant向量数据库中召回相关文本
# 支持向量检索和关键词检索

from camel.retrievers import VectorRetriever
from .qdrant import QdrantDB
import jieba
import re
from collections import defaultdict
from typing import List, Dict, Optional
import math
try:
    from pinecone import Pinecone  
except Exception:
    Pinecone = None  


try:
    from pystopwords import stopwords as _pys_stopwords 
    _HAS_PYSTOPWORDS = True
except Exception:
    _pys_stopwords = None  
    _HAS_PYSTOPWORDS = False



class VecRetriever:
    """向量检索器"""
    
    def __init__(self, qdrant_db: QdrantDB):
        self.qdrant_db = qdrant_db
        self.vector_retriever = VectorRetriever(
            embedding_model=qdrant_db.embedding_instance,
            storage=qdrant_db.storage_instance
        )
    
    def search(self, question: str, top_k: int = 5):
        """根据问题检索相关文本"""
        raw_results = self.vector_retriever.query(
            query=question,
            top_k=top_k,
            similarity_threshold=0.5
        )
        
        results = []
        seen_contents = set()
        
        for item in raw_results:
            if hasattr(item, 'payload'):
                payload = item.payload
                content = payload.get('text', '')
                source = payload.get('source_file', 'unknown')
                score = getattr(item, 'score', None)
                metadata = dict(payload)
            elif isinstance(item, dict):
                content = item.get('text', '') or item.get('content', '')
                source = item.get('source_file', '') or item.get('file_name', 'unknown')
                score = item.get('score')
                metadata = item.get('metadata', {})
            else:
                continue
            
            if content and hash(content) not in seen_contents:
                result_item = {
                    'file_name': source,
                    'content': content,
                    'metadata': metadata
                }
                if score is not None:
                    result_item['score'] = score
                results.append(result_item)
                seen_contents.add(hash(content))
        
        return results


class KeywordRetriever:
    """基于BM25的关键词检索器"""
    
    def __init__(self, documents: List[Dict] = None):
        self.documents = documents or []
        self.document_terms = []  # 每个文档的分词结果
        self.term_doc_freq = defaultdict(int)  # 词项文档频率
        self.avg_doc_length = 0  # 平均文档长度
        self.doc_lengths = []  # 每个文档的长度
        
        # BM25 参数
        self.k1 = 1.5  # 控制词频饱和度
        self.b = 0.75  # 控制文档长度归一化程度
        
        # 初始化中文停用词
        self.stop_words = self._initialize_stopwords()
        
        if self.documents:
            self._build_index()
    
    def _initialize_stopwords(self) -> set:
        """初始化中文停用词集合。"""
       
        if _HAS_PYSTOPWORDS and callable(_pys_stopwords):
         
            try:
                words = _pys_stopwords(langs="zh")  # type: ignore
                if words:
                    return set(words)
            except TypeError:
             
                try:
                    words = _pys_stopwords('zh')  # type: ignore
                    if words:
                        return set(words)
                except Exception:
                    pass
            except Exception:
                pass

 
        return {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '可以', '但是', '只是', '如果', '因为', '所以', '或者',
            '而且', '虽然', '然而', '不过', '除了', '包括', '关于', '通过', '由于'
        }

    
    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理：分词和清理"""
        # 清理特殊字符，保留中文、英文、数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
        
        # 使用jieba分词
        terms = list(jieba.cut(text.lower()))
        
        # 过滤停用词和短词
        terms = [term.strip() for term in terms 
                if term.strip() and len(term.strip()) > 1 and term.strip() not in self.stop_words]
        
        return terms
    
    def _build_index(self):
        """构建BM25索引"""
        self.document_terms = []
        self.doc_lengths = []
        self.term_doc_freq = defaultdict(int)
        
        for doc in self.documents:
            terms = self._preprocess_text(doc['content'])
            self.document_terms.append(terms)
            self.doc_lengths.append(len(terms))
            
            for term in set(terms):
                self.term_doc_freq[term] += 1
        
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
    
    def _calculate_bm25_score(self, query_terms: List[str], doc_idx: int) -> float:
        """计算BM25得分"""
        doc_terms = self.document_terms[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        score = 0.0
        
        # 统计文档中每个词的频率
        term_freq = defaultdict(int)
        for term in doc_terms:
            term_freq[term] += 1
        
        for term in query_terms:
            if term in term_freq:
                # 计算IDF
                df = self.term_doc_freq[term]  # 包含该词的文档数
                idf = math.log((len(self.documents) - df + 0.5) / (df + 0.5))
                
                # 计算TF部分
                tf = term_freq[term]
                tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length)))
                
                # 累加得分
                score += idf * tf_component
        
        return score
    
    def add_documents(self, documents: List[Dict]):
        self.documents.extend(documents)
        self._build_index()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """BM25检索"""
        if not self.documents:
            return []
        
        # 预处理查询
        query_terms = self._preprocess_text(query)
        if not query_terms:
            return []
        
        # 计算每个文档的得分
        doc_scores = []
        for i in range(len(self.documents)):
            score = self._calculate_bm25_score(query_terms, i)
            if score > 0:  # 只返回有得分的文档
                doc_scores.append((i, score))
        
        # 按得分排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 格式化结果
        results = []
        for doc_idx, score in doc_scores[:top_k]:
            doc = self.documents[doc_idx]
            
            content = doc['content']
            clean_content = ' '.join(content.split())
            # 若有明确来源则优先使用来源；否则退化为内容预览
            source_text = doc.get('source') or ""
            if isinstance(source_text, str) and source_text.strip():
                source_preview = source_text.strip()
            else:
                source_preview = clean_content[:80] + "..." if len(clean_content) > 80 else clean_content
            # 解析来源中的页码
            meta: Dict = {}
            m = re.search(r"OCR_page_(\d+)_", source_preview)
            if m:
                meta['page_idx'] = int(m.group(1))
            
            results.append({
                'file_name': source_preview,
                'content': content,
                'score': score,
                'metadata': meta
            })
        
        return results
    



class RRFReranker:
    """基于RRF算法的重排序器"""
    
    def __init__(self, k: int = 60):
        """
        初始化RRF重排序器
        
        参数:
            k: RRF算法的超参数，用于平滑排名影响
        """
        self.k = k
    
    # 注意：简化实现后仅保留加权版本 rerank_with_weights
    
    def rerank_with_weights(self, result_lists: List[List[Dict]], 
                           weights: List[float], top_k: int = 10) -> List[Dict]:
        """
        使用加权RRF算法对多路检索结果进行重排序
        
        参数:
            result_lists: 多个检索结果列表
            weights: 对应每个检索器的权重
            top_k: 返回前k个重排序结果
            
        返回:
            加权重排序后的结果列表
        """
        if len(result_lists) != len(weights):
            raise ValueError("检索结果列表数量与权重数量不匹配")
        
        doc_scores = {}
        doc_info = {}
        
        # 遍历所有检索结果列表及其权重
        for results, weight in zip(result_lists, weights):
            for rank, result in enumerate(results):
                content = result['content']
                
                # 计算加权RRF得分
                weighted_rrf_score = weight * (1.0 / (rank + 1 + self.k))
                
                # 累加得分
                if content in doc_scores:
                    doc_scores[content] += weighted_rrf_score
                else:
                    doc_scores[content] = weighted_rrf_score
                    doc_info[content] = result
        
        # 按融合分数排序
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 格式化输出
        reranked_results = []
        for content, score in sorted_results[:top_k]:
            result = doc_info[content].copy()
            result['weighted_rrf_score'] = score
            reranked_results.append(result)
        
        return reranked_results


class EnhancedHybridRetriever:
    """增强版混合检索器：集成RRF重排序"""
    
    def __init__(self, vector_retriever: VecRetriever, keyword_retriever: KeywordRetriever,
                 rrf_k: int = 60, vector_weight: float = 1.0, keyword_weight: float = 1.0,
                 external_reranker: Optional[object] = None):
        """
        初始化增强版混合检索器
        
        参数:
            vector_retriever: 向量检索器
            keyword_retriever: 关键词检索器
            rrf_k: RRF算法超参数
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            external_reranker: 额外的外部重排序器
        """
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.reranker = RRFReranker(k=rrf_k)
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.external_reranker = external_reranker
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        使用RRF重排序的混合检索（仅使用 top_k）
        
        参数:
            query: 查询文本
            top_k: 最终返回结果数量
            
        返回:
            RRF重排序后的检索结果
        """
        # 分别进行向量检索和关键词检索
        vector_results = self.vector_retriever.search(query, top_k)
        keyword_results = self.keyword_retriever.search(query, top_k)
        
        # 使用加权RRF重排序
        result_lists = [vector_results, keyword_results]
        weights = [self.vector_weight, self.keyword_weight]
        
        reranked_results = self.reranker.rerank_with_weights(result_lists, weights, top_k)
        # 可选：调用外部重排模型
        if self.external_reranker and reranked_results:
            reranked_results = self.external_reranker.rerank(query, reranked_results, top_n=top_k)
        return reranked_results
    
    def search_with_scores(self, query: str, top_k: int = 10) -> Dict:
        """
        带详细评分信息的混合检索
        
        参数:
            query: 查询文本
            top_k: 最终返回结果数量
            
        返回:
            包含详细评分信息的检索结果
        """
        # 分别进行检索
        vector_results = self.vector_retriever.search(query, top_k)
        keyword_results = self.keyword_retriever.search(query, top_k)

        # RRF重排序 + 可选外部重排
        reranked_results = self.search(query, top_k)

        # 仅当存在分数字段时统计最高分
        def extract_top_score(results: List[Dict]) -> float | None:
            scores = [r.get('score') for r in results if isinstance(r.get('score'), (int, float))]
            return max(scores) if scores else None

        stats: Dict[str, float] = {}
        vector_top = extract_top_score(vector_results)
        keyword_top = extract_top_score(keyword_results)
        if vector_top is not None:
            stats['vector_top_score'] = vector_top
        if keyword_top is not None:
            stats['keyword_top_score'] = keyword_top
        # RRF分数总是可用（融合后）
        stats['rrf_top_score'] = reranked_results[0].get('weighted_rrf_score', 0) if reranked_results else 0

        return {
            'query': query,
            'vector_results_count': len(vector_results),
            'keyword_results_count': len(keyword_results),
            'reranked_results': reranked_results,
            'retrieval_stats': stats,
        }


class PineconeCohereReranker:
    """基于 Pinecone Inference 的 Cohere Rerank 外部重排器。

    要求环境变量或入参提供 Pinecone API Key；
    输入 results 列表项需包含 'content' 字段文本。
    """

    def __init__(self, api_key: str, model: str = "cohere-rerank-3.5"):
        if Pinecone is None:
            raise ImportError("未安装 pinecone，请先安装: pip install pinecone>=5.0.0")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("Pinecone API Key 不能为空")
        self._client = Pinecone(api_key=api_key)
        self._model = model

    def rerank(self, query: str, results: List[Dict], top_n: Optional[int] = None) -> List[Dict]:
        if not results:
            return []
        documents = []
        for i, item in enumerate(results):
            text = (item.get('content') or '').strip()
            documents.append({"id": str(i), "text": text})

        k = min(top_n or len(documents), len(documents))
        resp = self._client.inference.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=k,
            return_documents=True,
        )

        # 根据返回 index 重排，并附加得分与位置
        reordered: List[Dict] = []
        for rank, item in enumerate(getattr(resp, 'data', []) or [], start=1):
            # SDK 既可能返回对象也可能是 dict
            idx = getattr(item, 'index', None)
            score = getattr(item, 'score', None)
            if idx is None and isinstance(item, dict):
                idx = item.get('index')
                score = item.get('score')
            if isinstance(idx, int) and 0 <= idx < len(results):
                r = dict(results[idx])
                if isinstance(score, (int, float)):
                    r['rerank_score'] = float(score)
                r['rerank_position'] = rank
                reordered.append(r)
        return reordered