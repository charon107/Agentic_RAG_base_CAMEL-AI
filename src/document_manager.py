"""
文档管理器模块：提供与底层数据源的解耦访问能力。

职责：
- 为关键词检索（BM25）准备文档集合（从 JSON 加载，或其他来源）。
- 当向量检索缺失 payload/metadata 时，基于原始 OCR 数据按文本内容尝试重建页码等信息。
"""

from typing import Any, Dict, List, Optional
import os
import json


class DocumentManager:
    """文档管理器，负责文档的加载和处理"""

    def __init__(self, config: Any):
        # 仅按需读取配置字段，避免与配置类型强耦合
        self.config = config

    def load_documents_from_json(self, json_file_path: Optional[str] = None) -> List[Dict]:
        """
        从OCR JSON加载文档为BM25所需格式，含空文本过滤与去重

        参数:
            json_file_path: JSON文件路径，如果为None则使用配置中的默认路径

        返回:
            文档列表，格式为 [{'content': '内容', 'source': '来源'}]
        """
        if json_file_path is None:
            json_file_path = os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..', getattr(self.config, 'ocr_data_path', 'data/small_ocr_content_list.json'))
            )

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            documents: List[Dict] = []
            seen_hashes = set()

            for item in data:
                text = (item.get('text') or '').strip()
                if not text:
                    continue
                h = hash(text)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                page_idx = item.get('page_idx', 0)
                text_type = item.get('type', 'text')
                source = f"OCR_page_{page_idx}_{text_type}"
                documents.append({'content': text, 'source': source})

            return documents
        except Exception:
            return []

    # 注意：从向量库全量导出文档的功能未在当前项目中使用，且底层无统一遍历API，故移除。

    def reconstruct_source_info(self, content: str) -> Dict:
        """
        基于文本内容重建页码和分块信息（兜底）。

        参数:
            content: 文档内容

        返回:
            重建的源信息字典
        """
        try:
            # 读取原始OCR数据
            ocr_file = os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..', getattr(self.config, 'ocr_data_path', 'data/small_ocr_content_list.json'))
            )
            if not os.path.exists(ocr_file):
                return {}

            with open(ocr_file, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)

            # 查找匹配的内容
            content_clean = content.strip()
            for item in ocr_data:
                item_text = item.get('text', '').strip()
                if item_text and item_text in content_clean:
                    return {
                        'page_idx': item.get('page_idx', 0),
                        'type': item.get('type', 'text'),
                        'original_length': len(item_text)
                    }

            # 如果没有精确匹配，尝试部分匹配（前100字符）
            content_prefix = content_clean[:100]
            for item in ocr_data:
                item_text = item.get('text', '').strip()
                if item_text and item_text[:100] == content_prefix:
                    return {
                        'page_idx': item.get('page_idx', 0),
                        'type': item.get('type', 'text'),
                        'original_length': len(item_text),
                        'partial_match': True
                    }

        except Exception:
            pass

        return {}

