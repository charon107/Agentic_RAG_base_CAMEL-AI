"""
数据加载器 - 将 OCR 数据导入向量数据库
支持方案B：按句/固定长度分块，并写入分块元数据到payload
"""

import json
import os
from typing import List, Dict
from qdrant import QdrantDB
import re


class DataLoader:
    """OCR数据加载器"""
    
    def __init__(self, qdrant_db: QdrantDB):
        """
        初始化数据加载器
        
        参数:
            qdrant_db: QdrantDB数据库实例
        """
        self.qdrant_db = qdrant_db
    
    def load_ocr_data(
        self,
        json_file_path: str,
        enable_dedup: bool = True,
        chunking: str = "sentence",  
        chunk_size: int = 300,
        overlap: int = 50,
    ) -> int:
        """
        加载OCR数据到向量数据库
        
        参数:
            json_file_path: OCR数据JSON文件路径
            enable_dedup: 是否启用去重
            chunking: 分块方式
            chunk_size: 目标分块长度（字符数）
            overlap: 分块重叠（字符数）
            
        返回:
            成功导入的文本数量
        """
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"文件不存在: {json_file_path}")
        
        # 读取JSON数据
        with open(json_file_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
        
        success_count = 0
        duplicate_count = 0
        seen_texts = set() if enable_dedup else None
        
        print(f"开始导入OCR数据，共 {len(ocr_data)} 条记录...")
        print(f"分块配置: chunking={chunking}, chunk_size={chunk_size}, overlap={overlap}, enable_dedup={enable_dedup}")
        
        for i, item in enumerate(ocr_data):
            try:
                # 提取文本内容
                text = item.get('text', '').strip()
                
                # 跳过空文本
                if not text:
                    continue
                
                # 分块：仅支持按句切分；其他情况不分块
                chunks: List[str]
                if chunking == "sentence":
                    chunks = self._split_sentences(text)
                else:
                    chunks = [text]

                # 页面与类型信息
                page_idx = item.get('page_idx', 0)
                text_type = item.get('type', 'text')
                
                # 添加分块调试日志
                print(f"  📄 第{i}条记录: page_idx={page_idx}, type={text_type}, 原文长度={len(text)}, 分块数={len(chunks)}")

                # 遍历每个分块，做去重并写入
                total_chunks = len(chunks)
                for ci, chunk in enumerate(chunks, start=1):
                    chunk_clean = chunk.strip()
                    if not chunk_clean:
                        continue

                    if enable_dedup:
                        h = hash(chunk_clean)
                        if h in seen_texts:
                            duplicate_count += 1
                            continue
                        seen_texts.add(h)

                    # 来源标识：带分块信息
                    source_file = f"OCR_page_{page_idx}_{text_type}"
                    if total_chunks > 1:
                        source_file += f"_ch{ci}of{total_chunks}"

                    # 分块元数据
                    payload_extra = {
                        "page_idx": page_idx,
                        "type": text_type,
                        "chunk_index": ci,
                        "chunk_count": total_chunks,
                        "chunking": chunking,
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                    }

                    # 保存到数据库
                    print(f"    💾 写入分块{ci}/{total_chunks}: source_file='{source_file}', 长度={len(chunk_clean)}")
                    print(f"       元数据: {payload_extra}")
                    self.qdrant_db.save_text(chunk_clean, source_file, payload_extra=payload_extra)
                    success_count += 1
                    print(f"    ✅ 成功写入分块{ci}")
                
                # 显示进度
                if (i + 1) % 10 == 0:
                    print(f"已处理 {i + 1}/{len(ocr_data)} 条记录...")
                    
            except Exception as e:
                print(f"处理第 {i} 条记录时出错: {e}")
                continue
        
        print(f"数据导入完成！成功导入 {success_count} 条文本记录")
        if enable_dedup and duplicate_count > 0:
            print(f"🔍 去重统计：跳过 {duplicate_count} 条重复记录")
        return success_count
    
    def batch_save_texts(self, texts: List[str], source_prefix: str = "batch") -> int:
        """
        批量保存文本列表
        
        参数:
            texts: 文本列表
            source_prefix: 源文件前缀
            
        返回:
            成功保存的数量
        """
        success_count = 0
        
        for i, text in enumerate(texts):
            try:
                if text and text.strip():
                    source_file = f"{source_prefix}_{i}"
                    self.qdrant_db.save_text(text.strip(), source_file)
                    success_count += 1
            except Exception as e:
                print(f"保存第 {i} 条文本时出错: {e}")
                continue
        
        return success_count

    def _split_sentences(self, text: str) -> List[str]:
        """按中文与常见标点断句，保留标点。"""
        if not text:
            return []
        parts = re.split(r'([。！？!?；;：:])', text)
        sentences: List[str] = []
        for i in range(0, len(parts), 2):
            seg = parts[i].strip()
            if not seg:
                continue
            punct = parts[i + 1] if i + 1 < len(parts) else ''
            sentences.append((seg + punct).strip())
        return sentences

