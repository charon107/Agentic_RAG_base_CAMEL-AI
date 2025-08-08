# Agentic RAG (CAMEL-AI)

一个基于 CAMEL-AI 框架的 Retrieval-Augmented Generation（RAG）示例工程：
- 本地持久化向量库（QdrantStorage）
- 稠密向量检索 + 关键词检索（BM25）
- RRF 融合与可选的外部重排（Pinecone Inference 的 Cohere Rerank）
- 对话式问答，支持展示来源、页码、文本内容预览

---

## 环境要求
- Python 3.10+
- 操作系统：Windows / Linux / macOS

## 依赖安装
```bash
pip install -r requirements.txt
```

## 环境变量配置
本项目需要 OpenAI 官方 API Key。请在项目根目录创建 `.env` 文件，写入：

```bash
OPENAI_API_KEY=your_openai_api_key_here

PINECONE_API_KEY=
```

说明：
- 本项目已强制要求存在 `OPENAI_API_KEY` 才能运行。
- 不配置 `PINECONE_API_KEY` 时，不启用外部重排。

---

## 数据准备
项目内置示例数据：`data/small_ocr_content_list.json`（OCR 内容的列表）。

向量数据文件将持久化在：`src/qdrant_data/`（已加入 `.gitignore`）。

---

## 一键运行
使用主入口 `main.py`：
```bash
python main.py
```
菜单说明：
- 1 构建/管理向量数据库（调用 `database_builder.py`）
- 2 启动 RAG 对话系统（调用 `rag_runner.py`）
- 3 一键运行完整流程（如无向量库先构建后启动）
- 4 帮助信息
- 5 退出

首次运行建议：先选 1 构建向量数据库，再选 2 启动对话系统。

---

## 分步运行
### 1) 构建/重建向量数据库
```bash
python database_builder.py
```
该工具会：
- 使用嵌入模型 `Qwen/Qwen3-Embedding-4B` 生成向量
- 读取 `data/small_ocr_content_list.json`
- 将文本写入 Qdrant 向量库（`src/qdrant_data/collection/rag_collection`）
- 默认去重，默认“按句切分”（仅断句，不再做分块聚合/定长切分）

参数与行为见 `database_builder.py`：
- 允许选择是否重建数据库
- 默认数据文件：`data/small_ocr_content_list.json`

### 2) 启动 RAG 对话系统
```bash
python rag_runner.py
```
你将进入交互式问答：输入问题并回车。系统会检索相关片段并融合后给出回答，同时显示信息来源（来源/页码/文本预览）。

---

## 项目结构（主要 Python 文件）
- `main.py`：统一入口菜单
- `database_builder.py`：向量库构建工具（读取 JSON → 写入 Qdrant）
- `rag_runner.py`：RAG 对话启动器（初始化并进入交互式问答）
- `process_ocr_content.py`：OCR 相关处理脚本（如需）

- `src/qdrant.py`：QdrantDB 封装
  - 使用 `SentenceTransformerEncoder` 编码文本
  - 使用 `QdrantStorage` 本地持久化向量及 payload
- `src/vector_retriever.py`：检索与融合
  - `VecRetriever`：向量检索（Qdrant，余弦相似度），默认相似度阈值 0.3
  - `KeywordRetriever`：BM25 关键词检索（jieba 分词 + 精简中文停用词/可选 pystopwords）
  - `RRFReranker`：RRF 融合（仅保留加权版本）
  - `EnhancedHybridRetriever`：向量+关键词加权融合，支持可选外部重排
  - `PineconeCohereReranker`：基于 Pinecone Inference 的 Cohere Rerank
- `src/rag_chat_agent.py`：RAG 聊天代理
  - `RAGConfig`：配置（模型名、权重、RRF k、数据路径、是否启用外部重排等）
  - `DocumentManager`（在 `src/document_manager.py`）：加载 BM25 文档与兜底重建页码
  - `RetrievalManager`：统一检索接口，混合检索 + RRF（+ 外部重排）
  - `RAGChatAgent`：对话代理，构建上下文并调用 LLM
- `src/document_manager.py`：文档加载与兜底重建
- `src/data_loader.py`：数据导入到向量库（仅按句断句；不再做聚合/定长分块）
- `src/db_utils.py`：数据库存在性检查
- `src/__init__.py`：包初始化

---

## 工作流程（RAG Pipeline）
1. 数据加载（`database_builder.py` → `src/data_loader.py`）
   - 读取 `small_ocr_content_list.json`
   - 仅按句断句；写入 Qdrant 向量库（payload 含 `text`、`source_file`）

2. 检索（`src/vector_retriever.py`）
   - 向量检索：余弦相似度，`top_k` 候选，阈值 0.3
   - 关键词检索：BM25（jieba + 停用词）
   - 融合：加权 RRF，支持可选外部重排（Pinecone Cohere Rerank）

3. 生成回答（`src/rag_chat_agent.py`）
   - 组装上下文（含来源、页码、分块信息解析、文本预览）
   - 调用 OpenAI 官方模型（`OPENAI_API_KEY`）
   - 输出回答与信息来源

---

## 可配置项（`RAGConfig`）
- `model_name`: 聊天模型名（默认 `gpt-5`）
- `embedding_model`: 嵌入模型名（默认 `Qwen/Qwen3-Embedding-4B`）
- `reranker_model`: 外部重排模型（默认 `cohere-rerank-3.5`）
- `vector_weight` / `keyword_weight`: 各通道权重（默认 0.5/0.5）
- `rrf_k`: RRF 平滑参数（默认 20）
- `ocr_data_path`: OCR JSON 路径（默认 `data/small_ocr_content_list.json`）
- `enable_reranking`: 是否启用外部重排（默认 True；需 `PINECONE_API_KEY`）

---

## 常见问题（FAQ）
- 启动报错“未检测到 OPENAI_API_KEY”：
  - 在项目根目录创建 `.env` 并填写 `OPENAI_API_KEY`。
- 信息来源只有一条：
  - 可能是召回太少或来源为空被过滤；已加兜底逻辑（内容预览作为来源）。
  - 也可调大 `top_k` 或确认数据库构建成功。
- 外部重排无效：
  - 确认已安装 `pinecone>=5.0.0` 且配置 `PINECONE_API_KEY`。
- 中文停用词报错：
  - 已做兼容处理；如 `pystopwords` 不可用，会回退到内置精简集合。

---

## 许可证
MIT（如未另行声明）。