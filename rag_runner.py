"""
RAG对话系统运行器
专门负责运行智能对话系统
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv
from src.db_utils import check_database_exists

# 加载环境变量
load_dotenv()

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.qdrant import QdrantDB
from src.rag_chat_agent import RAGChatAgent, RAGConfig


class RAGRunner:
    """RAG对话系统运行器"""
    
    def __init__(self, 
                 model_name: str = "gpt-5",
                 embedding_model: str = "Qwen/Qwen3-Embedding-4B",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        初始化RAG运行器
        
        参数:
            model_name: 聊天模型名称
            embedding_model: 嵌入模型名称  
            api_key: API密钥
            base_url: API基础URL
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.base_url = base_url
        
        # 初始化组件
        self.qdrant_db = None
        self.chat_agent = None
    
    def check_database_exists(self) -> bool:
        """代理到通用数据库检查函数"""
        return check_database_exists()
    
    def initialize_system(self):
        """初始化RAG系统组件"""
        try:
            print("🤖 RAG系统初始化中...")
            
            # 检查数据库是否存在
            if not self.check_database_exists():
                print("❌ 未检测到向量数据库！")
                print("💡 请先运行 'python database_builder.py' 创建向量数据库")
                return False
            
            # 1. 初始化向量数据库连接
            print("📊 连接向量数据库...")
            self.qdrant_db = QdrantDB(model_name=self.embedding_model)
            
            # 2. 初始化对话代理
            print("💬 初始化对话代理...")
            config = RAGConfig(
                model_name=self.model_name,
                embedding_model=self.embedding_model,
            )
            self.chat_agent = RAGChatAgent(
                config=config,
                api_key=self.api_key,
                base_url=self.base_url,
            )
            
            print("✅ RAG系统初始化完成！")
            return True
            
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            return False
    
    def chat_interactive(self):
        """交互式聊天模式"""
        if not self.chat_agent:
            print("❌ 系统未初始化，无法启动对话")
            return
        
        print("\n" + "="*50)
        print("🎯 欢迎使用智能RAG对话系统！")
        print("📖 基于CAMEL-AI框架，支持基于文档的智能问答")
        print("💡 输入 'quit' 或 'exit' 退出系统")
        print("💡 输入 'help' 查看帮助信息")
        print("="*50 + "\n")
        
        while True:
            try:
                # 获取用户输入
                question = input("🤔 请输入您的问题: ").strip()
                
                if not question:
                    continue
                
                # 处理特殊命令
                if question.lower() in ['quit', 'exit', '退出']:
                    print("👋 感谢使用RAG系统，再见！")
                    break
                
                if question.lower() in ['help', '帮助']:
                    self._show_help()
                    continue
                
                if question.lower() in ['info', '信息']:
                    self._show_system_info()
                    continue
                
                # 处理正常问题
                print("\n🔍 正在检索相关信息...")
                result = self.chat_agent.query(question, top_k=3)
                
                print(f"\n🤖 回答: {result['answer']}")
                
                # 显示信息来源
                sources = result.get('sources', [])
                if sources:
                    print(f"\n📚 信息来源:")
                    for i, source in enumerate(sources, 1):
                        print(f"   {i}. {source}")
                else:
                    print(f"\n📚 信息来源: 无法获取相关文档")
                
                print("\n" + "-"*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 感谢使用RAG系统，再见！")
                break
            except Exception as e:
                print(f"\n❌ 处理问题时出错: {e}")
                print("请重试或输入其他问题。\n")
    
    def single_query(self, question: str, top_k: int = 3) -> dict:
        """
        单次查询接口
        
        参数:
            question: 用户问题
            top_k: 检索文档数量
            
        返回:
            查询结果字典
        """
        if not self.chat_agent:
            return {"error": "系统未初始化"}
        
        return self.chat_agent.query(question, top_k)
    
    def _show_help(self):
        """显示帮助信息"""
        help_text = """
📖 RAG系统帮助信息：

🔸 基本用法：
   - 直接输入问题，系统会基于文档内容回答
   - 支持关于马克思主义政治经济学的问题

🔸 特殊命令：
   - help/帮助: 显示此帮助信息
   - info/信息: 显示系统状态信息
   - quit/exit/退出: 退出系统

🔸 示例问题：
   - "什么是使用价值？"
   - "什么是交换价值？"
   - "劳动如何创造价值？"
   - "商品的二重性是什么？"

💡 提示：问题越具体，回答越准确！
        """
        print(help_text)
    
    def _show_system_info(self):
        """显示系统信息"""
        if not self.chat_agent:
            print("❌ 系统未初始化")
            return
        
        # 简化系统信息输出：移除对 chat_agent.get_database_info 的依赖
        print(f"""
📊 系统状态信息：
   - 聊天模型: {self.model_name}
   - 嵌入模型: {self.embedding_model}
   - OPENAI_API_KEY: 已配置
        """)


def check_api_key():
    """检查并强制要求 OPENAI_API_KEY 存在。"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ 未检测到 OPENAI_API_KEY 环境变量。")
        print("请在 .env 中设置 OPENAI_API_KEY=your_api_key 或在环境中导出该变量后重试。")
        return None, None
    print(f"✅ 检测到API密钥: {api_key[:10]}...")
    # 统一走官方 OpenAI 路径，不再支持无密钥时的兼容端点输入
    return api_key, None


def main():
    """主函数"""
    print("💬 启动RAG对话系统...")
    
    # 检查API密钥
    api_key, base_url = check_api_key()
    if api_key is None:
        return
    
    try:
        # 创建RAG运行器
        rag_runner = RAGRunner(
            model_name="gpt-5",
            embedding_model="Qwen/Qwen3-Embedding-4B",
            api_key=api_key,
            base_url=base_url
        )
        
        # 初始化系统
        if not rag_runner.initialize_system():
            print("\n❌ 系统初始化失败")
            print("💡 建议步骤：")
            print("   1. 运行 'python database_builder.py' 创建向量数据库")
            print("   2. 确保数据文件 'data/small_ocr_content_list.json' 存在")
            print("   3. 检查网络连接和API密钥配置")
            return
        
        # 启动交互式聊天
        rag_runner.chat_interactive()
        
    except Exception as e:
        print(f"❌ 系统运行失败: {e}")
        print("请检查配置和依赖是否正确安装")


if __name__ == "__main__":
    main()