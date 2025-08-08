"""
智能RAG对话系统 - 统一入口
基于 CAMEL-AI 框架实现

本文件作为系统的统一入口点，提供：
1. 数据库构建功能 (database_builder.py)
2. RAG对话功能 (rag_runner.py)
"""

import os
import sys
import subprocess
from dotenv import load_dotenv
from src.db_utils import check_database_exists

# 加载环境变量
load_dotenv()

def show_menu():
    """显示主菜单"""
    print("\n" + "="*60)
    print("🐫 智能RAG对话系统 - 基于 CAMEL-AI 框架")
    print("="*60)
    
    # 检查数据库状态
    db_exists = check_database_exists()
    if db_exists:
        print("✅ 向量数据库状态: 已存在")
    else:
        print("❌ 向量数据库状态: 未创建")
    
    print("\n📋 请选择功能:")
    print("1. 🗄️  构建/管理向量数据库")
    print("2. 💬 启动RAG对话系统")
    print("3. 🔄 一键运行完整流程")
    print("4. ❓ 帮助信息")
    print("5. 🚪 退出")
    
    if not db_exists:
        print("\n⚠️  提示: 首次使用请先选择选项1创建向量数据库")


def run_database_builder():
    """运行数据库构建器"""
    try:
        print("\n🏗️  启动数据库构建器...")
        result = subprocess.run([sys.executable, "database_builder.py"], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 启动数据库构建器失败: {e}")
        return False


def run_rag_system():
    """运行RAG对话系统"""
    try:
        print("\n💬 启动RAG对话系统...")
        result = subprocess.run([sys.executable, "rag_runner.py"], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 启动RAG对话系统失败: {e}")
        return False


def run_complete_flow():
    """运行完整流程"""
    print("\n🚀 开始完整流程...")
    
    # 检查数据库是否存在
    if not check_database_exists():
        print("📊 第一步: 构建向量数据库")
        success = run_database_builder()
        if not success:
            print("❌ 数据库构建失败，流程中止")
            return False
    else:
        print("✅ 检测到现有数据库，跳过构建步骤")
    
    # 启动RAG系统
    print("\n💬 第二步: 启动RAG对话系统")
    return run_rag_system()


def show_help():
    """显示帮助信息"""
    help_text = """
📖 智能RAG对话系统使用指南：

🏗️  数据库构建 (选项1):
   - 创建向量数据库
   - 加载OCR文档数据
   - 管理数据库状态

💬 RAG对话 (选项2):
   - 启动智能问答系统
   - 基于文档内容回答问题
   - 支持马克思主义政治经济学相关问题

🔄 完整流程 (选项3):
   - 自动检查数据库状态
   - 如需要则先构建数据库
   - 然后启动对话系统

📋 系统要求:
   - Python 3.10+
   - 已安装依赖包 (requirements.txt)
   - 配置OPENAI_API_KEY环境变量

🔧 配置文件:
   - .env: 存放API密钥等环境变量
   - data/: 存放训练数据文件
   - src/: 核心代码模块

💡 提示:
   - 首次使用请先选择选项1创建数据库
   - 确保网络连接正常以下载模型
   - API调用需要有效的OpenAI密钥
    """
    print(help_text)


def main():
    """主函数"""
    try:
        while True:
            show_menu()
            
            choice = input("\n请输入选择 (1-5): ").strip()
            
            if choice == '1':
                # 构建/管理向量数据库
                run_database_builder()
                
            elif choice == '2':
                # 启动RAG对话系统
                if not check_database_exists():
                    print("\n❌ 未检测到向量数据库！")
                    print("💡 请先选择选项1创建向量数据库")
                    continue
                run_rag_system()
                
            elif choice == '3':
                # 一键运行完整流程
                run_complete_flow()
                
            elif choice == '4':
                # 显示帮助信息
                show_help()
                
            elif choice == '5':
                # 退出
                print("👋 感谢使用智能RAG对话系统，再见！")
                break
                
            else:
                print("❌ 无效选择，请输入1-5")
            
            # 添加分隔符
            print("\n" + "-"*60)
    
    except KeyboardInterrupt:
        print("\n\n👋 感谢使用智能RAG对话系统，再见！")
    except Exception as e:
        print(f"❌ 程序运行失败: {e}")


if __name__ == "__main__":
    main()