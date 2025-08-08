"""
向量数据库构建器
专门负责创建向量数据库和加载数据
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.qdrant import QdrantDB
from src.data_loader import DataLoader
from src.db_utils import check_database_exists


class DatabaseBuilder:
    """向量数据库构建器"""
    
    def __init__(self, embedding_model: str = "Qwen/Qwen3-Embedding-4B"):
        """
        初始化数据库构建器
        
        参数:
            embedding_model: 嵌入模型名称
        """
        self.embedding_model = embedding_model
        self.qdrant_db = None
        self.data_loader = None
        
    def check_database_exists(self) -> bool:
        """代理到通用数据库检查函数"""
        return check_database_exists()
    
    def get_database_info(self) -> dict:
        """
        获取数据库信息
        
        返回:
            dict: 数据库信息
        """
        db_path = os.path.join("src", "qdrant_data")
        collection_path = os.path.join(db_path, "collection", "rag_collection")
        
        info = {
            "exists": self.check_database_exists(),
            "db_path": db_path,
            "collection_path": collection_path,
            "embedding_model": self.embedding_model
        }
        
        if info["exists"]:
            try:
                # 统计文件数量（简单估计数据量）
                files = []
                for root, dirs, filenames in os.walk(collection_path):
                    files.extend(filenames)
                info["file_count"] = len(files)
            except:
                info["file_count"] = "未知"
        
        return info
    
    def initialize_database(self):
        """初始化数据库组件"""
        try:
            print("📊 初始化向量数据库...")
            self.qdrant_db = QdrantDB(model_name=self.embedding_model)
            
            print("📁 初始化数据加载器...")
            self.data_loader = DataLoader(self.qdrant_db)
            
            print("✅ 数据库组件初始化完成！")
            
        except Exception as e:
            print(f"❌ 数据库初始化失败: {e}")
            raise
    
    def load_data(self, data_file: Optional[str] = None, force_reload: bool = False):
        """
        加载数据到向量数据库
        
        参数:
            data_file: 数据文件路径（默认使用 data/small_ocr_content_list.json）
            force_reload: 是否强制重新加载数据
        """
        if data_file is None:
            data_file = os.path.join("data", "small_ocr_content_list.json")
        
        if not os.path.exists(data_file):
            print(f"❌ 数据文件不存在: {data_file}")
            return False
        
        # 检查是否已存在数据库
        if self.check_database_exists() and not force_reload:
            print("⚠️  检测到向量数据库已存在")
            choice = input("是否重新构建数据库？(y/n): ").lower().strip()
            if choice != 'y':
                print("💡 跳过数据加载，使用现有数据库")
                return True
        
        try:
            # 如果组件未初始化，先初始化
            if self.qdrant_db is None:
                self.initialize_database()
            
            print("📥 开始加载OCR数据...")
            
            count = self.data_loader.load_ocr_data(
                data_file,
                enable_dedup=True,
                chunking="sentence",
                chunk_size=300,
                overlap=50,
            )
            print(f"✅ 成功加载 {count} 条数据记录")
            return True
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def build_database(self, data_file: Optional[str] = None):
        """
        完整的数据库构建流程
        
        参数:
            data_file: 数据文件路径
        """
        print("🏗️  开始构建向量数据库...")
        
        # 显示当前状态
        info = self.get_database_info()
        print(f"📍 数据库路径: {info['db_path']}")
        print(f"🤖 嵌入模型: {info['embedding_model']}")
        
        if info["exists"]:
            print("✅ 检测到现有数据库")
            print(f"📁 集合路径: {info['collection_path']}")
            print(f"📊 文件数量: {info.get('file_count', '未知')}")
        else:
            print("🆕 将创建新的数据库")
        
        # 执行构建
        success = self.load_data(data_file)
        
        if success:
            print("\n🎉 向量数据库构建完成！")
            print("💡 现在可以运行 'python rag_runner.py' 开始对话")
        else:
            print("\n❌ 向量数据库构建失败")
        
        return success


def main():
    """主函数"""
    print("🗄️  向量数据库构建工具")
    print("="*50)
    
    try:
        # 创建数据库构建器
        builder = DatabaseBuilder(embedding_model="Qwen/Qwen3-Embedding-4B")
        
        # 显示菜单
        while True:
            print("\n📋 请选择操作:")
            print("1. 构建/重建向量数据库")
            print("2. 检查数据库状态")
            print("3. 退出")
            
            choice = input("\n请输入选择 (1-3): ").strip()
            
            if choice == '1':
                # 构建数据库
                data_file = input("数据文件路径 (回车使用默认): ").strip()
                if not data_file:
                    data_file = None
                builder.build_database(data_file)
                
            elif choice == '2':
                # 检查状态
                info = builder.get_database_info()
                print(f"\n📊 数据库状态:")
                print(f"  存在: {'✅ 是' if info['exists'] else '❌ 否'}")
                print(f"  路径: {info['db_path']}")
                print(f"  嵌入模型: {info['embedding_model']}")
                if info['exists']:
                    print(f"  文件数量: {info.get('file_count', '未知')}")
                
            elif choice == '3':
                print("👋 退出数据库构建工具")
                break
                
            else:
                print("❌ 无效选择，请重试")
    
    except Exception as e:
        print(f"❌ 程序运行失败: {e}")


if __name__ == "__main__":
    main()