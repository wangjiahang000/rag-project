#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清除所有已下载的数据，包括：
- papers/pdf/ 下的所有 PDF 文件
- papers/txt/ 下的所有 TXT 文件
- storage/chroma_db/ 向量数据库
- storage/bm25_index.pkl 和 documents_cache.pkl
- storage/import_progress.json 进度文件
- MySQL 数据库 rag_project 中的所有表记录
"""

import os
import shutil
import pymysql
from dotenv import load_dotenv
import glob

load_dotenv()

# ========== 配置路径 ==========
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PAPER_PDF_DIR = os.path.join(PROJECT_ROOT, "papers", "pdf")
PAPER_TXT_DIR = os.path.join(PROJECT_ROOT, "papers", "txt")
STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")
CHROMA_DB_DIR = os.path.join(STORAGE_DIR, "chroma_db")
BM25_INDEX = os.path.join(STORAGE_DIR, "bm25_index.pkl")
DOC_CACHE = os.path.join(STORAGE_DIR, "documents_cache.pkl")
PROGRESS_FILE = os.path.join(STORAGE_DIR, "import_progress.json")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")

# MySQL 配置
MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'port': int(os.getenv('MYSQL_PORT', 3306)),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', ''),
    'database': os.getenv('MYSQL_DATABASE', 'rag_project'),
    'charset': 'utf8mb4'
}


def confirm_action():
    """二次确认"""
    print("⚠️  警告：此操作将永久删除以下所有数据：")
    print(f"   - {PAPER_PDF_DIR} 中的所有 PDF 文件")
    print(f"   - {PAPER_TXT_DIR} 中的所有 TXT 文件")
    print(f"   - {CHROMA_DB_DIR} 向量数据库")
    print(f"   - BM25 索引和缓存文件")
    print(f"   - 导入进度文件")
    print(f"   - MySQL 数据库 {MYSQL_CONFIG['database']} 中的 papers 和 import_log 表记录")
    print(f"   - {UPLOADS_DIR} 中的上传文件（如果存在）")
    print("\n此操作不可逆！")
    response = input("\n确认继续？请输入 'YES' 继续，其他任意键取消: ")
    return response == "YES"


def clear_directory(dir_path: str, description: str):
    """清空指定目录中的所有文件，保留目录本身"""
    if not os.path.exists(dir_path):
        print(f"⏭️  {description} 目录不存在，跳过: {dir_path}")
        return

    file_count = 0
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
                file_count += 1
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                file_count += 1
        except Exception as e:
            print(f"  ❌ 删除失败 {item_path}: {e}")

    print(f"✅ 已清空 {description}: {file_count} 个项目")


def clear_mysql_tables():
    """清空 MySQL 中的 papers 和 import_log 表"""
    try:
        conn = pymysql.connect(**MYSQL_CONFIG)
        with conn.cursor() as cursor:
            # 清空 import_log 表
            cursor.execute("TRUNCATE TABLE import_log")
            print("✅ 已清空 import_log 表")

            # 清空 papers 表
            cursor.execute("TRUNCATE TABLE papers")
            print("✅ 已清空 papers 表")

            conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ MySQL 清理失败: {e}")


def remove_file(file_path: str, description: str):
    """删除单个文件"""
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"✅ 已删除 {description}: {file_path}")
        except Exception as e:
            print(f"❌ 删除失败 {file_path}: {e}")
    else:
        print(f"⏭️  {description} 不存在，跳过")


def main():
    if not confirm_action():
        print("❌ 操作已取消")
        return

    print("\n开始清理...\n")

    # 1. 清空 PDF 目录
    clear_directory(PAPER_PDF_DIR, "PDF 目录")

    # 2. 清空 TXT 目录
    clear_directory(PAPER_TXT_DIR, "TXT 目录")

    # 3. 清空上传目录
    if os.path.exists(UPLOADS_DIR):
        clear_directory(UPLOADS_DIR, "上传目录")

    # 4. 删除向量数据库
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)
        print(f"✅ 已删除向量数据库: {CHROMA_DB_DIR}")
    else:
        print("⏭️  向量数据库不存在，跳过")

    # 5. 删除 BM25 索引和缓存
    remove_file(BM25_INDEX, "BM25 索引")
    remove_file(DOC_CACHE, "文档缓存")

    # 6. 删除进度文件
    remove_file(PROGRESS_FILE, "导入进度文件")

    # 7. 清空 MySQL 表
    clear_mysql_tables()

    print("\n🎉 所有数据已清除完毕！")
    print("提示：如需重新导入，请运行: python -m batch_import.batch_import")


if __name__ == "__main__":
    main()