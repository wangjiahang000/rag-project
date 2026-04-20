#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_import.config import BATCH_CONFIG, API_DELAY_SECONDS, PROGRESS_FILE
from batch_import.utils import ensure_directories, load_json_file, save_json_file
from batch_import.mysql_manager import init_db, get_connection
from batch_import.semantic_scholar_client import search_by_category
from batch_import.paper_importer import import_single_paper

# 配置：每积累多少篇论文的 chunks 后批量写入向量库
# 每篇论文约 5-10 个 chunks，所以 10 篇 ≈ 50-100 个 chunks
BATCH_VECTOR_SIZE = 10  # 每 10 篇论文批量写入一次


def load_progress():
    """加载进度，确保包含所有必需的键"""
    default = {"completed_ids": [], "categories": {}, "current_year": {}}
    progress = load_json_file(PROGRESS_FILE, default)
    
    if "completed_ids" not in progress:
        progress["completed_ids"] = []
    if "categories" not in progress:
        progress["categories"] = {}
    if "current_year" not in progress:
        progress["current_year"] = {}
    
    return progress


def save_progress(progress):
    save_json_file(PROGRESS_FILE, progress)


def flush_chunks_to_vector_store(chunks_buffer):
    """
    将缓冲区中的所有 chunks 批量写入向量数据库
    
    参数:
        chunks_buffer: List[Document] - 待写入的文档块列表
    """
    if not chunks_buffer:
        return False
    
    from backend.vector_store import get_vector_store, create_vector_store
    
    # 统计信息
    paper_ids = set()
    for chunk in chunks_buffer:
        if hasattr(chunk, 'metadata') and 'arxiv_id' in chunk.metadata:
            paper_ids.add(chunk.metadata['arxiv_id'])
    
    print(f"\n📦 批量写入向量库: {len(chunks_buffer)} 个 chunks（来自 {len(paper_ids)} 篇论文）")
    
    try:
        existing_store = get_vector_store()
        
        if existing_store:
            existing_store.add_documents(chunks_buffer)
            existing_store.persist()
            print(f"  ✅ 追加成功")
        else:
            create_vector_store(chunks_buffer)
            print(f"  ✅ 新建成功")
        
        return True
    except Exception as e:
        print(f"  ❌ 批量写入失败: {e}")
        return False


def import_category(category: str, target: int, batch_size: int):
    """
    按分类导入论文（带批量向量化）
    """
    print(f"\n{'='*60}")
    print(f"开始处理分类: {category}")
    print(f"目标数量: {target}")
    print(f"{'='*60}")
    
    progress = load_progress()
    
    # 从 MySQL 获取已下载数量
    from batch_import.mysql_manager import get_connection
    conn = get_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM papers WHERE categories LIKE %s", (f"%{category}%",))
        downloaded = cursor.fetchone()[0]
    conn.close()
    
    if downloaded >= target:
        print(f"✅ {category} 已完成 ({downloaded}/{target})，跳过")
        return
    
    need = target - downloaded
    print(f"还需要下载: {need} 篇")
    
    # 获取当前年份
    current_year = progress["current_year"].get(category, 2026)
    
    success_count = 0
    chunks_buffer = []      # 待批量写入的 chunks 缓冲区
    
    # 从当前年份向下搜索到 2010
    for year in range(current_year, 2009, -1):
        if success_count >= need:
            break
        
        print(f"\n📅 搜索年份: {year}")
        
        papers = search_by_category(category, limit=batch_size, year=year)
        
        if not papers:
            print(f"  {year} 年没有找到论文，继续...")
            continue
        
        for paper in papers:
            arxiv_id = paper.get("arxiv_id", "")
            
            # 检查是否已完成
            if arxiv_id in progress["completed_ids"]:
                print(f"  ⏭️ 跳过已完成: {arxiv_id}")
                continue
            
            # 导入单篇论文（返回 chunks，不立即向量化）
            success, arxiv_id, chunks, metadata = import_single_paper(paper, category)
            
            if success and chunks:
                success_count += 1
                progress["completed_ids"].append(arxiv_id)
                
                # 将 chunks 加入缓冲区
                chunks_buffer.extend(chunks)
                print(f"  📊 进度: {downloaded + success_count}/{target} (缓冲区: {len(chunks_buffer)} chunks)")
                
                # 达到批量阈值，触发写入
                if len(chunks_buffer) >= BATCH_VECTOR_SIZE * 5:  # 每篇约5个chunks
                    flush_chunks_to_vector_store(chunks_buffer)
                    chunks_buffer = []
            
            # 更新进度（年份）
            progress["current_year"][category] = year
            save_progress(progress)
            
            if success_count >= need:
                break
        
        # API 限流延迟
        time.sleep(API_DELAY_SECONDS)
    
    # 最后剩余的 chunks（不足一批的）
    if chunks_buffer:
        flush_chunks_to_vector_store(chunks_buffer)
    
    # 最终统计
    conn = get_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM papers WHERE categories LIKE %s", (f"%{category}%",))
        final_total = cursor.fetchone()[0]
    conn.close()
    
    print(f"\n✅ {category} 完成: 新增 {success_count} 篇，总计 {final_total}/{target}")


def main():
    print("=" * 60)
    print("批量导入 arXiv 论文（优化版 - 批量向量化）")
    print("=" * 60)
    
    ensure_directories()
    print("✅ 目录检查完成")
    
    init_db()
    
    # 显示当前已入库总数
    from batch_import.mysql_manager import get_connection
    conn = get_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM papers")
        total = cursor.fetchone()[0]
    conn.close()
    print(f"\n📊 当前已入库论文: {total} 篇")
    print(f"📦 批量向量化阈值: {BATCH_VECTOR_SIZE} 篇/次")
    
    for category, config in BATCH_CONFIG.items():
        import_category(
            category=category,
            target=config["target"],
            batch_size=config["batch_size"]
        )
    
    # 最终统计
    conn = get_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM papers")
        final_total = cursor.fetchone()[0]
    conn.close()
    
    print("\n" + "=" * 60)
    print("🎉 批量导入完成！")
    print(f"   总入库论文: {final_total} 篇")
    print("=" * 60)


if __name__ == "__main__":
    main()