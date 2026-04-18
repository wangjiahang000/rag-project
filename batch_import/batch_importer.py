#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_import.config import BATCH_CONFIG, API_DELAY_SECONDS, PROGRESS_FILE
from batch_import.utils import ensure_directories, load_json_file, save_json_file
from batch_import.mysql_manager import init_db
from batch_import.semantic_scholar_client import search_by_category
from batch_import.paper_importer import import_single_paper


def load_progress():
    """加载进度，确保包含所有必需的键"""
    default = {"completed_ids": [], "categories": {}, "current_year": {}}
    progress = load_json_file(PROGRESS_FILE, default)
    
    # 确保必需的键存在
    if "completed_ids" not in progress:
        progress["completed_ids"] = []
    if "categories" not in progress:
        progress["categories"] = {}
    if "current_year" not in progress:
        progress["current_year"] = {}
    
    return progress


def save_progress(progress):
    save_json_file(PROGRESS_FILE, progress)


def import_category(category: str, target: int, batch_size: int):
    print(f"\n{'='*60}")
    print(f"开始处理分类: {category}")
    print(f"目标数量: {target}")
    print(f"{'='*60}")
    
    progress = load_progress()
    
    # 获取已下载数量（从 completed_ids 统计该分类的论文数）
    downloaded = 0
    for arxiv_id in progress["completed_ids"]:
        # 简单统计，实际可以从 MySQL 查询更准确
        downloaded = len(progress["completed_ids"])
        break  # 临时：直接用总长度
    
    # 更准确的方法：从 MySQL 查询
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
            
            if arxiv_id in progress["completed_ids"]:
                print(f"  ⏭️ 跳过已完成: {arxiv_id}")
                continue
            
            success = import_single_paper(paper, category)
            
            if success:
                success_count += 1
                progress["completed_ids"].append(arxiv_id)
                print(f"  📊 进度: {downloaded + success_count}/{target}")
            
            # 更新进度
            progress["current_year"][category] = year
            save_progress(progress)
            
            if success_count >= need:
                break
        
        time.sleep(API_DELAY_SECONDS)
    
    print(f"\n✅ {category} 完成: 新增 {success_count} 篇，总计 {downloaded + success_count}/{target}")


def main():
    print("=" * 60)
    print("批量导入 arXiv 论文")
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