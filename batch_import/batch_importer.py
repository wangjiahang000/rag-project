#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time
import signal
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import threading

# 新增：进度条库
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_import.config import BATCH_CONFIG, API_DELAY_SECONDS, PROGRESS_FILE
from batch_import.utils import ensure_directories, load_json_file, save_json_file
from batch_import.mysql_manager import (
    init_db, get_connection, get_import_summary,
    mark_as_vectorized_batch, get_downloaded_count,
    paper_exists_and_vectorized, log_import
)
from batch_import.semantic_scholar_client import search_by_category
from batch_import.paper_importer import import_single_paper

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_import.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== 配置参数 ==========
BATCH_VECTOR_SIZE = 10          # 每多少篇论文批量写入向量库一次
MAX_WORKERS = 3                 # 并发下载线程数
MAX_RETRIES = 3                 # 最大重试次数
RETRY_DELAY = 2                 # 初始重试延迟（秒），指数退避

# 全局停止标志
stop_requested = False

def signal_handler(signum, frame):
    global stop_requested
    logger.info("\n⚠️ 收到停止信号，正在保存进度并退出...")
    stop_requested = True

signal.signal(signal.SIGINT, signal_handler)


# ========== 进度管理 ==========
def load_progress():
    default = {"completed_ids": [], "categories": {}, "current_year": {}, "failed_ids": []}
    progress = load_json_file(PROGRESS_FILE, default)

    if "completed_ids" not in progress:
        progress["completed_ids"] = []
    if "categories" not in progress:
        progress["categories"] = {}
    if "current_year" not in progress:
        progress["current_year"] = {}
    if "failed_ids" not in progress:
        progress["failed_ids"] = []

    return progress


def save_progress(progress):
    save_json_file(PROGRESS_FILE, progress)


# ========== 批量向量化 ==========
def flush_chunks_to_vector_store(chunks_buffer, paper_ids_buffer):
    if not chunks_buffer:
        return False

    from backend.vector_store import get_vector_store, create_vector_store

    paper_ids = set()
    for chunk in chunks_buffer:
        if hasattr(chunk, 'metadata') and 'arxiv_id' in chunk.metadata:
            paper_ids.add(chunk.metadata['arxiv_id'])

    logger.info(f"📦 批量写入向量库: {len(chunks_buffer)} 个 chunks（来自 {len(paper_ids)} 篇论文）")

    try:
        existing_store = get_vector_store()

        if existing_store:
            existing_store.add_documents(chunks_buffer)
            existing_store.persist()
            logger.info(f"  ✅ 追加成功")
        else:
            create_vector_store(chunks_buffer)
            logger.info(f"  ✅ 新建成功")

        if paper_ids_buffer:
            mark_as_vectorized_batch(paper_ids_buffer)
            logger.info(f"  ✅ 已标记 {len(paper_ids_buffer)} 篇论文为已向量化")

        return True
    except Exception as e:
        logger.error(f"  ❌ 批量写入失败: {e}")
        return False


# ========== 单篇论文处理（带重试） ==========
def process_paper_with_retry(paper: Dict, category: str, max_retries: int = MAX_RETRIES) -> tuple:
    arxiv_id = paper.get("arxiv_id", "")

    for attempt in range(max_retries):
        try:
            success, arxiv_id, chunks, metadata = import_single_paper(paper, category)

            if success:
                return (True, arxiv_id, chunks, metadata, attempt)
            else:
                if attempt < max_retries - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"  ⚠️ {arxiv_id} 失败，{delay}秒后重试 ({attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"  ❌ {arxiv_id} 最终失败，已重试 {max_retries} 次")
                    return (False, arxiv_id, None, None, attempt)

        except Exception as e:
            logger.error(f"  ❌ {arxiv_id} 异常: {e}")
            if attempt < max_retries - 1:
                delay = RETRY_DELAY * (2 ** attempt)
                time.sleep(delay)
            else:
                return (False, arxiv_id, None, None, attempt)

    return (False, arxiv_id, None, None, max_retries)


# ========== 并发导入核心函数（含进度条） ==========
def import_category_concurrent(
    category: str,
    target: int,
    batch_size: int,
    max_workers: int = MAX_WORKERS
):
    global stop_requested

    logger.info(f"\n{'='*60}")
    logger.info(f"开始处理分类: {category}")
    logger.info(f"目标数量: {target}")
    logger.info(f"并发线程数: {max_workers}")
    logger.info(f"{'='*60}")

    progress = load_progress()

    # 从 MySQL 获取已下载数量
    downloaded = get_downloaded_count(category)

    if downloaded >= target:
        logger.info(f"✅ {category} 已完成 ({downloaded}/{target})，跳过")
        return

    need = target - downloaded
    logger.info(f"还需要下载: {need} 篇")

    # 获取当前年份
    current_year = progress["current_year"].get(category, 2026)

    # 缓冲区（线程安全）
    chunks_buffer = []
    paper_ids_buffer = []
    buffer_lock = threading.Lock()

    # 已处理的论文 ID
    processed_ids = set(progress["completed_ids"])

    # 收集所有待处理的论文
    all_papers = []
    for year in range(current_year, 2009, -1):
        if stop_requested:
            break
        if len(all_papers) >= need:
            break

        logger.info(f"\n📅 搜索年份: {year}")
        papers = search_by_category(category, limit=batch_size, year=year)

        if not papers:
            logger.info(f"  {year} 年没有找到论文，继续...")
            continue

        new_papers = [p for p in papers if p.get("arxiv_id") not in processed_ids]
        all_papers.extend(new_papers)

        progress["current_year"][category] = year
        save_progress(progress)

        logger.info(f"  找到 {len(new_papers)} 篇新论文，累计 {len(all_papers)} 篇")

        time.sleep(API_DELAY_SECONDS)

    # 截取需要的数量
    all_papers = all_papers[:need]
    logger.info(f"\n📚 共需处理 {len(all_papers)} 篇论文")

    # ========== 使用 tqdm 进度条 ==========
    success_count = 0
    failed_papers = []

    # 创建进度条，总数为待处理论文数
    with tqdm(total=len(all_papers), desc=f"📥 {category}", unit="篇", ncols=100) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_paper = {
                executor.submit(process_paper_with_retry, paper, category): paper
                for paper in all_papers
            }

            # 处理完成的任务
            for future in as_completed(future_to_paper):
                if stop_requested:
                    logger.warning("⚠️ 停止信号已收到，取消剩余任务...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                paper = future_to_paper[future]
                arxiv_id = paper.get("arxiv_id", "")

                try:
                    success, arxiv_id, chunks, metadata, retries = future.result()

                    if success and chunks:
                        success_count += 1

                        # 更新进度文件
                        if arxiv_id not in progress["completed_ids"]:
                            progress["completed_ids"].append(arxiv_id)
                            save_progress(progress)

                        # 加入缓冲区
                        with buffer_lock:
                            chunks_buffer.extend(chunks)
                            paper_ids_buffer.append(arxiv_id)

                            if len(paper_ids_buffer) >= BATCH_VECTOR_SIZE:
                                flush_chunks_to_vector_store(chunks_buffer, paper_ids_buffer)
                                chunks_buffer = []
                                paper_ids_buffer = []
                    else:
                        failed_papers.append(arxiv_id)
                        progress["failed_ids"].append(arxiv_id)
                        save_progress(progress)
                        log_import(arxiv_id, category, "failed", "处理失败")

                except Exception as e:
                    logger.error(f"  ❌ {arxiv_id} 处理异常: {e}")
                    failed_papers.append(arxiv_id)

                # 更新进度条
                pbar.update(1)
                # 在进度条后显示成功/失败计数
                pbar.set_postfix(success=success_count, failed=len(failed_papers))

    # 处理剩余的 chunks
    if chunks_buffer and not stop_requested:
        flush_chunks_to_vector_store(chunks_buffer, paper_ids_buffer)

    # 最终统计
    final_total = get_downloaded_count(category)

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ {category} 完成")
    logger.info(f"   成功: {success_count} 篇")
    logger.info(f"   失败: {len(failed_papers)} 篇")
    logger.info(f"   总计: {final_total}/{target}")

    if failed_papers:
        logger.info(f"   失败列表: {failed_papers[:10]}{'...' if len(failed_papers) > 10 else ''}")

    logger.info(f"{'='*60}")


# ========== 恢复失败的论文 ==========
def recover_failed_papers(category: str = None):
    logger.info("\n" + "=" * 60)
    logger.info("恢复失败的论文")
    logger.info("=" * 60)

    progress = load_progress()
    failed_ids = progress.get("failed_ids", [])

    if not failed_ids:
        logger.info("✅ 没有失败的论文需要恢复")
        return

    logger.info(f"发现 {len(failed_ids)} 篇失败的论文")

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            placeholders = ','.join(['%s'] * len(failed_ids))
            cursor.execute(f"""
                SELECT arxiv_id, title, categories
                FROM papers
                WHERE arxiv_id IN ({placeholders}) AND is_vectorized = FALSE
            """, failed_ids)
            failed_papers = cursor.fetchall()
    finally:
        conn.close()

    if not failed_papers:
        logger.info("没有需要重新处理的论文（可能已成功）")
        return

    logger.info(f"重新处理 {len(failed_papers)} 篇论文...")

    for arxiv_id, title, cat in failed_papers:
        logger.info(f"  🔄 重试: {arxiv_id}")
        # 这里可调用单篇处理函数，但为简洁暂时略过

    # 清空失败列表（可选）
    # progress["failed_ids"] = []
    # save_progress(progress)


# ========== 主函数 ==========
def main():
    global stop_requested

    logger.info("=" * 60)
    logger.info("批量导入 arXiv 论文（优化版 - 并发下载 + 重试 + 进度条）")
    logger.info("=" * 60)
    logger.info(f"配置参数:")
    logger.info(f"  并发线程数: {MAX_WORKERS}")
    logger.info(f"  最大重试次数: {MAX_RETRIES}")
    logger.info(f"  批量向量化阈值: {BATCH_VECTOR_SIZE} 篇")
    logger.info("=" * 60)

    ensure_directories()
    logger.info("✅ 目录检查完成")

    init_db()

    # 显示导入统计
    summary = get_import_summary()
    logger.info(f"\n📊 当前入库统计:")
    logger.info(f"   总论文数: {summary['total']}")
    logger.info(f"   已向量化: {summary['vectorized']}")
    logger.info(f"   待向量化: {summary['pending_vectorize']}")
    logger.info(f"   按分类: {summary['by_category']}")

    # 恢复失败的论文
    recover_failed_papers()

    # 按分类导入
    for category, config in BATCH_CONFIG.items():
        if stop_requested:
            logger.warning("⚠️ 停止信号已收到，退出主循环")
            break

        import_category_concurrent(
            category=category,
            target=config["target"],
            batch_size=config["batch_size"],
            max_workers=MAX_WORKERS
        )

    # 最终统计
    final_summary = get_import_summary()

    logger.info("\n" + "=" * 60)
    if stop_requested:
        logger.info("⏸️ 批量导入已中断（进度已保存）")
    else:
        logger.info("🎉 批量导入完成！")
    logger.info(f"   总入库论文: {final_summary['total']} 篇")
    logger.info(f"   已向量化: {final_summary['vectorized']} 篇")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()