import pymysql
from typing import Optional, Dict, List, Tuple
from .config import MYSQL_CONFIG
from dbutils.pooled_db import PooledDB

# 创建连接池（全局单例）
_pool = None

def get_pool():
    global _pool
    if _pool is None:
        _pool = PooledDB(
            creator=pymysql,
            maxconnections=20,
            mincached=2,
            maxcached=10,
            blocking=True,
            **MYSQL_CONFIG
        )
    return _pool

def get_connection():
    """从连接池获取连接"""
    return get_pool().connection()


def init_db():
    """初始化数据库（自动创建数据库和表）"""
    # 1. 先连接 MySQL（不指定数据库），创建 rag_project 数据库
    temp_config = {
        'host': MYSQL_CONFIG['host'],
        'port': MYSQL_CONFIG['port'],
        'user': MYSQL_CONFIG['user'],
        'password': MYSQL_CONFIG['password'],
        'charset': 'utf8mb4'
    }
    
    conn_temp = pymysql.connect(**temp_config)
    try:
        with conn_temp.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']} "
                f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            print(f"✅ 数据库 {MYSQL_CONFIG['database']} 已就绪")
        conn_temp.commit()
    finally:
        conn_temp.close()
    
    # 2. 连接到指定数据库，创建表
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # 创建 papers 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    arxiv_id VARCHAR(50) UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    authors TEXT,
                    year INT,
                    journal VARCHAR(255),
                    doi VARCHAR(100),
                    categories VARCHAR(255),
                    citation_count INT DEFAULT 0,
                    pdf_path VARCHAR(500),
                    txt_path VARCHAR(500),
                    is_vectorized BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 检查并添加 is_vectorized 字段（兼容旧表）
            cursor.execute("SHOW COLUMNS FROM papers LIKE 'is_vectorized'")
            if not cursor.fetchone():
                cursor.execute("ALTER TABLE papers ADD COLUMN is_vectorized BOOLEAN DEFAULT FALSE")
            
            # 创建 import_log 表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS import_log (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    arxiv_id VARCHAR(50),
                    category VARCHAR(20),
                    status VARCHAR(20),
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建索引（兼容旧版 MySQL，捕获重复索引错误）
            try:
                cursor.execute("CREATE INDEX idx_arxiv_id ON papers(arxiv_id)")
            except pymysql.err.ProgrammingError as e:
                if e.args[0] != 1061:  # Duplicate key name
                    raise
            try:
                cursor.execute("CREATE INDEX idx_categories ON papers(categories)")
            except pymysql.err.ProgrammingError as e:
                if e.args[0] != 1061:
                    raise
            try:
                cursor.execute("CREATE INDEX idx_is_vectorized ON papers(is_vectorized)")
            except pymysql.err.ProgrammingError as e:
                if e.args[0] != 1061:
                    raise
                    
        conn.commit()
        print("✅ MySQL 表初始化完成")
    finally:
        conn.close()
        
def paper_exists(arxiv_id: str) -> bool:
    """检查论文是否已存在"""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM papers WHERE arxiv_id = %s", (arxiv_id,))
            return cursor.fetchone() is not None
    finally:
        conn.close()


def paper_exists_and_vectorized(arxiv_id: str) -> Tuple[bool, bool]:
    """
    检查论文是否存在及是否已向量化
    返回 (exists, is_vectorized)
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT is_vectorized FROM papers WHERE arxiv_id = %s",
                (arxiv_id,)
            )
            result = cursor.fetchone()
            if result:
                return (True, result[0])
            return (False, False)
    finally:
        conn.close()


def save_paper(paper_data: dict) -> bool:
    """保存单篇论文"""
    sql = """
    INSERT INTO papers 
    (arxiv_id, title, authors, year, journal, doi, categories, citation_count, pdf_path, txt_path)
    VALUES (%(arxiv_id)s, %(title)s, %(authors)s, %(year)s, %(journal)s, %(doi)s, 
            %(categories)s, %(citation_count)s, %(pdf_path)s, %(txt_path)s)
    ON DUPLICATE KEY UPDATE
    title = VALUES(title), authors = VALUES(authors), year = VALUES(year),
    journal = VALUES(journal), doi = VALUES(doi), categories = VALUES(categories),
    citation_count = VALUES(citation_count)
    """
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, paper_data)
        conn.commit()
        return True
    except Exception as e:
        print(f"保存失败: {e}")
        return False
    finally:
        conn.close()


def save_papers_batch(papers_data: List[dict]) -> int:
    """批量保存论文，返回成功数量"""
    if not papers_data:
        return 0
    
    sql = """
    INSERT INTO papers 
    (arxiv_id, title, authors, year, journal, doi, categories, citation_count, pdf_path, txt_path)
    VALUES (%(arxiv_id)s, %(title)s, %(authors)s, %(year)s, %(journal)s, %(doi)s, 
            %(categories)s, %(citation_count)s, %(pdf_path)s, %(txt_path)s)
    ON DUPLICATE KEY UPDATE
    title = VALUES(title), authors = VALUES(authors), year = VALUES(year)
    """
    
    conn = get_connection()
    success_count = 0
    try:
        with conn.cursor() as cursor:
            for paper_data in papers_data:
                try:
                    cursor.execute(sql, paper_data)
                    success_count += 1
                except Exception as e:
                    print(f"保存失败 {paper_data.get('arxiv_id')}: {e}")
            conn.commit()
        return success_count
    finally:
        conn.close()


def mark_as_vectorized(arxiv_id: str):
    """标记论文已完成向量化"""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE papers SET is_vectorized = TRUE WHERE arxiv_id = %s",
                (arxiv_id,)
            )
        conn.commit()
    finally:
        conn.close()


def mark_as_vectorized_batch(arxiv_ids: List[str]):
    """批量标记论文已完成向量化"""
    if not arxiv_ids:
        return
    
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            placeholders = ','.join(['%s'] * len(arxiv_ids))
            cursor.execute(
                f"UPDATE papers SET is_vectorized = TRUE WHERE arxiv_id IN ({placeholders})",
                arxiv_ids
            )
        conn.commit()
    finally:
        conn.close()


def get_non_vectorized_papers(limit: int = 100) -> List[Tuple]:
    """获取未向量化的论文（用于断点续传）"""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT arxiv_id, title, categories, pdf_path, txt_path FROM papers "
                "WHERE is_vectorized = FALSE AND txt_path IS NOT NULL LIMIT %s",
                (limit,)
            )
            return cursor.fetchall()
    finally:
        conn.close()


def get_downloaded_count(category: str = None) -> int:
    """获取已下载论文数量"""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            if category:
                cursor.execute("SELECT COUNT(*) FROM papers WHERE categories LIKE %s", (f"%{category}%",))
            else:
                cursor.execute("SELECT COUNT(*) FROM papers")
            return cursor.fetchone()[0]
    finally:
        conn.close()


def get_import_summary() -> dict:
    """获取导入统计摘要"""
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # 总数
            cursor.execute("SELECT COUNT(*) FROM papers")
            total = cursor.fetchone()[0]
            
            # 按分类统计
            cursor.execute("""
                SELECT categories, COUNT(*) 
                FROM papers 
                GROUP BY categories
            """)
            by_category = dict(cursor.fetchall())
            
            # 向量化进度
            cursor.execute("SELECT COUNT(*) FROM papers WHERE is_vectorized = TRUE")
            vectorized = cursor.fetchone()[0]
            
            return {
                "total": total,
                "by_category": by_category,
                "vectorized": vectorized,
                "pending_vectorize": total - vectorized
            }
    finally:
        conn.close()


def log_import(arxiv_id: str, category: str, status: str, error_message: str = None):
    """记录导入日志"""
    sql = "INSERT INTO import_log (arxiv_id, category, status, error_message) VALUES (%s, %s, %s, %s)"
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (arxiv_id, category, status, error_message))
        conn.commit()
    finally:
        conn.close()


def get_paper_title(arxiv_id: str) -> Optional[str]:
    """根据 arxiv_id 获取论文标题"""
    arxiv_id = arxiv_id.replace('.txt', '')
    
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT title FROM papers WHERE arxiv_id = %s", (arxiv_id,))
            result = cursor.fetchone()
            return result[0] if result else None
    finally:
        conn.close()


def get_paper_titles_batch(arxiv_ids: List[str]) -> Dict[str, str]:
    """批量获取论文标题"""
    clean_ids = [id_.replace('.txt', '') for id_ in arxiv_ids]
    
    if not clean_ids:
        return {}
    
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            placeholders = ','.join(['%s'] * len(clean_ids))
            cursor.execute(f"SELECT arxiv_id, title FROM papers WHERE arxiv_id IN ({placeholders})", clean_ids)
            results = cursor.fetchall()
            return {row[0]: row[1] for row in results}
    finally:
        conn.close()