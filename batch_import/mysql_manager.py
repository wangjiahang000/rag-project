import pymysql
from typing import Optional, Dict
from .config import MYSQL_CONFIG


def get_connection():
    return pymysql.connect(**MYSQL_CONFIG)


def init_db():
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
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
        conn.commit()
        print("✅ MySQL 表初始化完成")
    finally:
        conn.close()


def paper_exists(arxiv_id: str) -> bool:
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM papers WHERE arxiv_id = %s", (arxiv_id,))
            return cursor.fetchone() is not None
    finally:
        conn.close()


def save_paper(paper_data: dict) -> bool:
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


def get_downloaded_count(category: str = None) -> int:
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


def log_import(arxiv_id: str, category: str, status: str, error_message: str = None):
    sql = "INSERT INTO import_log (arxiv_id, category, status, error_message) VALUES (%s, %s, %s, %s)"
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (arxiv_id, category, status, error_message))
        conn.commit()
    finally:
        conn.close()

# 在 db.py 中添加
def get_paper_title(arxiv_id: str) -> Optional[str]:
    """根据 arxiv_id 获取论文标题"""
    # 去掉可能的 .txt 后缀
    arxiv_id = arxiv_id.replace('.txt', '')
    
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT title FROM papers WHERE arxiv_id = %s", (arxiv_id,))
            result = cursor.fetchone()
            return result[0] if result else None
    finally:
        conn.close()


def get_paper_titles_batch(arxiv_ids: list) -> Dict[str, str]:
    """批量获取论文标题（性能更好）"""
    # 去掉 .txt 后缀
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