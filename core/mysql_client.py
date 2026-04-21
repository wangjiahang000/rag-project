import pymysql
from dbutils.pooled_db import PooledDB
from typing import List, Dict

class MySQLClient:
    def __init__(self, **config):
        self.config = config
        self._pool = None
    
    def _get_pool(self):
        if self._pool is None:
            self._pool = PooledDB(
                creator=pymysql,
                maxconnections=10,
                mincached=2,
                maxcached=5,
                blocking=True,
                **self.config
            )
        return self._pool
    
    def get_conn(self):
        return self._get_pool().connection()
    
    def init_db(self):
        temp = {k: v for k, v in self.config.items() if k != 'database'}
        conn = pymysql.connect(**temp)
        with conn.cursor() as c:
            c.execute(f"CREATE DATABASE IF NOT EXISTS {self.config['database']} CHARACTER SET utf8mb4")
        conn.close()
        
        conn = self.get_conn()
        with conn.cursor() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    arxiv_id VARCHAR(50) UNIQUE NOT NULL,
                    title TEXT,
                    authors TEXT,
                    year INT,
                    category VARCHAR(100),
                    pdf_path VARCHAR(500),
                    txt_path VARCHAR(500),
                    is_vectorized BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS import_log (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    arxiv_id VARCHAR(50),
                    category VARCHAR(50),
                    status VARCHAR(20),
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()
        conn.close()
    
    def save_paper(self, data: dict):
        conn = self.get_conn()
        with conn.cursor() as c:
            c.execute("""
                INSERT INTO papers (arxiv_id, title, authors, year, category, pdf_path, txt_path)
                VALUES (%(arxiv_id)s, %(title)s, %(authors)s, %(year)s, %(category)s, %(pdf_path)s, %(txt_path)s)
                ON DUPLICATE KEY UPDATE title=VALUES(title), authors=VALUES(authors), year=VALUES(year)
            """, data)
        conn.commit()
        conn.close()
    
    def mark_vectorized(self, arxiv_ids: List[str]):
        if not arxiv_ids:
            return
        conn = self.get_conn()
        with conn.cursor() as c:
            placeholders = ','.join(['%s'] * len(arxiv_ids))
            c.execute(f"UPDATE papers SET is_vectorized=TRUE WHERE arxiv_id IN ({placeholders})", arxiv_ids)
        conn.commit()
        conn.close()
    
    def get_titles(self, arxiv_ids: List[str]) -> Dict[str, str]:
        if not arxiv_ids:
            return {}
        clean_ids = [id.replace('.txt', '') for id in arxiv_ids]
        conn = self.get_conn()
        with conn.cursor() as c:
            placeholders = ','.join(['%s'] * len(clean_ids))
            c.execute(f"SELECT arxiv_id, title FROM papers WHERE arxiv_id IN ({placeholders})", clean_ids)
            return {row[0]: row[1] for row in c.fetchall()}
        conn.close()