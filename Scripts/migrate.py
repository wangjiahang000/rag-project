#!/usr/bin/env python
"""
数据库迁移工具
用法:
    python scripts/migrate.py                 # 执行所有未执行的迁移
    python scripts/migrate.py --create name   # 创建新的迁移文件
    python scripts/migrate.py --status        # 查看迁移状态
"""
import sys
import os
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
import pymysql

MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "migrations")

def get_connection():
    return pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
        charset='utf8mb4'
    )

def init_migration_table(conn):
    with conn.cursor() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                version VARCHAR(50) UNIQUE NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    conn.commit()

def get_applied_versions(conn):
    with conn.cursor() as c:
        c.execute("SELECT version FROM schema_migrations ORDER BY version")
        return [row[0] for row in c.fetchall()]

def get_migration_files():
    if not os.path.exists(MIGRATIONS_DIR):
        return []
    files = []
    pattern = re.compile(r'^V(\d+__\w+)\.sql$')
    for f in os.listdir(MIGRATIONS_DIR):
        match = pattern.match(f)
        if match:
            files.append((f, match.group(1)))
    return sorted(files, key=lambda x: x[1])

def apply_migration(conn, filename, version):
    filepath = os.path.join(MIGRATIONS_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        sql = f.read()
    with conn.cursor() as c:
        for statement in sql.split(';'):
            statement = statement.strip()
            if statement:
                c.execute(statement)
        c.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
    conn.commit()
    print(f"✅ 已应用: {filename}")

def create_migration(name):
    if not os.path.exists(MIGRATIONS_DIR):
        os.makedirs(MIGRATIONS_DIR)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"V{timestamp}__{name}.sql"
    filepath = os.path.join(MIGRATIONS_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("-- 在此处编写 SQL 语句\n")
    print(f"✅ 已创建迁移文件: {filepath}")

def migrate():
    conn = get_connection()
    try:
        init_migration_table(conn)
        applied = get_applied_versions(conn)
        files = get_migration_files()
        pending = [(f, v) for f, v in files if v not in applied]
        if not pending:
            print("✅ 数据库已是最新状态")
            return
        print(f"发现 {len(pending)} 个待应用的迁移:")
        for f, v in pending:
            print(f"  - {f}")
        for f, v in pending:
            apply_migration(conn, f, v)
        print("✅ 迁移完成")
    finally:
        conn.close()

def status():
    conn = get_connection()
    try:
        init_migration_table(conn)
        applied = get_applied_versions(conn)
        files = get_migration_files()
        print("已应用的迁移:")
        for v in applied:
            print(f"  [x] {v}")
        print("待应用的迁移:")
        for f, v in files:
            if v not in applied:
                print(f"  [ ] {v} ({f})")
    finally:
        conn.close()

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "--create" and len(sys.argv) > 2:
            create_migration(sys.argv[2])
        elif sys.argv[1] == "--status":
            status()
        else:
            print("用法: migrate.py [--create name] [--status]")
    else:
        migrate()

if __name__ == "__main__":
    main()