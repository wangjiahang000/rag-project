import sys
import os
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from core.mysql_client import MySQLClient

def confirm():
    print("⚠️ 警告：将删除所有数据！")
    return input("输入 YES 确认: ") == "YES"

def main():
    if not confirm():
        return
    
    # 清理文件
    for d in [settings.papers_pdf_dir, settings.papers_txt_dir, settings.chroma_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
    for f in [settings.progress_file]:
        if os.path.exists(f):
            os.remove(f)
    
    # 清空MySQL表
    client = MySQLClient(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database
    )
    conn = client.get_conn()
    with conn.cursor() as c:
        c.execute("TRUNCATE TABLE papers")
        c.execute("TRUNCATE TABLE import_log")
    conn.commit()
    conn.close()
    print("✅ 清理完成")

if __name__ == "__main__":
    main()