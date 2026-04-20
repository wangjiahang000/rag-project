import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAPER_PDF_DIR = os.path.join(PROJECT_ROOT, "papers", "pdf")
PAPER_TXT_DIR = os.path.join(PROJECT_ROOT, "papers", "txt")
PROGRESS_FILE = os.path.join(PROJECT_ROOT, "storage", "import_progress.json")

MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'port': int(os.getenv('MYSQL_PORT', 3306)),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', ''),
    'database': os.getenv('MYSQL_DATABASE', 'rag_project'),
    'charset': 'utf8mb4'
}

# 计算机科学分类配置
BATCH_CONFIG = {
    "cs.CL": {"target": 400, "batch_size": 50},
    "cs.CV": {"target": 300, "batch_size": 50},
    "cs.LG": {"target": 100, "batch_size": 50},
    "cs.AI": {"target": 100, "batch_size": 50},
    "cs.RO": {"target": 100, "batch_size": 50},
}

API_DELAY_SECONDS = 3.0
MAX_RETRIES = 3

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100