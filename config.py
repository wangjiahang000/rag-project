import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.resolve()

class Settings(BaseSettings):
    project_root: str = str(PROJECT_ROOT)

    # 嵌入模型：BAAI/bge-m3（支持中文、英文、中英混合）
    embedding_model: str = str(PROJECT_ROOT / "models" / "BAAI-bge-m3")
    embedding_device: str = "cpu"

    # 存储路径
    chroma_dir: str = str(PROJECT_ROOT / "storage" / "chroma_db")
    papers_pdf_dir: str = str(PROJECT_ROOT / "storage" / "papers" / "pdf")
    papers_txt_dir: str = str(PROJECT_ROOT / "storage" / "papers" / "txt")
    progress_file: str = str(PROJECT_ROOT / "storage" / "import_progress.json")

    # MySQL
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "rag_project"

    # DeepSeek
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"

    # 分块参数（适配学术论文：段落级分割，兼顾细节与上下文）
    chunk_size: int = 512
    chunk_overlap: int = 128

    # 检索参数
    retrieval_k: int = 5
    bm25_weight: float = 0.3
    vector_weight: float = 0.7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
