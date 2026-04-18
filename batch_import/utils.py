import re
import os
import json
from typing import Set, List


def normalize_arxiv_id(arxiv_id: str) -> str:
    """标准化 arXiv ID"""
    if not arxiv_id:
        return ""
    arxiv_id = re.sub(r'^arXiv:', '', arxiv_id, flags=re.IGNORECASE)
    arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
    arxiv_id = arxiv_id.replace('/', '')
    return arxiv_id.strip()


def safe_filename(arxiv_id: str) -> str:
    """生成安全的文件名"""
    safe = re.sub(r'[\\/*?:"<>|]', "", arxiv_id)
    return safe


def load_json_file(filepath: str, default=None):
    if default is None:
        default = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return default
    return default


def save_json_file(filepath: str, data: dict):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_directories():
    from .config import PAPER_PDF_DIR, PAPER_TXT_DIR
    os.makedirs(PAPER_PDF_DIR, exist_ok=True)
    os.makedirs(PAPER_TXT_DIR, exist_ok=True)