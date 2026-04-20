import re
import os
import json
import hashlib
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
    """生成安全的文件名（仅使用 arXiv ID）"""
    safe = re.sub(r'[\\/*?:"<>|]', "", arxiv_id)
    return safe


def safe_pdf_filename(title: str, arxiv_id: str) -> str:
    """
    生成安全的 PDF 文件名，包含标题和哈希防冲突

    格式: {arxiv_id}_{safe_title}_{hash}.pdf

    规则:
        - 移除 Windows/Linux 非法字符: < > : " / \\ | ? *
        - 去除首尾空格和点
        - 限制长度 80 字符
        - 添加 6 位 MD5 哈希避免冲突
    """
    # 清理非法字符
    safe_title = re.sub(r'[<>:"/\\|?*]', '', title)
    # 去除首尾空格和点
    safe_title = safe_title.strip('. ')
    # 限制长度
    if len(safe_title) > 80:
        safe_title = safe_title[:80]
    # 如果清理后为空，使用 arxiv_id 作为标题部分
    if not safe_title:
        safe_title = arxiv_id

    # 计算短哈希
    title_hash = hashlib.md5(title.encode('utf-8')).hexdigest()[:6]

    return f"{arxiv_id}_{safe_title}_{title_hash}.pdf"


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