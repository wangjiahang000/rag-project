import re
import os
import hashlib
import json

def normalize_arxiv_id(raw: str) -> str:
    raw = re.sub(r'^arXiv:', '', raw, flags=re.IGNORECASE)
    raw = re.sub(r'v\d+$', '', raw)
    return raw.replace('/', '').strip()

def safe_filename(title: str, arxiv_id: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', '', title).strip('. ')
    if len(safe) > 80:
        safe = safe[:80]
    if not safe:
        safe = arxiv_id
    h = hashlib.md5(title.encode()).hexdigest()[:6]
    return f"{arxiv_id}_{safe}_{h}.pdf"

def load_json(path: str, default=None):
    if default is None:
        default = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return default

def save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)