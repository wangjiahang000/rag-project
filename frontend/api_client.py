import requests
import json as json_lib
from typing import List, Dict, Any

class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def upload_file(self, file_path: str) -> dict:
        with open(file_path, 'rb') as f:
            resp = requests.post(f"{self.base_url}/upload", files={"file": f})
        return resp.json()

    def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict]:
        resp = requests.post(f"{self.base_url}/arxiv/search", json={
            "query": query, "max_results": max_results
        })
        return resp.json().get("papers", [])

    def add_paper(self, paper_id: str, title: str = "", category: str = "manual") -> dict:
        resp = requests.post(f"{self.base_url}/arxiv/add", json={
            "paper_id": paper_id, "paper_title": title, "category": category
        })
        return resp.json()

    def chat(self, question: str, history: List[List[str]] = None) -> str:
        body = {"question": question, "history": history or []}
        print(f"\n[DEBUG] POST /chat 请求体: {json_lib.dumps(body, ensure_ascii=False)[:500]}")
        resp = requests.post(f"{self.base_url}/chat", json=body)
        print(f"[DEBUG] 响应状态码: {resp.status_code}")
        if resp.status_code != 200:
            print(f"[DEBUG] 响应体: {resp.text[:500]}")
        return resp.json().get("answer", "")