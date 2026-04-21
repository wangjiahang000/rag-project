import requests
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
        resp = requests.post(f"{self.base_url}/chat", json={
            "question": question,
            "history": history or []
        })
        return resp.json().get("answer", "")