import arxiv
import os
import time
import hashlib
import re
from typing import List, Dict, Optional


class LiteratureSearcher:
    """arXiv文献检索器（无翻译，直接使用英文）"""

    def __init__(self):
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3.0,
            num_retries=3
        )

        # 中文查询词快速映射（常用术语，无需网络）
        self.query_map = {
            "人工智能": "Artificial Intelligence",
            "机器学习": "Machine Learning",
            "深度学习": "Deep Learning",
            "自然语言处理": "Natural Language Processing",
            "计算机视觉": "Computer Vision",
            "大语言模型": "Large Language Model",
            "神经网络": "Neural Network",
            "强化学习": "Reinforcement Learning",
            "注意力机制": "Attention Mechanism",
            "推荐系统": "Recommender System",
            "知识图谱": "Knowledge Graph",
            "数据挖掘": "Data Mining",
            "图像识别": "Image Recognition",
            "语音识别": "Speech Recognition",
            "机器人": "Robotics",
            "自动驾驶": "Autonomous Driving",
            "区块链": "Blockchain",
            "边缘计算": "Edge Computing",
            "云计算": "Cloud Computing",
            "物联网": "Internet of Things"
        }

    def translate_query(self, query: str) -> str:
        """中文查询转英文（使用映射表）"""
        for cn, en in self.query_map.items():
            if cn in query:
                print(f"映射表转换: {cn} -> {en}")
                return query.replace(cn, en)

        print(f"提示: '{query}' 未在映射表中，请使用英文关键词")
        return query

    def search_papers(self, query: str, max_results: int = 10) -> List[Dict]:
        """检索arXiv论文"""
        translated_query = self.translate_query(query)
        print(f"最终搜索关键词: {translated_query}")

        if any('\u4e00' <= char <= '\u9fff' for char in query):
            if translated_query == query:
                print("❌ 请使用英文关键词或以下中文术语:")
                print("   " + ", ".join(list(self.query_map.keys())[:10]))
                return []

        search = arxiv.Search(
            query=translated_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        try:
            for paper in self.client.results(search):
                paper_id = paper.entry_id.split("/abs/")[-1]
                if "/pdf/" in paper_id:
                    paper_id = paper_id.split("/pdf/")[-1]

                authors = [str(a) for a in paper.authors]
                authors_display = ", ".join(authors[:3])
                if len(authors) > 3:
                    authors_display += " et al."

                results.append({
                    "id": paper_id,
                    "title": paper.title,
                    "authors": authors,
                    "authors_display": authors_display,
                    "summary": paper.summary[:500],
                    "published": paper.published.strftime("%Y-%m-%d") if paper.published else "未知",
                    "pdf_url": paper.pdf_url
                })

                print(f"  📄 {paper.title[:60]}...")

        except Exception as e:
            print(f"检索失败: {e}")

        print(f"找到 {len(results)} 篇论文")
        return results

    def download_paper(self, paper_id: str, download_path: str = "./uploads", max_retries: int = 3) -> Optional[str]:
        """
        下载论文PDF，带重试和完整性校验

        参数:
            paper_id: arXiv 论文 ID
            download_path: 保存目录
            max_retries: 最大重试次数

        返回:
            成功时返回文件路径，失败返回 None
        """
        os.makedirs(download_path, exist_ok=True)

        for attempt in range(max_retries):
            try:
                search = arxiv.Search(id_list=[paper_id])
                paper = next(self.client.results(search))

                # 生成安全的文件名
                safe_title = re.sub(r'[<>:"/\\|?*]', '', paper.title)
                safe_title = safe_title.strip('. ')
                if len(safe_title) > 80:
                    safe_title = safe_title[:80]
                title_hash = hashlib.md5(paper.title.encode()).hexdigest()[:6]
                filename = f"{paper_id}_{safe_title}_{title_hash}.pdf"
                filepath = os.path.join(download_path, filename)

                # 如果已存在有效文件，直接返回
                if os.path.exists(filepath) and os.path.getsize(filepath) > 10240:
                    print(f"PDF 已存在: {filename}")
                    return filepath

                # 下载
                paper.download_pdf(dirpath=download_path, filename=filename)

                # 验证完整性
                if os.path.exists(filepath) and os.path.getsize(filepath) > 10240:
                    print(f"下载成功: {filename}")
                    return filepath
                else:
                    if os.path.exists(filepath):
                        os.unlink(filepath)
                    print(f"下载文件无效 (尝试 {attempt+1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    continue

            except StopIteration:
                print(f"论文 {paper_id} 不存在")
                return None
            except Exception as e:
                print(f"下载失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        return None


_searcher = None

def get_searcher() -> LiteratureSearcher:
    global _searcher
    if _searcher is None:
        _searcher = LiteratureSearcher()
    return _searcher