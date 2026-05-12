import jieba
from rank_bm25 import BM25Okapi

def rerank(docs: list[str], query: str) -> list[str]:
    """用 BM25 对文档列表重排序"""
    if not docs:
        return []
    tokenized = [list(jieba.cut(doc)) for doc in docs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(list(jieba.cut(query)))
    sorted_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_pairs]