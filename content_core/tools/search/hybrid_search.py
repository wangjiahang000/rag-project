from content_core.data.vector_store import VectorStore

_vector_store: VectorStore = None

def set_vector_store(vs: VectorStore):
    global _vector_store
    _vector_store = vs

def hybrid_search(query: str, k: int = 5) -> list:
    """混合检索，返回文档文本列表"""
    if not _vector_store:
        raise RuntimeError("VectorStore 未注入")
    results = _vector_store.hybrid_search(query, k=k)
    return [doc for doc, _ in results]