from core import RAGSystem

_rag_instance = None

def get_rag() -> RAGSystem:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGSystem()
    return _rag_instance