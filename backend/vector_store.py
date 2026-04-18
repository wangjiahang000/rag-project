import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/bge-small-zh-v1.5")

embeddings = HuggingFaceEmbeddings(
    model_name=LOCAL_MODEL_PATH,  # 使用本地路径
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

VECTOR_STORE_PATH = "./storage/chroma_db"

def get_vector_store():
    if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        return Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )
    return None

def create_vector_store(documents):
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    vector_store.persist()
    return vector_store

def search_documents_with_score(query, k=6,score_threshold=1.0):
    vector_store = get_vector_store()
    if not vector_store:
        print("⚠️ 向量库为空，请先上传文档")
        return []
    results = vector_store.similarity_search_with_score(query, k=k)
    print(f"\n{'='*50}")
    print(f"🔍 查询: {query}")
    print(f"{'='*50}")
    for doc, score in results:
        source = doc.metadata.get("source", "未知")
        content_preview = doc.page_content[:50].replace("\n", " ")
        print(f"   📄 {source}: 相似度 {score:.4f}")
        print(f"      内容: {content_preview}...")
    
    filtered = [(doc, score) for doc, score in results if score <= score_threshold]
    print(f"\n   ✅ 过滤后(>={score_threshold}): {len(filtered)} 个")
    print(f"{'='*50}\n")
    
    return filtered
def search_documents_mmr(query, k=4, lambda_mult=0.5):
    vector_store = get_vector_store()
    if not vector_store:
        return []
    
    # MMR 检索，平衡相关性和多样性
    results = vector_store.max_marginal_relevance_search(
        query, 
        k=k, 
        lambda_mult=lambda_mult  # 0=完全多样性，1=完全相关性
    )
    return results