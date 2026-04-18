import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import logging

# 禁用警告
logging.set_verbosity_error()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 使用本地模型路径（你手动下载的位置）
model_path = r"C:\Users\35717\Desktop\my_rag_project\models\paraphrase-multilingual-MiniLM-L12-v2"

# 检查模型是否存在
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ 模型路径不存在: {model_path}\n请确保模型已下载到该目录")

print(f"✅ 加载本地模型: {model_path}")

embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

VECTOR_STORE_PATH = "./storage/chroma_db"

def get_vector_store():
    """获取已存在的向量数据库"""
    if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        return Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )
    return None

def create_vector_store(documents):
    """创建新的向量数据库（会覆盖旧的）"""
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    return vector_store

def search_documents_with_score(query, k=10, score_threshold=1.0):
    """
    检索文档并返回带相似度分数的结果
    注意：这个模型返回的是欧氏距离（L2距离），分数越小越相似！
    """
    vector_store = get_vector_store()
    if not vector_store:
        return []
    
    results = vector_store.similarity_search_with_score(query, k=k)
    
    # 打印调试信息
    print(f"\n{'='*60}")
    print(f"🔍 查询: {query}")
    print(f"{'='*60}")
    for doc, score in results:
        source = doc.metadata.get("source", "未知")
        preview = doc.page_content[:50].replace("\n", " ")
        print(f"   📄 {source}: 距离 {score:.4f} (越小越相似)")
        print(f"      内容: {preview}...")
    print(f"{'='*60}\n")
    
    # 欧氏距离：保留距离 <= 阈值的
    filtered = [(doc, score) for doc, score in results if score <= score_threshold]
    print(f"📊 阈值: {score_threshold} | 保留 {len(filtered)}/{len(results)} 个文档")
    
    return filtered

def search_documents_mmr(query, k=8, lambda_mult=0.5):
    """
    使用 MMR (最大边际相关性) 检索，平衡相关性和多样性
    """
    vector_store = get_vector_store()
    if not vector_store:
        return []
    
    results = vector_store.max_marginal_relevance_search(
        query, 
        k=k, 
        lambda_mult=lambda_mult
    )
    return results