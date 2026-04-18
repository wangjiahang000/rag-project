import os
import pickle
import jieba
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import logging

# 禁用警告
logging.set_verbosity_error()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 使用本地模型路径
model_path = r"C:\Users\35717\Desktop\my_rag_project\models\paraphrase-multilingual-MiniLM-L12-v2"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ 模型路径不存在: {model_path}")

print(f"✅ 加载本地模型: {model_path}")

embeddings = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

VECTOR_STORE_PATH = "./storage/chroma_db"
BM25_INDEX_PATH = "./storage/bm25_index.pkl"
DOCUMENTS_CACHE_PATH = "./storage/documents_cache.pkl"


def get_vector_store():
    """获取向量数据库"""
    if os.path.exists(VECTOR_STORE_PATH) and os.listdir(VECTOR_STORE_PATH):
        return Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )
    return None


def tokenize_chinese(text):
    """中文分词（用于BM25）"""
    return list(jieba.cut(text))


def build_bm25_index(documents):
    """构建BM25索引"""
    print(f"📝 正在构建BM25索引，共 {len(documents)} 个文档...")
    
    # 对所有文档内容进行分词
    tokenized_docs = [tokenize_chinese(doc.page_content) for doc in documents]
    
    # 创建BM25索引
    bm25 = BM25Okapi(tokenized_docs)
    
    # 保存索引和文档内容（用于后续检索时恢复）
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump(bm25, f)
    
    # 保存文档列表（用于检索时返回原始文档）
    with open(DOCUMENTS_CACHE_PATH, 'wb') as f:
        pickle.dump(documents, f)
    
    print(f"✅ BM25索引构建完成")
    return bm25


def load_bm25_index():
    """加载BM25索引"""
    if os.path.exists(BM25_INDEX_PATH) and os.path.exists(DOCUMENTS_CACHE_PATH):
        with open(BM25_INDEX_PATH, 'rb') as f:
            bm25 = pickle.load(f)
        with open(DOCUMENTS_CACHE_PATH, 'rb') as f:
            documents = pickle.load(f)
        return bm25, documents
    return None, None


def search_bm25(query, k=10):
    """BM25关键词检索"""
    bm25, documents = load_bm25_index()
    if not bm25:
        return []
    
    tokenized_query = tokenize_chinese(query)
    scores = bm25.get_scores(tokenized_query)
    
    # 获取top-k的索引
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                'document': documents[idx],
                'bm25_score': scores[idx]
            })
    
    print(f"🔍 BM25检索到 {len(results)} 个结果")
    return results


def create_vector_store(documents):
    """创建向量数据库和BM25索引"""
    # 创建向量数据库
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_PATH
    )
    
    # 构建BM25索引
    build_bm25_index(documents)
    
    return vector_store


def hybrid_search(query, k=10, bm25_weight=0.3, vector_weight=0.7, score_threshold=1.05):
    """
    混合检索：结合向量检索和BM25关键词检索
    
    参数:
        query: 查询字符串
        k: 最终返回的文档数量
        bm25_weight: BM25权重（0-1之间）
        vector_weight: 向量检索权重（0-1之间）
        score_threshold: 向量检索的距离阈值（越小越严格）
    
    返回:
        排序后的文档列表
    """
    vector_store = get_vector_store()
    if not vector_store:
        print("❌ 向量数据库不存在")
        return []
    
    # 1. 向量检索
    vector_results = vector_store.similarity_search_with_score(query, k=k * 2)
    
    # 过滤掉距离大于阈值的
    vector_results = [(doc, score) for doc, score in vector_results if score <= score_threshold]
    
    print(f"\n🔍 向量检索到 {len(vector_results)} 个结果")
    for doc, score in vector_results[:3]:
        preview = doc.page_content[:50].replace("\n", " ")
        print(f"   向量: {preview}... (距离: {score:.4f})")
    
    # 2. BM25检索
    bm25_results = search_bm25(query, k=k * 2)
    print(f"🔍 BM25检索到 {len(bm25_results)} 个结果")
    for item in bm25_results[:3]:
        preview = item['document'].page_content[:50].replace("\n", " ")
        print(f"   BM25: {preview}... (分数: {item['bm25_score']:.2f})")
    
    # 3. 合并结果（使用加权分数）
    doc_scores = {}
    
    # 向量检索结果（距离越小越相似，需要转换为分数）
    for doc, distance in vector_results:
        # 将距离转换为相似度分数（距离越小，分数越高）
        vector_score = 1.0 / (1.0 + distance)
        doc_scores[doc.page_content] = {
            'doc': doc,
            'total_score': vector_weight * vector_score,
            'vector_score': vector_score,
            'bm25_score': 0
        }
    
    # BM25检索结果
    for item in bm25_results:
        doc = item['document']
        bm25_score = item['bm25_score']
        # BM25分数归一化（假设最大分数为50左右）
        normalized_bm25 = min(bm25_score / 50.0, 1.0)
        
        if doc.page_content in doc_scores:
            doc_scores[doc.page_content]['bm25_score'] = normalized_bm25
            doc_scores[doc.page_content]['total_score'] += bm25_weight * normalized_bm25
        else:
            doc_scores[doc.page_content] = {
                'doc': doc,
                'total_score': bm25_weight * normalized_bm25,
                'vector_score': 0,
                'bm25_score': normalized_bm25
            }
    
    # 按总分排序
    sorted_results = sorted(doc_scores.values(), key=lambda x: x['total_score'], reverse=True)
    
    # 打印调试信息
    print(f"\n{'='*60}")
    print(f"🔍 混合检索结果 (BM25权重={bm25_weight}, 向量权重={vector_weight})")
    print(f"查询: {query}")
    print(f"{'='*60}")
    for i, item in enumerate(sorted_results[:k]):
        doc = item['doc']
        source = doc.metadata.get("source", "未知")
        print(f"   {i+1}. {source}: 总分={item['total_score']:.4f} "
              f"(向量={item['vector_score']:.4f}, BM25={item['bm25_score']:.4f})")
        preview = doc.page_content[:60].replace("\n", " ")
        print(f"      内容: {preview}...")
    print(f"{'='*60}\n")
    
    return [item['doc'] for item in sorted_results[:k]]


def search_documents_with_score(query, k=10, score_threshold=1.05):
    """
    兼容旧接口：使用混合检索
    """
    return hybrid_search(query, k=k, score_threshold=score_threshold)