import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 指向本地模型路径（根据你实际存放的位置调整）
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

def search_documents(query, k=4):
    vector_store = get_vector_store()
    if not vector_store:
        return []
    results = vector_store.similarity_search(query, k=k)
    return results