from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
import os
import tempfile
from dotenv import load_dotenv

from .vector_store import create_vector_store, search_documents_with_score
from .doc_loader import load_and_split
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
        return {"status": "error", "message": "仅支持PDF和TXT文件"}
    
    original_filename = file.filename
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        docs = load_and_split(tmp_path)
        
        for doc in docs:
            doc.metadata['source'] = original_filename
            
        create_vector_store(docs)
        
        return {"status": "success", "chunks": len(docs)}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        os.unlink(tmp_path)

@app.post("/chat")
async def chat(request: dict):
    question = request.get("question", "")
    retrieved = search_documents_with_score(question, k=6, score_threshold=1.0)
    
    if retrieved:
        # 筛选高分文档（相似度 >= 0.7）
        high_relevant = [(doc, score) for doc, score in retrieved if score <= 0.9]
        
        if high_relevant:  # 有高分文档，使用 RAG
            # 对高相关文档去重
            unique_sources = {}
            for doc, score in high_relevant:
                source = doc.metadata.get("source", "未知")
                if source not in unique_sources:
                    unique_sources[source] = doc
            
            relevant_docs = list(unique_sources.values())[:3]
            context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
            sources = list(set([doc.metadata.get("source", "未知") for doc in relevant_docs]))
            
            system_prompt = f"""你是一个智能助手。请基于以下参考内容回答用户问题。

参考内容：
{context}

注意：如果问题无法从参考内容中找到答案，请直接说"资料库中没有相关信息"，不要编造。"""
            
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.7
            )
            answer = response.choices[0].message.content
            
            # 添加参考来源
            if "资料库中没有相关信息" not in answer:
                formatted_sources = [f"{i+1}. {source}" for i, source in enumerate(sources)]
                answer = answer + "\n\n---\n📚 参考来源：\n" + "\n".join(formatted_sources)
            
            return {"answer": answer}
    
    # 没有检索到文档 或 没有高分文档 → 使用模型自身知识
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个智能助手，根据你的知识回答用户问题。"},
            {"role": "user", "content": question}
        ],
        temperature=0.7
    )
    answer = response.choices[0].message.content
    
    return {"answer": answer}

@app.get("/health")
async def health():
    return {"status": "ok"}