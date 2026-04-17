from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
import os
import tempfile
from dotenv import load_dotenv

from .vector_store import create_vector_store, search_documents
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
    
    # 保存原始文件名
    original_filename = file.filename
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        docs = load_and_split(tmp_path)
        
        # 修改1：将临时路径替换为原始文件名
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
    retrieved_docs = search_documents(question, k=4)
    
    if retrieved_docs:
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
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
        
        # 修改2：只有当答案不是"资料库中没有相关信息"时，才添加参考来源
        if "资料库中没有相关信息" not in answer:
            sources = []
            for i, doc in enumerate(retrieved_docs):
                source = doc.metadata.get("source", "未知")
                page = doc.metadata.get("page", "")
                sources.append(f"{i+1}. {source}" + (f" 第{page}页" if page else ""))
            answer = answer + "\n\n---\n📚 参考来源：\n" + "\n".join(sources)
    else:
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