import os 
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import tempfile
from dotenv import load_dotenv

from .vector_store import create_vector_store, search_documents_with_score
from .doc_loader import load_and_split
from .literature_search import get_searcher
from batch_import.mysql_manager import get_paper_titles_batch
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

@app.post("/search_literature")
async def search_literature(request: dict):
    """搜索arXiv论文"""
    query = request.get("query", "")
    max_results = request.get("max_results", 10)
    
    if not query:
        return {"status": "error", "message": "请输入搜索关键词"}
    
    searcher = get_searcher()
    papers = searcher.search_papers(query, max_results)
    
    return {"status": "success", "papers": papers, "count": len(papers)}


@app.post("/add_paper_to_kb")
async def add_paper_to_kb(request: dict):
    """下载论文PDF并加入知识库"""
    paper_id = request.get("paper_id", "")
    paper_title = request.get("paper_title", "")
    
    if not paper_id:
        return {"status": "error", "message": "论文ID不能为空"}
    
    searcher = get_searcher()
    
    # 下载PDF
    filepath = searcher.download_paper(paper_id, "./uploads")
    
    if not filepath:
        return {"status": "error", "message": f"下载失败: {paper_id}"}
    
    try:
        # 加载并切片
        from .doc_loader import load_and_split
        docs = load_and_split(filepath)
        
        # 设置来源名称
        for doc in docs:
            doc.metadata['source'] = paper_title or paper_id
        
        # 加入向量库（增量添加）
        from .vector_store import get_vector_store, create_vector_store
        existing_store = get_vector_store()
        
        if existing_store:
            # 增量添加
            existing_store.add_documents(docs)
            existing_store.persist()
        else:
            # 新建向量库
            create_vector_store(docs)
        
        # 可选：删除下载的PDF文件
        # os.unlink(filepath)
        
        return {"status": "success", "message": f"已添加论文: {paper_title}", "chunks": len(docs)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat")
async def chat(request: dict):
    question = request.get("question", "")
    history = request.get("history", [])
    print(f"收到历史对话: {history}")
    
    # 检索相关文档
    retrieved = search_documents_with_score(question, k=10, score_threshold=1.05)

    # 构建消息列表
    messages = []
    
    if retrieved:
        # 按来源合并同一文档的所有块
        doc_chunks = {}
        for doc in retrieved:
            source = doc.metadata.get("source", "未知")
            if source not in doc_chunks:
                doc_chunks[source] = []
            doc_chunks[source].append(doc.page_content)
        
        # 合并每个文档的内容（取前3个文档）
        merged_docs = []
        for source, chunks in list(doc_chunks.items())[:5]:
            # 合并同一文档的所有块
            full_content = "\n\n...\n\n".join(chunks)
            merged_docs.append(full_content)
        
        context = "\n\n---\n\n".join(merged_docs)
        sources = list(doc_chunks.keys())[:3] 
        
        print(f"📚 使用文档: {sources}")
        print(f"📄 上下文长度: {len(context)} 字符")
    
        system_prompt = f"""你是一个智能猫娘文献内容问答助手。请基于以下参考内容回答用户问题.

参考内容：
{context}
1. 如果参考内容中包含某个概念的定义或解释，请直接引用并说明
2. 如果参考内容中只有零散的提及（如机构名称、项目名称），请明确告诉用户"参考内容中没有找到完整的定义，只提到了以下相关内容："
3. 如果问题无法从参考内容中找到答案，请直接说"资料库中没有相关信息"，然后使用你的知识库回答，你不要编造参考文献"""

        messages.append({"role": "system", "content": system_prompt})
    else:
        print("❌ 没有检索到文档，使用模型自身知识")
        messages.append({"role": "system", "content": "你是一个智能助手，根据你的知识回答用户问题。"})
    
    # 添加历史对话
    for h in history[-5:]:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    
    # 添加当前问题
    messages.append({"role": "user", "content": question})
    print(f"发送给DeepSeek的消息: {messages}")
    
    # 调用DeepSeek
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.7
    )
    answer = response.choices[0].message.content
    
    # 如果有检索结果且答案不是拒绝回答，添加来源（使用论文标题）
    if retrieved and "资料库中没有相关信息" not in answer:
        # 提取 arxiv_id（去掉 .txt 后缀）
        arxiv_ids = [s.replace('.txt', '') for s in sources]
        
        # 批量从数据库获取论文标题
        titles_map = get_paper_titles_batch(arxiv_ids)
        
        # 格式化参考文献，显示标题
        formatted_sources = []
        for i, arxiv_id in enumerate(arxiv_ids):
            title = titles_map.get(arxiv_id)
            if title:
                # 截断过长的标题
                if len(title) > 80:
                    title = title[:77] + "..."
                formatted_sources.append(f"{i+1}. [{arxiv_id}] {title}")
            else:
                formatted_sources.append(f"{i+1}. {arxiv_id}（标题未找到）")
        
        answer = answer + "\n\n---\n📚 参考来源：\n" + "\n".join(formatted_sources)
    
    print(f"🤖 LLM 回答:\n{answer}\n")
    
    return {"answer": answer}

@app.get("/health")
async def health():
    return {"status": "ok"}