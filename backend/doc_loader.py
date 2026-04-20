from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re


def extract_paper_id_from_path(file_path: str) -> str:
    """
    从文件路径中提取论文 ID
    支持格式: /path/to/2305.12345.pdf 或 /path/to/2305.12345.txt
    """
    basename = os.path.basename(file_path)
    # 移除扩展名
    name_without_ext = os.path.splitext(basename)[0]
    # arXiv ID 格式: YYMM.NNNNN 或 YYMMNNNNN
    match = re.search(r'(\d{4}\.\d{4,5}|\d{7})', name_without_ext)
    if match:
        return match.group(1)
    return name_without_ext


def extract_category_from_path(file_path: str) -> str:
    """
    从文件路径中推断分类（如果路径中包含分类信息）
    例如: /path/to/cs.CV/paper.pdf
    """
    path_parts = file_path.split(os.sep)
    for part in path_parts:
        if part.startswith('cs.') or part.startswith('CV') or part.startswith('CL'):
            return part
    return "unknown"


def load_and_split(file_path, metadata_override: dict = None):
    """
    加载并分割文档，同时增强元数据
    
    参数:
        file_path: PDF 或 TXT 文件路径
        metadata_override: 额外的元数据（如 title, category, arxiv_id 等）
    
    返回:
        List[Document] 分割后的文档块，每个块都包含完整的元数据
    """
    # 1. 加载文档
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    
    documents = loader.load()
    
    # 2. 提取基础元数据（从文件路径）
    base_metadata = {
        "source": os.path.basename(file_path),
        "file_path": file_path,
        "paper_id": extract_paper_id_from_path(file_path),
        "category": extract_category_from_path(file_path)
    }
    
    # 3. 合并外部传入的元数据（优先级更高）
    if metadata_override:
        base_metadata.update(metadata_override)
    
    # 4. 为每个文档添加元数据
    for doc in documents:
        doc.metadata.update(base_metadata)
    
    # 5. 分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # 6. 为每个 chunk 添加索引（便于追溯）
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_total"] = len(chunks)
    
    print(f"  📄 文档分割完成: {len(chunks)} 个 chunks")
    if chunks:
        sample_meta = chunks[0].metadata
        print(f"     元数据示例: paper_id={sample_meta.get('paper_id', 'N/A')}, "
              f"category={sample_meta.get('category', 'N/A')}")
    
    return chunks


def load_and_split_simple(file_path):
    """
    简化版本（向后兼容）
    仅分割，不增强元数据
    """
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    
    return text_splitter.split_documents(documents)