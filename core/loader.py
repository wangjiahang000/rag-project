import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Optional

class DocumentLoader:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
    
    def load_file(self, path: str) -> List[Document]:
        if path.endswith('.pdf'):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path, encoding='utf-8')
        return loader.load()
    
    def split(self, docs: List[Document], metadata: dict = None) -> List[Document]:
        for doc in docs:
            if metadata:
                doc.metadata.update(metadata)
        chunks = self.splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
        return chunks
    
    def process(self, path: str, metadata: dict = None) -> List[Document]:
        docs = self.load_file(path)
        return self.split(docs, metadata)