from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_and_split(file_path):

    if file_path.endswith('.pdf'):
        loader=PyPDFLoader(file_path)
    else:
        loader=TextLoader(file_path,encoding='utf-8')

    documents=loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
   
    return text_splitter.split_documents(documents)