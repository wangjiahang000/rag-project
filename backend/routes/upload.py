import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Depends
from core import RAGSystem
from backend.dependencies import get_rag
from backend.models import UploadResponse

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    rag: RAGSystem = Depends(get_rag)
):
    if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
        return UploadResponse(status="error", chunks=0, message="仅支持PDF和TXT文件")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        chunks = rag.upload(tmp_path, {'source': file.filename})
        return UploadResponse(status="success", chunks=chunks)
    except Exception as e:
        return UploadResponse(status="error", chunks=0, message=str(e))
    finally:
        os.unlink(tmp_path)