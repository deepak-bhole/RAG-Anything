from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from raganything.batch_parser import BatchParser
from app.core.rag_manager import get_rag_instance, rag_configs
import os
import shutil
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    working_dir: str
    multimodal_content: Optional[List[Dict[str, Any]]] = None

def get_upload_directory():
    upload_dir = "./uploaded_files"
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir

@router.post("/process-batch", summary="Upload and process a batch of documents")
async def process_document_batch(
    files: List[UploadFile] = File(...),
    working_dir: str = Form("./rag_storage_ollama"),
    parser: str = Form("mineru"),
    host: str = Form("http://localhost:11434"),
    llm_model: str = Form("gemma3:1b"),
    vision_model: str = Form("llava:latest"),
    embedding_model: str = Form("bge-m3:latest"),
    embedding_dim: int = Form(768),
    upload_directory: str = Depends(get_upload_directory)
):
    config_data = {
        "working_dir": working_dir,
        "parser": parser,
        "host": host,
        "llm_model": llm_model,
        "vision_model": vision_model,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
    }

    file_paths = []
    try:
        for file in files:
            file_path = os.path.join(upload_directory, file.filename)
            file_paths.append(file_path)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        rag_instance = get_rag_instance(config_data)
        
        # Storing the config is crucial for the query endpoint
        rag_configs[working_dir] = config_data
        
        batch_parser = BatchParser(parser_type=config_data["parser"], max_workers=2) # Reduced workers to prevent overload
        result = await batch_parser.process_batch_async(
            file_paths=file_paths,
            output_dir=config_data["working_dir"],
            parse_method="auto"
        )
        
        for file_path in result.successful_files:
             await rag_instance.process_document_complete(
                 file_path=file_path, 
                 output_dir=config_data["working_dir"]
            )

        return JSONResponse(content={
            "message": f"Batch processing complete. {len(result.successful_files)} of {len(files)} files processed.",
            "summary": result.summary()
        })
    except Exception as e:
        logger.error(f"Error during batch processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)

@router.post("/query", summary="Query the processed documents")
async def query_model(request: QueryRequest):
    if request.working_dir not in rag_configs:
         raise HTTPException(status_code=404, detail=f"Working directory '{request.working_dir}' has not been processed in this session. Please process documents first.")
    
    try:
        rag_instance = get_rag_instance(rag_configs[request.working_dir])

        if request.multimodal_content:
            result = await rag_instance.aquery_with_multimodal(
                request.query,
                multimodal_content=request.multimodal_content,
                mode="hybrid"
            )
        else:
            result = await rag_instance.aquery(request.query, mode="hybrid")
            
        return JSONResponse(content={"response": result})
    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    


