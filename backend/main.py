from typing import List, Optional

import os
import shutil

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import llm_provider
from page_index import BookIndex
from rag_engine import process_and_store_document, query_rag_system

app = FastAPI(title="Zenvyrolabs Task 3 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    book_type: str = "coding"
    # Bug 3: the frontend now sends the full transcript. Optional with a default
    # so an old client (or curl) still works against the same endpoint.
    history: Optional[List[ChatTurn]] = None


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...),
                          book_type: str = Form("coding")):
    try:
        file_path = os.path.join(config.TEMP_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        detail = process_and_store_document(file_path, book_type)
        return {"message": f"Processed {file.filename} as {book_type}. {detail}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        history = [t.model_dump() for t in request.history] if request.history else None
        answer = query_rag_system(request.message, request.book_type, history)
        return {"answer": answer}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to query system: {str(e)}"}


@app.get("/api/status")
async def status():
    """Handy during the live demo - proves which book is loaded in which mode."""
    books = {}
    for bt in config.BOOK_TYPES:
        idx = BookIndex.load(bt)
        books[bt] = None if idx is None else {
            "filename": idx.filename,
            "pages": idx.page_count,
            "label_method": idx.label_method,
            "offset": idx.offset,
        }
    return {
        "llm": llm_provider.describe(),
        "embedding_model": config.EMBEDDING_MODEL,
        "books": books,
    }


frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    print(f"[startup] LLM      : {llm_provider.describe()}")
    print(f"[startup] Embedding: {config.EMBEDDING_MODEL} ({config.EMBEDDING_DEVICE})")
    uvicorn.run("main:app", host=config.HOST, port=config.PORT)
