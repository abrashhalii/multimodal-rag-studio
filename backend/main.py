from typing import List, Optional

import os
import shutil

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import llm_provider
import rate_limit
import summarize
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
    """System state - what is loaded, how it was resolved, and how the client
    side rate limiter is behaving. Useful live: it lets the limiter be SHOWN
    rather than described."""
    books = {}
    for bt in config.BOOK_TYPES:
        idx = BookIndex.load(bt)
        if idx is None:
            books[bt] = None
            continue
        tree = summarize.load_summaries(bt)
        pages_with_text = sum(1 for p in idx.pages if p.lines)
        books[bt] = {
            "filename": idx.filename,
            "pages": idx.page_count,
            "pages_with_text": pages_with_text,
            "page_numbering": idx.label_method,
            "margin_bands": idx.margin_note,
            "offset": idx.offset,
            "offset_rule": f"pdf index = printed page + {idx.offset} (0-based)",
            "chapters": len(idx.chapters),
            "chapter_detection": idx.chapter_method,
            "extraction": idx.pages[0].source if idx.pages else "unknown",
            "summary_tree": None if not tree else {
                "chapters_summarised": len(tree["chapters"]),
                "book_overview_words": len(tree["book"].split()),
            },
        }

    return {
        "llm": llm_provider.describe(),
        "embedding_model": config.EMBEDDING_MODEL,
        "rate_limit": rate_limit.get_bucket().stats(),
        "config": {
            "declared_rpm": config.LLM_RPM,
            "safety_factor": config.RATE_LIMIT_SAFETY,
            "effective_rpm": max(1, int(config.LLM_RPM * config.RATE_LIMIT_SAFETY)),
            "retrieval_k": config.RETRIEVAL_K,
            "chunk_chars": config.CHUNK_CHARS,
        },
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
