"""
Ingestion: page-bounded chunking with location headers injected into the text.

Two decisions here that the starter code gets wrong, and that matter more than
anything else in this project:

1. CHUNKS NEVER CROSS A PAGE BOUNDARY.
   The starter concatenates the whole document and slices it every 1000 chars,
   so a chunk routinely spans pages 33-34. Then "what is on page 34?" has no
   provable answer, because half the chunk is not on page 34. Chunking inside
   page boundaries costs a little packing efficiency and buys exact attribution.
   This is also why semantic chunking is the wrong tool for this assignment:
   it places breakpoints at meaning boundaries, which ignore page boundaries.

2. LOCATION IS WRITTEN INTO page_content, NOT JUST metadata.
   Chroma metadata never reaches the LLM's context window - only page_content
   does. Injecting "[Source: PDF Viewer Page 38 | Printed Page 34 | Lines 1-13]"
   into the text means the location is embedded (so it is at least partly
   searchable) AND visible to the model (so it can cite the page instead of
   inventing one).

Chunking is line-based rather than character-based so that every chunk carries
an exact, verifiable line range and no line is ever split down the middle.
"""

from __future__ import annotations

import shutil
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from page_index import BookIndex, PageRecord, build_index, index_report

_embeddings = None


def get_embeddings():
    """Loaded once, lazily - the model download happens on first use."""
    global _embeddings
    if _embeddings is None:
        model = config.EMBEDDING_MODEL
        model_kwargs = {"device": config.EMBEDDING_DEVICE}
        encode_kwargs = {
            "batch_size": config.EMBEDDING_BATCH_SIZE,
            "normalize_embeddings": True,
        }
        if model.lower().startswith("baai/bge"):
            # BGE models expect an instruction prefix on the QUERY side only.
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
            _embeddings = HuggingFaceBgeEmbeddings(
                model_name=model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                query_instruction="Represent this sentence for searching relevant passages: ",
            )
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
    return _embeddings


# ---------------------------------------------------------------- chunking
def chunk_page(page: PageRecord, book_type: str, filename: str) -> List[Document]:
    """Split one page into line-aligned chunks. Never crosses into another page."""
    if not page.lines:
        return []

    chunks, start = [], 0
    n = len(page.lines)

    while start < n:
        end, size = start, 0
        while end < n and (size + len(page.lines[end]) + 1) <= config.CHUNK_CHARS:
            size += len(page.lines[end]) + 1
            end += 1
        if end == start:            # a single line longer than CHUNK_CHARS
            end = start + 1

        body = "\n".join(page.lines[start:end])
        first_line, last_line = start + 1, end

        header = (
            f"[Source: PDF Viewer Page {page.pdf_index + 1} | "
            f"Printed Page {page.printed_page} | "
            f"Lines {first_line}-{last_line} | {book_type}]"
        )

        chunks.append(Document(
            page_content=f"{header}\n{body}",
            metadata={
                "book_type": book_type,
                "filename": filename,
                "pdf_index": page.pdf_index,
                "pdf_viewer_page": page.pdf_index + 1,
                "printed_page": str(page.printed_page),
                "printed_page_num": int(page.printed_page)
                if str(page.printed_page).isdigit() else -1,
                "line_start": first_line,
                "line_end": last_line,
                "is_front_matter": page.is_front_matter,
                "is_index": page.is_index,
                "is_sfx_only": bool(page.source == "vision" and page.lines
                                    and all(l.startswith("[sfx]")
                                            for l in page.lines)),
                "chapter_id": page.chapter_id,
                "chapter_title": page.chapter_title or "",
                "source": page.source,
            },
        ))

        if end >= n:
            break
        start = max(end - config.CHUNK_OVERLAP_LINES, start + 1)

    return chunks


def build_documents(idx: BookIndex) -> List[Document]:
    docs = []
    for page in idx.pages:
        docs.extend(chunk_page(page, idx.book_type, idx.filename))
    return docs


# ---------------------------------------------------------------- vector store
def get_vector_db(book_type: str) -> Chroma:
    """One collection per book type. The starter dumped all three into one."""
    return Chroma(
        collection_name=config.collection_name(book_type),
        embedding_function=get_embeddings(),
        persist_directory=config.DB_DIR,
    )


def reset_collection(book_type: str) -> None:
    """Clear before re-ingesting, or every upload silently duplicates chunks."""
    try:
        db = get_vector_db(book_type)
        db.delete_collection()
    except Exception:
        pass


def ingest_pdf(file_path: str, book_type: str = "coding",
               force_vision: bool = False) -> dict:
    """Full pipeline: extract -> index -> chunk -> embed -> store."""
    if book_type == "manga":
        import vision
        from page_index import build_index_from_vision
        result = vision.extract_pdf(file_path, book_type, force=force_vision)
        idx = build_index_from_vision(file_path, book_type, result)
    else:
        idx = build_index(file_path, book_type)
    docs = build_documents(idx)

    reset_collection(book_type)
    db = get_vector_db(book_type)

    batch = 256
    for i in range(0, len(docs), batch):
        db.add_documents(docs[i:i + batch])

    stats = {
        "book_type": book_type,
        "filename": idx.filename,
        "pages": idx.page_count,
        "chunks": len(docs),
        "label_method": idx.label_method,
        "offset": idx.offset,
        "collection": config.collection_name(book_type),
    }

    if config.DEBUG_LOG:
        print(f"\n[ingest] {book_type}")
        print(index_report(idx))
        print(f"  chunks          : {len(docs)}")
        print(f"  collection      : {stats['collection']}")
        if docs:
            print(f"  sample chunk    : {docs[0].page_content[:120]}...")
        print()

    return stats
