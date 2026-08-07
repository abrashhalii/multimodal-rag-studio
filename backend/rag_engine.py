"""
RAG orchestration.

Keeps the two public function names main.py imports:
    process_and_store_document(file_path, book_type)
    query_rag_system(user_message, book_type, history=None)

Day 1 scope: routing, deterministic location answers, page answers, semantic
answers. Global summarisation (Bug 2) and conversation memory (Bug 3) get their
full implementations on Day 2 - the hooks are already wired here.
"""

from __future__ import annotations

from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
import router
from ingest import get_vector_db, ingest_pdf
from llm_provider import LLMNotConfigured, get_llm
from page_index import BookIndex


# ---------------------------------------------------------------- prompts
ANSWER_TEMPLATE = """You are a precise Interactive Study Tutor answering questions about a book.

Ground every statement in the context below. You may SYNTHESISE an answer from what
the passages show, even when no single passage states it outright - for a question
like "who is X?", build the picture from how X is described and what X does across
the passages. What you must not do is bring in knowledge from outside the context,
or invent page numbers.

Only say "I cannot find the answer to this in the book." when the context contains
nothing relevant to the question at all. If it is partially relevant, answer with what
is there and say plainly what is missing.

Every context block begins with a location header in this form:
[Source: PDF Viewer Page X | Printed Page Y | Lines A-B | book]
When you state a fact, cite its Printed Page. "Printed Page" is the number printed on
the paper, which is what the reader sees - always cite that, not the PDF viewer page.

{history_block}
Context:
{context}

Question: {question}

Answer:"""

PAGE_TEMPLATE = """You are a precise Interactive Study Tutor.

Below is the complete, verbatim text of printed page {printed_page} of the book
(PDF viewer page {viewer_page}). Lines are numbered exactly as they appear on the page.

Answer the reader's question using only this page. If the page does not contain the
answer, say so plainly rather than guessing.

{history_block}
--- BEGIN PAGE {printed_page} ---
{page_text}
--- END PAGE {printed_page} ---

Question: {question}

Answer:"""


def _history_block(history: Optional[List[dict]], limit: int = 6) -> str:
    """Day 2 replaces this with rewriting + summarisation. Day 1 just formats."""
    if not history:
        return ""
    recent = history[-limit:]
    lines = []
    for turn in recent:
        role = (turn.get("role") or "user").lower()
        who = "Reader" if role in ("user", "human") else "Tutor"
        lines.append(f"{who}: {turn.get('content', '').strip()}")
    return "Conversation so far:\n" + "\n".join(lines) + "\n"


def _format_docs(docs) -> str:
    out = []
    for d in docs:
        head = d.metadata.get("location_header", "")
        out.append(f"{head}\n{d.page_content}" if head else d.page_content)
    return "\n\n".join(out)


# ---------------------------------------------------------------- public API
def process_and_store_document(file_path: str, book_type: str = "coding") -> str:
    stats = ingest_pdf(file_path, book_type)
    return (f"Indexed {stats['pages']} pages into {stats['chunks']} chunks "
            f"({stats['label_method']}, offset {stats['offset']}).")


def query_rag_system(user_message: str,
                     book_type: str = "coding",
                     history: Optional[List[dict]] = None) -> str:
    route = router.classify(user_message)
    if config.DEBUG_LOG:
        print(f"[route] {route}  ({route.reason})  book={book_type}")

    if route.kind == "page_line":
        return _answer_page_line(route, book_type)
    if route.kind == "page":
        return _answer_page(route, user_message, book_type, history)
    if route.kind == "memory":
        return _answer_memory(user_message, history)
    if route.kind == "global":
        # Day 2: hierarchical summary tree. Day 1: wide retrieval so it degrades
        # gracefully instead of erroring during testing.
        return _answer_semantic(user_message, book_type, history, k=config.GLOBAL_K)
    return _answer_semantic(user_message, book_type, history, k=config.RETRIEVAL_K)


# ---------------------------------------------------------------- routes
def _answer_page_line(route, book_type: str) -> str:
    """Deterministic. No embeddings, no LLM, no possibility of hallucination."""
    idx = BookIndex.load(book_type)
    if idx is None:
        return "No book has been indexed for this mode yet. Please upload a PDF first."

    page = idx.by_printed(route.page)
    if page is None:
        return (f"Printed page {route.page} is not in this book "
                f"({idx.page_count} pages indexed).")

    text = page.line(route.line)
    if text is None:
        return (f"Printed page {route.page} has {len(page.lines)} lines, "
                f"so there is no line {route.line}.")

    return (f"**Printed page {route.page}, line {route.line}:**\n\n"
            f"> {text}\n\n"
            f"*(PDF viewer page {page.pdf_index + 1}; "
            f"this page has {len(page.lines)} lines.)*")


def _answer_page(route, question: str, book_type: str,
                 history: Optional[List[dict]]) -> str:
    """Whole page pulled by exact lookup, then read by the LLM."""
    idx = BookIndex.load(book_type)
    if idx is None:
        return "No book has been indexed for this mode yet. Please upload a PDF first."

    page = idx.by_printed(route.page)
    if page is None:
        return (f"Printed page {route.page} is not in this book "
                f"({idx.page_count} pages indexed).")

    numbered = "\n".join(f"{i:>3} | {t}" for i, t in enumerate(page.lines, 1))

    try:
        llm = get_llm()
    except LLMNotConfigured as e:
        return f"LLM not configured: {e}"

    chain = ChatPromptTemplate.from_template(PAGE_TEMPLATE) | llm | StrOutputParser()
    return chain.invoke({
        "printed_page": page.printed_page,
        "viewer_page": page.pdf_index + 1,
        "page_text": numbered,
        "question": question,
        "history_block": _history_block(history),
    })


def _answer_memory(question: str, history: Optional[List[dict]]) -> str:
    """Answered from the transcript itself - retrieval is irrelevant here."""
    if not history:
        return ("I don't have any earlier messages in this conversation yet - "
                "this looks like your first question.")
    prior = [t for t in history if (t.get("role") or "").lower() in ("user", "human")]
    if not prior:
        return "I don't see any earlier questions from you in this conversation."
    return f"Your previous question was:\n\n> {prior[-1].get('content', '').strip()}"


def _answer_semantic(question: str, book_type: str,
                     history: Optional[List[dict]], k: int) -> str:
    try:
        llm = get_llm()
    except LLMNotConfigured as e:
        return f"LLM not configured: {e}"

    db = get_vector_db(book_type)
    # Back-of-book index rows ("term, 194") are pointers, never answers. Front
    # matter stays IN: the table of contents is the best chunk in the book for
    # structural questions about what it covers.
    docs = db.similarity_search(question, k=k,
                                filter={"is_index": {"$eq": False}})
    
    if not docs:
        return ("Nothing has been indexed for this mode yet, or the book contains "
                "no matching passages. Try uploading the PDF again.")

    if config.DEBUG_LOG:
        pages = [d.metadata.get("printed_page") for d in docs[:8]]
        print(f"[retrieve] k={k} hits, top pages: {pages}")

    chain = ChatPromptTemplate.from_template(ANSWER_TEMPLATE) | llm | StrOutputParser()
    return chain.invoke({
        "context": _format_docs(docs),
        "question": question,
        "history_block": _history_block(history),
    })
