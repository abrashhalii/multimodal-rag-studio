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
import query_rewrite
import rate_limit
import router
import summarize
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


GLOBAL_TEMPLATE = """You are a precise Interactive Study Tutor answering a question about
a book AS A WHOLE.

You are given a whole-book overview followed by a summary of EVERY chapter, in order.
Together these are derived from every page of the book, so you may answer broad questions
about structure, themes, coverage and comparisons across the whole work.

Cite chapters by name and give their printed page range when you refer to them. Do not
invent details that are not in the summaries; if the question needs a specific passage
rather than an overview, say so and suggest what to ask instead.

{history_block}
--- WHOLE-BOOK OVERVIEW ---
{book_summary}

--- CHAPTER SUMMARIES ---
{chapter_summaries}
--- END ---

Question: {question}

Answer:"""

CHAPTER_TEMPLATE = """You are a precise Interactive Study Tutor answering a question about
one chapter of a book.

You are given that chapter's summary followed by verbatim passages from it. Prefer the
verbatim passages for specifics and the summary for shape and scope. Cite printed pages.

{history_block}
--- CHAPTER: {chapter_title} (printed pages {chapter_pages}) ---
{chapter_summary}

--- PASSAGES FROM THIS CHAPTER ---
{context}
--- END ---

Question: {question}

Answer:"""


def _history_block(history: Optional[List[dict]]) -> str:
    """Transcript for the model. Retrieval uses the REWRITTEN query, not this."""
    text = query_rewrite.format_history(history)
    return f"Conversation so far:\n{text}\n" if text else ""


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


# ---------------------------------------------------------------- public API
def process_and_store_document(file_path: str, book_type: str = "coding") -> str:
    stats = ingest_pdf(file_path, book_type)
    detail = (f"Indexed {stats['pages']} pages into {stats['chunks']} chunks "
              f"({stats['label_method']}, offset {stats['offset']}).")

    if config.BUILD_SUMMARIES_ON_INGEST:
        idx = BookIndex.load(book_type)
        if idx and idx.chapters:
            try:
                res = summarize.build_summaries(idx)
                detail += (f" Summary tree: {len(res['chapters'])} chapters "
                           f"({res['built']} built, {res['reused']} cached).")
            except Exception as e:
                detail += f" Summary tree failed: {e}"
        else:
            detail += " No chapters detected - global questions will use wide retrieval."
    return detail


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
    # Follow-ups are rewritten into standalone questions BEFORE retrieval.
    # The original transcript still reaches the model via _history_block.
    search_q = query_rewrite.rewrite(user_message, history)

    if route.kind == "global":
        return _answer_global(user_message, book_type, history)
    if route.kind == "chapter":
        return _answer_chapter(route, user_message, search_q, book_type, history)
    return _answer_semantic(search_q, book_type, history,
                            k=config.RETRIEVAL_K, question=user_message)


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
    rate_limit.throttle("page answer")
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


def _answer_semantic(search_q: str, book_type: str,
                     history: Optional[List[dict]], k: int,
                     question: str = None) -> str:
    """search_q drives retrieval; question is what the reader actually typed."""
    question = question or search_q
    try:
        llm = get_llm()
    except LLMNotConfigured as e:
        return f"LLM not configured: {e}"

    db = get_vector_db(book_type)
    # search_q, not question: for a follow-up these differ, and retrieval must
    # run on the REWRITTEN standalone form or the pronoun never resolves.
    # Back-of-book index rows ("term, 194") are pointers, never answers.
    # Front matter stays IN - the table of contents is the best chunk in the
    # book for structural questions.
    docs = db.similarity_search(search_q, k=k,
                                filter={"is_index": {"$eq": False}})
    if not docs:
        return ("Nothing has been indexed for this mode yet, or the book contains "
                "no matching passages. Try uploading the PDF again.")

    if config.DEBUG_LOG:
        pages = [d.metadata.get("printed_page") for d in docs[:8]]
        print(f"[retrieve] k={k} hits, top pages: {pages}")

    chain = ChatPromptTemplate.from_template(ANSWER_TEMPLATE) | llm | StrOutputParser()
    rate_limit.throttle("semantic answer")
    return chain.invoke({
        "context": _format_docs(docs),
        "question": question,
        "history_block": _history_block(history),
    })


def _answer_global(question: str, book_type: str,
                   history: Optional[List[dict]]) -> str:
    """Answered from the summary tree - derived from EVERY page, not a top-k sample."""
    try:
        llm = get_llm()
    except LLMNotConfigured as e:
        return f"LLM not configured: {e}"

    tree = summarize.load_summaries(book_type)
    if not tree or not tree["chapters"]:
        idx = BookIndex.load(book_type)
        if not (idx and idx.chapters):
            # No chapter structure (e.g. a comic). Fall back to wide retrieval.
            return _answer_semantic(question, book_type, history,
                                    k=config.GLOBAL_K)
        # Deliberately NOT built here. Building takes ~50 paced API calls and
        # minutes of wall clock; doing that inside a chat request means the user
        # stares at a spinner and a quota error surfaces as a broken answer.
        # Build it out of band instead.
        return ("The summary tree for this book has not been built yet, so I "
                "can't answer whole-book questions. Run:\n\n"
                f"    python tools/build_summaries.py {book_type}\n\n"
                "It takes a couple of minutes and is cached afterwards. "
                "Page, chapter and passage questions work now.")

    joined = "\n\n".join(
        f"{c['title']} (printed pages {c['pages']}):\n{c['summary']}"
        for c in tree["chapters"])

    if config.DEBUG_LOG:
        print(f"[global] summary tree: {len(tree['chapters'])} chapters")

    chain = ChatPromptTemplate.from_template(GLOBAL_TEMPLATE) | llm | StrOutputParser()
    rate_limit.throttle("global answer")
    return chain.invoke({
        "book_summary": tree["book"] or "(not available)",
        "chapter_summaries": joined,
        "question": question,
        "history_block": _history_block(history),
    })


def _answer_chapter(route, question: str, search_q: str, book_type: str,
                    history: Optional[List[dict]]) -> str:
    """Chapter summary for shape + passages from that chapter only for detail."""
    idx = BookIndex.load(book_type)
    if idx is None:
        return "No book has been indexed for this mode yet. Please upload a PDF first."

    chapter = idx.find_chapter(route.chapter) if route.chapter else None
    if chapter is None:
        return _answer_semantic(search_q, book_type, history,
                                k=config.RETRIEVAL_K, question=question)

    try:
        llm = get_llm()
    except LLMNotConfigured as e:
        return f"LLM not configured: {e}"

    db = get_vector_db(book_type)
    docs = db.similarity_search(search_q, k=config.RETRIEVAL_K, filter={
        "$and": [{"chapter_id": {"$eq": chapter["id"]}},
                 {"is_index": {"$eq": False}}]})

    summary = summarize.chapter_summary(book_type, chapter["id"]) or "(not available)"
    pages = [p.printed_page for p in idx.chapter_pages(chapter["id"]) if p.lines]

    if config.DEBUG_LOG:
        print(f"[chapter] {chapter['title'][:50]} -> {len(docs)} passages")

    chain = ChatPromptTemplate.from_template(CHAPTER_TEMPLATE) | llm | StrOutputParser()
    rate_limit.throttle("chapter answer")
    return chain.invoke({
        "chapter_title": chapter["title"],
        "chapter_pages": f"{pages[0]}-{pages[-1]}" if pages else "?",
        "chapter_summary": summary,
        "context": _format_docs(docs),
        "question": question,
        "history_block": _history_block(history),
    })
