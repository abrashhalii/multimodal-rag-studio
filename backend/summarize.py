"""
Hierarchical summarisation - the fix for "Myopic Context" (Bug 2).

WHY NOT JUST RAISE k
--------------------
Top-k retrieval samples the book; it does not survey it. Asking "what is the
best exercise in this book?" with k=120 gives the model 120 semantically
similar chunks out of 663 - it has still never seen most of the book, and it
cannot rank what it has not read. No value of k fixes that, because the
operation is a survey, not a search.

THE SHAPE OF THE FIX
--------------------
Build a summary tree at ingest, bottom-up:

    chapter text  ->  chapter summary   (one call per chapter)
    all summaries ->  book summary      (one call)

Global questions are then answered from the tree, which is derived from EVERY
page, rather than from a top-k sample.

WHY CHAPTER LEVEL AND NOT PAGE LEVEL
------------------------------------
Budget. Free-tier Gemini allows 500 requests/day. Page-level summaries would
cost 244 calls for the textbook and 403 for the novel - 647 calls, over the
daily cap, for two books. Chapter level costs 28 + ~20 = ~48. Same structure,
one twentieth of the budget, and chapters are the unit readers actually ask
about anyway.

CACHING
-------
Each summary is keyed by the SHA-256 of its source text. Re-ingesting an
unchanged book costs zero API calls. Without this, every restart during
development would burn the daily quota.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
import rate_limit
from llm_provider import get_llm
from page_index import BookIndex

CHAPTER_PROMPT = """Summarise the following chapter from a book.

Write {words} words or fewer. Cover what actually happens or is taught, the key
terms, people or concepts introduced, and anything a reader would need in order to
recall this chapter later. Where the chapter contains exercises, worked examples or
notable passages, name them specifically.

Write plain prose. Do not add a preamble, a title, or commentary about the text.

Chapter: {title}
Printed pages: {pages}

--- CHAPTER TEXT ---
{text}
--- END ---

Summary:"""

BOOK_PROMPT = """Below are summaries of every chapter of a book, in order.

Using ALL of them, write a {words}-word overview of the whole book: what it covers,
how it is structured, how the parts connect, and what its main themes or arguments are.

Write plain prose. Do not add a preamble or commentary about the summaries themselves.

Book: {title}

--- CHAPTER SUMMARIES ---
{summaries}
--- END ---

Overview:"""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]


def _cache_path(book_type: str) -> str:
    return os.path.join(config.CACHE_DIR, f"summaries_{book_type}.json")


def _load_cache(book_type: str) -> dict:
    p = _cache_path(book_type)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {"chapters": {}, "book": {}}


def _save_cache(book_type: str, data: dict) -> None:
    with open(_cache_path(book_type), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)


def _summarise_chapter(title: str, pages: str, text: str) -> str:
    if len(text) > config.SUMMARY_MAX_CHAPTER_CHARS:
        half = config.SUMMARY_MAX_CHAPTER_CHARS // 2
        text = text[:half] + "\n[...]\n" + text[-half:]
    chain = ChatPromptTemplate.from_template(CHAPTER_PROMPT) | get_llm() | StrOutputParser()
    return rate_limit.retry_on_quota(
        lambda: chain.invoke({
            "words": config.SUMMARY_CHAPTER_WORDS,
            "title": title, "pages": pages, "text": text,
        }).strip(),
        label=f"chapter summary: {title[:36]}")


def build_summaries(idx: BookIndex, force: bool = False) -> dict:
    """Build (or reuse) the summary tree for one book. Safe to call repeatedly."""
    cache = {"chapters": {}, "book": {}} if force else _load_cache(idx.book_type)
    built = reused = 0

    if not idx.chapters:
        return {"chapters": [], "book": "", "built": 0, "reused": 0,
                "note": "no chapters detected in this document"}

    ordered = []
    for c in idx.chapters:
        text = idx.chapter_text(c["id"])
        if not text.strip():
            continue
        key = str(c["id"])
        h = _hash(text)
        cached = cache["chapters"].get(key)

        if cached and cached.get("hash") == h:
            summary = cached["summary"]
            reused += 1
        else:
            pages = _page_span(idx, c["id"])
            summary = _summarise_chapter(c["title"], pages, text)
            cache["chapters"][key] = {"hash": h, "title": c["title"],
                                      "pages": pages, "summary": summary}
            # Save after EVERY chapter. A crash on chapter 17 must not throw
            # away 16 completed API calls - at 500 requests/day that is a
            # meaningful fraction of the daily budget, and the first version
            # only saved at the end, so every failed run started from zero.
            _save_cache(idx.book_type, cache)
            built += 1
            if config.DEBUG_LOG:
                done = built + reused
                print(f"[summary] {done:>2}/{len(idx.chapters)} "
                      f"{c['title'][:44]:46} ({pages})")

        ordered.append({"id": c["id"], "title": c["title"],
                        "pages": cache["chapters"][key].get("pages", ""),
                        "summary": summary})

    joined = "\n\n".join(f"{o['title']} (pages {o['pages']}):\n{o['summary']}"
                         for o in ordered)
    bh = _hash(joined)
    if cache["book"].get("hash") == bh:
        book_summary = cache["book"]["summary"]
    else:
        chain = ChatPromptTemplate.from_template(BOOK_PROMPT) | get_llm() | StrOutputParser()
        book_summary = rate_limit.retry_on_quota(
            lambda: chain.invoke({
                "words": config.SUMMARY_BOOK_WORDS,
                "title": idx.filename, "summaries": joined,
            }).strip(),
            label="book summary")
        cache["book"] = {"hash": bh, "summary": book_summary}
        built += 1

    _save_cache(idx.book_type, cache)
    return {"chapters": ordered, "book": book_summary,
            "built": built, "reused": reused, "note": ""}


def _page_span(idx: BookIndex, chapter_id: int) -> str:
    pages = [p.printed_page for p in idx.chapter_pages(chapter_id) if p.lines]
    return f"{pages[0]}-{pages[-1]}" if pages else "?"


def load_summaries(book_type: str) -> Optional[dict]:
    """Read the tree without building. Returns None if it was never built."""
    cache = _load_cache(book_type)
    if not cache.get("chapters"):
        return None
    ordered = [{"id": int(k), "title": v.get("title", ""),
                "pages": v.get("pages", ""), "summary": v["summary"]}
               for k, v in sorted(cache["chapters"].items(), key=lambda kv: int(kv[0]))]
    return {"chapters": ordered, "book": cache.get("book", {}).get("summary", "")}


def chapter_summary(book_type: str, chapter_id: int) -> Optional[str]:
    cache = _load_cache(book_type)
    entry = cache["chapters"].get(str(chapter_id))
    return entry["summary"] if entry else None
