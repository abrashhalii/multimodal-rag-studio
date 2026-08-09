"""
Hybrid retrieval - lexical (BM25) fused with dense (embeddings).

WHY, MEASURED RATHER THAN ASSUMED
---------------------------------
The retrieval eval exposed the problem. Querying with a verbatim 14-word phrase
lifted straight off a page should find that page nearly every time. Dense-only
retrieval managed hit@5 = 0.58, and the failures were concentrated in the novel:

    manga   - strong (distinctive dialogue, proper nouns, all-caps)
    coding  - good   (code identifiers are lexically unique)
    novel   - weak   (1,203 chunks of homogeneous Victorian prose)

That pattern is the diagnosis. A bi-encoder maps text to a single vector, so
when every chunk is anxious first-person narration about illness and dread they
land close together and cosine similarity cannot separate them. Meanwhile the
exact phrase - the strongest possible signal - is invisible to it, because
embeddings encode meaning, not tokens.

BM25 is the complement: it scores rare terms highly and matches phrases exactly.
Proper nouns, code identifiers, and quoted dialogue are precisely what it is
good at, and precisely where dense retrieval blurs.

FUSION: RECIPROCAL RANK FUSION
------------------------------
The two retrievers produce incomparable scores - cosine distance and BM25
relevance are not on the same scale, and normalising them is fragile because the
ranges shift per query. RRF sidesteps that by using only RANK:

    score(doc) = sum over retrievers of  1 / (K + rank)

A document ranked well by either retriever scores well; one ranked well by both
scores best. No tuning, no normalisation, no assumption about score distribution.
K=60 is the value from the original RRF paper and is not sensitive.

Set HYBRID_RETRIEVAL=false in .env to fall back to dense-only.
"""

from __future__ import annotations

import re
from typing import List, Optional

from langchain_core.documents import Document

import config
from ingest import get_vector_db

RRF_K = 60
_cache: dict = {}

_TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall((text or "").lower())


def _doc_key(meta: dict) -> tuple:
    """Identity for fusion: a chunk is one page plus one line range."""
    return (str(meta.get("printed_page")),
            meta.get("line_start"), meta.get("line_end"))


def _corpus(book_type: str):
    """Load every chunk once and build the BM25 index in memory.

    A few thousand chunks tokenise in milliseconds, so there is no separate
    index file to persist, invalidate or keep in sync with Chroma - the vector
    store stays the single source of truth for the corpus.
    """
    if book_type in _cache:
        return _cache[book_type]

    db = get_vector_db(book_type)
    raw = db._collection.get(include=["documents", "metadatas"])
    docs, metas = [], []
    for text, meta in zip(raw.get("documents") or [], raw.get("metadatas") or []):
        if meta.get("is_index"):
            continue
        docs.append(text or "")
        metas.append(meta or {})

    bm25 = None
    if docs:
        try:
            from rank_bm25 import BM25Okapi
            bm25 = BM25Okapi([tokenize(d) for d in docs])
        except ImportError:
            if config.DEBUG_LOG:
                print("[hybrid] rank_bm25 not installed - dense only")

    _cache[book_type] = (docs, metas, bm25)
    return _cache[book_type]


def invalidate(book_type: Optional[str] = None) -> None:
    """Call after re-ingesting, or BM25 keeps scoring the previous corpus."""
    if book_type:
        _cache.pop(book_type, None)
    else:
        _cache.clear()


def _bm25_rank(book_type: str, query: str, n: int) -> List[Document]:
    docs, metas, bm25 = _corpus(book_type)
    if bm25 is None or not docs:
        return []
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
    return [Document(page_content=docs[i], metadata=metas[i])
            for i in order if scores[i] > 0]


def search(book_type: str, query: str, k: int,
           extra_filter: Optional[dict] = None) -> List[Document]:
    """Top-k documents. Hybrid when enabled and BM25 is available, else dense."""
    flt = {"is_index": {"$eq": False}}
    if extra_filter:
        flt = {"$and": [flt, extra_filter]}

    db = get_vector_db(book_type)

    if not config.HYBRID_RETRIEVAL:
        return db.similarity_search(query, k=k, filter=flt)

    pool = max(k, config.CANDIDATE_K)
    dense = db.similarity_search(query, k=pool, filter=flt)
    lexical = _bm25_rank(book_type, query, pool)
    if not lexical:
        return dense[:k]

    # Chapter-scoped queries filter on the dense side only, so apply the same
    # constraint to the lexical side rather than letting it leak past.
    if extra_filter:
        want = extra_filter.get("chapter_id", {}).get("$eq")
        if want is not None:
            lexical = [d for d in lexical if d.metadata.get("chapter_id") == want]

    scored, seen = {}, {}
    for rank, d in enumerate(dense, 1):
        key = _doc_key(d.metadata)
        scored[key] = scored.get(key, 0.0) + 1.0 / (RRF_K + rank)
        seen.setdefault(key, d)
    for rank, d in enumerate(lexical, 1):
        key = _doc_key(d.metadata)
        scored[key] = scored.get(key, 0.0) + 1.0 / (RRF_K + rank)
        seen.setdefault(key, d)

    best = sorted(scored, key=lambda kk: -scored[kk])[:k]

    if config.DEBUG_LOG:
        dkeys = {_doc_key(d.metadata) for d in dense[:k]}
        added = sum(1 for kk in best if kk not in dkeys)
        print(f"[hybrid] dense={len(dense)} bm25={len(lexical)} "
              f"-> top{k}, {added} promoted by lexical match")

    return [seen[kk] for kk in best]
