"""
Central configuration. Everything is read from the environment (.env) so that
nothing is hardcoded and the same code runs locally, in Docker, or on a server.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _get(name, default=None):
    v = os.getenv(name)
    return default if v is None or v == "" else v


def _int(name, default):
    try:
        return int(_get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name, default):
    try:
        return float(_get(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name, default=False):
    return str(_get(name, str(default))).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- storage
DB_DIR = str(BASE_DIR / _get("DB_DIR", "chroma_db"))
INDEX_DIR = str(BASE_DIR / _get("INDEX_DIR", "page_index"))
TEMP_DIR = str(BASE_DIR / _get("TEMP_DIR", "temp"))
CACHE_DIR = str(BASE_DIR / _get("CACHE_DIR", "cache"))

for _d in (DB_DIR, INDEX_DIR, TEMP_DIR, CACHE_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- embeddings
# bge-small-en-v1.5 has a 512-token window (MiniLM only has 256, which silently
# truncates any chunk longer than ~1000 characters) and scores materially better
# on retrieval benchmarks. Swap back to all-MiniLM-L6-v2 if you need the speed.
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DEVICE = _get("EMBEDDING_DEVICE", "cpu")      # "cuda" if you have VRAM free
EMBEDDING_BATCH_SIZE = _int("EMBEDDING_BATCH_SIZE", 64)

# ---------------------------------------------------------------- chunking
CHUNK_CHARS = _int("CHUNK_CHARS", 900)          # stays inside the 512-token window
CHUNK_OVERLAP_LINES = _int("CHUNK_OVERLAP_LINES", 2)
PAGE_BOUNDED_CHUNKS = _bool("PAGE_BOUNDED_CHUNKS", True)

# ---------------------------------------------------------------- page extraction
HEADER_ZONE = _float("HEADER_ZONE", 0.08)       # top 8% of the page
FOOTER_ZONE = _float("FOOTER_ZONE", 0.92)       # bottom 8% of the page
MIN_FOLIO_AGREEMENT = _float("MIN_FOLIO_AGREEMENT", 0.30)
MAX_PAGES = _int("MAX_PAGES", 0)                # 0 = no limit; useful for smoke tests

# ---------------------------------------------------------------- retrieval
RETRIEVAL_K = _int("RETRIEVAL_K", 12)
# Fuse BM25 with dense retrieval. Dense-only measured hit@5 = 0.58 on verbatim
# phrase recall, with failures concentrated in the novel's homogeneous prose.
HYBRID_RETRIEVAL = _bool("HYBRID_RETRIEVAL", True)
CANDIDATE_K = _int("CANDIDATE_K", 40)           # pre-rerank pool (used from Day 4)
GLOBAL_K = _int("GLOBAL_K", 120)

# ---------------------------------------------------------------- LLM
LLM_PROVIDER = _get("LLM_PROVIDER", "gemini").lower()   # "gemini" | "ollama"

GEMINI_API_KEY = _get("GEMINI_API_KEY", "")
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FAST_MODEL = _get("GEMINI_FAST_MODEL", "gemini-2.5-flash-lite")

OLLAMA_MODEL = _get("OLLAMA_MODEL", "llama3.2")
OLLAMA_FAST_MODEL = _get("OLLAMA_FAST_MODEL", "llama3.2")
OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "http://localhost:11434")

LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.1)
LLM_MAX_TOKENS = _int("LLM_MAX_TOKENS", 2048)

# Free-tier ceiling for the chosen model. Everything outbound is paced to this.
LLM_RPM = _int("LLM_RPM", 15)
RATE_LIMIT_SAFETY = _float("RATE_LIMIT_SAFETY", 0.7)   # run at 70% of the cap

# ---------------------------------------------------------------- summarisation
SUMMARY_CHAPTER_WORDS = _int("SUMMARY_CHAPTER_WORDS", 220)
SUMMARY_BOOK_WORDS = _int("SUMMARY_BOOK_WORDS", 500)
SUMMARY_MAX_CHAPTER_CHARS = _int("SUMMARY_MAX_CHAPTER_CHARS", 60000)
BUILD_SUMMARIES_ON_INGEST = _bool("BUILD_SUMMARIES_ON_INGEST", False)

# ---------------------------------------------------------------- vision (comics)
# Pages per vision request. The binding free-tier constraint is REQUESTS per
# minute, not tokens - a downscaled page is ~1.5K tokens against a 250K/min
# ceiling - so batching cuts request count ~4x at no practical token cost.
VISION_BATCH_PAGES = _int("VISION_BATCH_PAGES", 4)
VISION_IMAGE_MAX_EDGE = _int("VISION_IMAGE_MAX_EDGE", 1400)
VISION_JPEG_QUALITY = _int("VISION_JPEG_QUALITY", 80)
COMIC_READING_ORDER = _get("COMIC_READING_ORDER", "ltr")   # ltr | rtl
# Comma-separated cast list. Comics rarely name a speaker in the panel, so a
# roster turns "unknown" into a real attribution - it is corpus metadata, not
# a licence to guess: the prompt still requires visual evidence.
COMIC_CHARACTERS = _get("COMIC_CHARACTERS", "")
INDEX_SFX = _bool("INDEX_SFX", False)   # keep sound effects out of embeddings

# ---------------------------------------------------------------- memory
HISTORY_TURNS = _int("HISTORY_TURNS", 6)
REWRITE_FOLLOWUPS = _bool("REWRITE_FOLLOWUPS", True)

# ---------------------------------------------------------------- server
HOST = _get("HOST", "0.0.0.0")
PORT = _int("PORT", 8000)
DEBUG_LOG = _bool("DEBUG_LOG", True)

BOOK_TYPES = ("coding", "novel", "manga")


def collection_name(book_type: str) -> str:
    """One Chroma collection per book type - never mix corpora."""
    bt = (book_type or "coding").lower()
    if bt not in BOOK_TYPES:
        bt = "coding"
    return f"book_{bt}"
