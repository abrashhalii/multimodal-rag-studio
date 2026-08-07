"""Pre-build the summary tree so the first global question is instant.

    python tools/build_summaries.py novel
    python tools/build_summaries.py novel --force

Run this BEFORE the demo. Building takes roughly (chapters / RPM) minutes -
about two minutes per book at 15 requests/minute - and you do not want that
happening while someone is watching. Summaries are cached by content hash, so
running it twice on an unchanged book costs zero API calls.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config, rate_limit, summarize
from page_index import BookIndex

if len(sys.argv) < 2:
    raise SystemExit(__doc__)
book_type = sys.argv[1]
force = "--force" in sys.argv

idx = BookIndex.load(book_type)
if idx is None:
    raise SystemExit(f"No index for '{book_type}'. Upload the PDF through the UI first.")
if not idx.chapters:
    raise SystemExit(f"No chapters detected in {idx.filename} "
                     f"(method: {idx.chapter_method}). Global questions will use wide retrieval.")

print(f"\n{idx.filename}: {len(idx.chapters)} chapters, {config.LLM_RPM} RPM budget")
print(f"estimated worst case: ~{len(idx.chapters) * 60 // max(1, config.LLM_RPM)}s\n")

t0 = time.monotonic()
res = summarize.build_summaries(idx, force=force)
elapsed = time.monotonic() - t0

print(f"\nchapters : {len(res['chapters'])}")
print(f"built    : {res['built']} (API calls made)")
print(f"reused   : {res['reused']} (from cache, free)")
print(f"elapsed  : {elapsed:.0f}s")
print(f"limiter  : {rate_limit.get_bucket().stats()}")
if res["book"]:
    print(f"\noverview ({len(res['book'].split())} words):\n{res['book'][:400]}...")
