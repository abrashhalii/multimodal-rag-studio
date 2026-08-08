"""Run vision extraction over the comic and ingest it.

    python tools/build_manga.py "D:\\...\\Books\\AoA_Volume.pdf"
    python tools/build_manga.py "...pdf" --pages 8      # prompt-tuning slice
    python tools/build_manga.py "...pdf" --force        # ignore cache, re-extract

Every page is cached by the SHA-256 of its rendered image, and the cache is
written after every batch. Re-running costs API calls only for pages that are
genuinely new, so iterating on the prompt does not re-bill the whole volume.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config, rate_limit
from ingest import ingest_pdf
from page_index import BookIndex

if len(sys.argv) < 2:
    raise SystemExit(__doc__)
path = sys.argv[1]
force = "--force" in sys.argv
if "--pages" in sys.argv:
    config.MAX_PAGES = int(sys.argv[sys.argv.index("--pages") + 1])

if not os.path.exists(path):
    raise SystemExit(f"Not found: {path}")

import fitz
n = len(fitz.open(path))
todo = min(n, config.MAX_PAGES) if config.MAX_PAGES else n
batch = config.VISION_BATCH_PAGES
print(f"\n{os.path.basename(path)}: {n} pages, processing {todo}")
print(f"batch size {batch} -> at most {(todo + batch - 1) // batch} requests")
print(f"paced at {int(config.LLM_RPM * config.RATE_LIMIT_SAFETY)} RPM\n")

t0 = time.monotonic()
stats = ingest_pdf(path, "manga", force_vision=force)
elapsed = time.monotonic() - t0

idx = BookIndex.load("manga")
withtext = [p for p in idx.pages if p.lines]
lines = sum(len(p.lines) for p in idx.pages)
speakers = {}
for p in idx.pages:
    for l in p.lines:
        if ":" in l and not l.startswith("["):
            speakers[l.split(":")[0].strip()] = speakers.get(l.split(":")[0].strip(), 0) + 1

print(f"\npages        : {idx.page_count}")
print(f"with text    : {len(withtext)}")
print(f"text items   : {lines}")
print(f"chunks       : {stats['chunks']}")
print(f"elapsed      : {elapsed:.0f}s")
print(f"limiter      : {rate_limit.get_bucket().stats()}")
top = sorted(speakers.items(), key=lambda kv: -kv[1])[:10]
print(f"\ntop speakers : {', '.join(f'{k} ({v})' for k, v in top)}")
if withtext:
    p = withtext[len(withtext)//2]
    print(f"\nsample - page {p.printed_page}:")
    for i, l in enumerate(p.lines[:6], 1):
        print(f"  {i}: {l}")
