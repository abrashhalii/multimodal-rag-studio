"""
Vision extraction for comics - the fix for Bug 4 (the 100-point item).

WHY BATCH PAGES PER REQUEST
---------------------------
The naive pipeline is one vision call per page. At 258 pages that is 258
requests against a 500/day free-tier cap, leaving almost nothing for testing,
and roughly 43 minutes of paced wall clock.

But the constraint that actually binds is REQUESTS per minute, not tokens: the
ceiling is 15 RPM against 250K tokens/minute, and a downscaled comic page is
only ~1.5K tokens. Sending four pages in one request costs four times the tokens
and ONE request - so the same work drops from 258 calls to 65, an 75% saving,
while still using under 10% of the token allowance.

Each image is labelled in the prompt and the model returns page-keyed JSON, so
attribution survives batching. Cache granularity stays per-page: a batch only
contains pages that are not already cached, so re-running after a prompt tweak
re-bills only what actually changed.

TEXT TYPES
----------
Comics carry four kinds of text and they are not interchangeable:
  speech  - dialogue in a balloon
  thought - internal, in a cloud balloon
  caption - narration boxes ("MEANWHILE, IN LATVERIA...")
  sfx     - sound effects (KRAKOOM, THWIP)

Sound effects are kept in the record but excluded from retrieval embeddings:
they are semantically noisy and match on phonetics rather than meaning.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
from typing import Dict, List, Optional

import fitz  # PyMuPDF

import config
import rate_limit

EXTRACT_PROMPT = """You are extracting text from comic book pages for a search index.

You are given {n} page images. Each is announced by a line "=== PAGE <id> ===" in
the order they appear.

For EVERY page, read EVERY piece of text in {order} reading order, top to bottom
within each tier of panels. Miss nothing - including text drawn as artwork.

TEXT TYPE - decide by the container, not the content:
  "caption" - text in a RECTANGULAR or irregular BOX with NO tail pointing at a
              character. Often coloured, often at a panel corner. Narration and
              scene-setting ("MEANWHILE...", "Sadly... this too will pass.").
              If it has no tail, it is a caption, not speech.
  "speech"  - text in a rounded balloon WITH a tail pointing to a speaker.
  "thought" - text in a cloud-shaped balloon, tail drawn as bubbles.
  "sfx"     - lettering drawn INTO the art rather than placed in a container:
              KRAKOOM, THWIP, NOOOO!, FWOOSH. Usually large, stylised, coloured
              or outlined. These are easy to overlook - look specifically for
              them on every page and include them.

SPEAKER - the tail decides, never the words:
  * Follow the balloon's TAIL to the figure it points at. That figure is the
    speaker. This is the only reliable evidence.
  * A name appearing INSIDE a balloon is almost always who is being ADDRESSED,
    not who is talking. "THERE SHE IS, LORD UNUS!" is spoken TO Unus by someone
    else - attributing it to Unus is wrong. Do not copy a name out of the text.
  * A name in one balloon can identify the figure it points AT, so a DIFFERENT
    balloon whose tail reaches that same figure may then be attributed to them.
  * Use the same name consistently across the page once you have established
    who a figure is.
  * Recurring narration in the same caption style is usually one narrator -
    name them if the text identifies them, otherwise "narrator".
{roster}
  * Only after all of that, use "unknown". A wrong name is worse than "unknown".
    Many balloons genuinely point off-panel or at unnamed background figures;
    "unknown" is the correct answer for those, not a failure.

Captions and sound effects have no speaker: leave it empty.

Return ONLY a JSON object, no markdown fences and no commentary, shaped exactly:

{{"pages": [
  {{"page": "<id>", "items": [
     {{"panel": 1, "type": "caption", "speaker": "", "text": "MEANWHILE..."}},
     {{"panel": 1, "type": "speech", "speaker": "Magneto", "text": "We must strike now."}},
     {{"panel": 2, "type": "sfx", "speaker": "", "text": "KRAKOOM"}}
  ]}}
]}}

Preserve the text as written, including ellipses and emphasis. If a page has no
text at all, return it with an empty items list. Never omit a page."""


# ---------------------------------------------------------------- images
def render_page(doc, index: int, max_edge: int, quality: int) -> bytes:
    """Render one PDF page to JPEG bytes at a size a vision model can read."""
    page = doc[index]
    rect = page.rect
    scale = max_edge / max(rect.width, rect.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img_bytes = pix.tobytes("png")

    from PIL import Image
    im = Image.open(io.BytesIO(img_bytes))
    if im.mode != "RGB":
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def image_hash(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------- cache
def _cache_path(book_type: str) -> str:
    return os.path.join(config.CACHE_DIR, f"vision_{book_type}.json")


def load_cache(book_type: str) -> dict:
    p = _cache_path(book_type)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            pass
    return {}


def save_cache(book_type: str, data: dict) -> None:
    with open(_cache_path(book_type), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)


# ---------------------------------------------------------------- model call
def _parse_json(raw: str) -> Optional[dict]:
    """Models wrap JSON in fences and prose no matter how firmly you ask."""
    if not raw:
        return None
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        pass
    start, end = txt.find("{"), txt.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(txt[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def extract_batch(images: List[tuple]) -> Dict[str, list]:
    """images = [(page_id, jpeg_bytes), ...]  ->  {page_id: [items]}"""
    from langchain_core.messages import HumanMessage
    from llm_provider import get_llm

    order = ("right-to-left (Japanese manga)"
             if config.COMIC_READING_ORDER == "rtl"
             else "left-to-right (Western comics)")

    roster = ""
    if config.COMIC_CHARACTERS:
        roster = ("  * Characters known to appear in this volume. Use these spellings when the\n"
                  "    FIGURE AT THE END OF THE TAIL matches one of them. Seeing a name\n"
                  "    in the dialogue is not evidence that they are speaking:\n    "
                  + config.COMIC_CHARACTERS)

    content = [{"type": "text",
                "text": EXTRACT_PROMPT.format(n=len(images), order=order,
                                              roster=roster)}]
    for page_id, blob in images:
        content.append({"type": "text", "text": f"=== PAGE {page_id} ==="})
        content.append({
            "type": "image_url",
            "image_url": "data:image/jpeg;base64,"
                         + base64.b64encode(blob).decode("ascii"),
        })

    llm = get_llm()
    msg = HumanMessage(content=content)

    def _call():
        resp = llm.invoke([msg])
        text = resp.content
        if isinstance(text, list):      # LangChain 1.x returns content blocks
            text = "".join(b.get("text", "") for b in text
                           if isinstance(b, dict))
        return text

    raw = rate_limit.retry_on_quota(
        _call, label=f"vision batch of {len(images)}")

    data = _parse_json(raw)
    out = {}
    if not data or "pages" not in data:
        if config.DEBUG_LOG:
            print(f"[vision] unparseable response ({len(raw or '')} chars)")
        return {pid: [] for pid, _ in images}

    for entry in data.get("pages", []):
        pid = str(entry.get("page", "")).strip()
        items = entry.get("items", []) or []
        clean = []
        for it in items:
            text = (it.get("text") or "").strip()
            if not text:
                continue
            # Balloons wrap text across several short lines. Keeping those
            # breaks fragments the phrase for both embedding and display, so
            # collapse them back into one line.
            text = re.sub(r"\s*\n\s*", " ", text)
            text = re.sub(r"[ \t]{2,}", " ", text).strip()
            ttype = (it.get("type") or "speech").strip().lower()
            if ttype not in ("speech", "thought", "caption", "sfx"):
                ttype = "speech"
            clean.append({
                "panel": it.get("panel", 1),
                "type": ttype,
                "speaker": (it.get("speaker") or "").strip(),
                "text": text,
            })
        out[pid] = clean

    for pid, _ in images:            # never silently lose a page
        out.setdefault(pid, [])
    return out


# ---------------------------------------------------------------- pipeline
def extract_pdf(file_path: str, book_type: str = "manga",
                force: bool = False) -> dict:
    """Run vision extraction over a comic PDF, batched, cached and resumable."""
    doc = fitz.open(file_path)
    total = len(doc)
    if config.MAX_PAGES:
        total = min(total, config.MAX_PAGES)

    cache = {} if force else load_cache(book_type)
    pending, rendered = [], {}
    reused = 0

    for i in range(total):
        blob = render_page(doc, i, config.VISION_IMAGE_MAX_EDGE,
                           config.VISION_JPEG_QUALITY)
        h = image_hash(blob)
        rendered[i] = h
        if h in cache:
            reused += 1
        else:
            pending.append((i, h, blob))

    doc.close()

    batch_size = max(1, config.VISION_BATCH_PAGES)
    calls = 0
    print(f"[vision] {total} pages: {reused} cached, {len(pending)} to extract "
          f"({(len(pending) + batch_size - 1) // batch_size} requests "
          f"at {batch_size} pages each)")

    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        images = [(str(idx + 1), blob) for idx, _, blob in chunk]
        result = extract_batch(images)
        calls += 1

        for idx, h, _ in chunk:
            items = result.get(str(idx + 1), [])
            cache[h] = {"pdf_index": idx, "items": items}
        # Persist every batch: a crash must not discard completed requests.
        save_cache(book_type, cache)

        got = sum(len(result.get(str(i + 1), [])) for i, _, _ in chunk)
        print(f"[vision] pages {chunk[0][0] + 1}-{chunk[-1][0] + 1}: "
              f"{got} text items")

    return {"total": total, "reused": reused, "extracted": len(pending),
            "requests": calls, "hashes": rendered, "cache": cache}


def page_lines(items: List[dict], include_sfx: bool = True) -> List[str]:
    """Turn extracted items into numbered lines - so page/line queries work here too."""
    lines = []
    for it in items:
        if it["type"] == "sfx" and not include_sfx:
            continue
        if it["type"] == "caption":
            lines.append(f"[caption] {it['text']}")
        elif it["type"] == "sfx":
            lines.append(f"[sfx] {it['text']}")
        elif it["type"] == "thought":
            who = it["speaker"] or "unknown"
            lines.append(f"{who} (thinking): {it['text']}")
        else:
            who = it["speaker"] or "unknown"
            lines.append(f"{who}: {it['text']}")
    return lines
