"""
Page and line extraction - the foundation of the "Absolute Precision" fix.

The core problem: a PDF's internal page index is NOT the page number printed on
the paper. Front matter, roman numerals and cover pages push them out of sync.
A reader asking "what is on page 34?" means the printed folio, not pages[34].

This module resolves the printed page number three independent ways, records
which one it used, and stores every page's lines under BOTH numbering systems.
That index is what makes an exact location query a deterministic lookup instead
of a similarity search that hopes for the best.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import fitz  # PyMuPDF

import config

ARABIC_RE = re.compile(r"^\s*(\d{1,4})\s*$")
ROMAN_RE = re.compile(r"^\s*([ivxlcdm]{1,8})\s*$", re.I)

# Back-of-book index rows look like "while statement, 71" or "term, 71, 194".
# Dense keyword lists: they match almost any technical query while being useless
# as an answer - the worst combination for retrieval.
INDEX_ENTRY_RE = re.compile(
    r"^.{2,70}?,\s*\d{1,4}(\s*[-\u2013]\s*\d{1,4})?(\s*,\s*\d{1,4})*\s*$")


def _index_ratio(lines: List[str]) -> float:
    """Fraction of a page's lines shaped like index entries."""
    if len(lines) < 8:
        return 0.0
    return sum(1 for l in lines if INDEX_ENTRY_RE.match(l.strip())) / len(lines)


# ---------------------------------------------------------------- data model
@dataclass
class PageRecord:
    pdf_index: int                    # 0-based position in the PDF file
    printed_page: str                 # what is printed on the page ("34", "iv")
    lines: List[str] = field(default_factory=list)   # body text, header/footer removed
    header: Optional[str] = None
    footer: Optional[str] = None
    folio: Optional[str] = None       # the number actually found in the margin
    is_front_matter: bool = False
    is_index: bool = False            # back-of-book index page
    chapter_id: int = -1              # index into BookIndex.chapters, -1 = none
    chapter_title: Optional[str] = None
    source: str = "text"              # "text" | "ocr" | "vision"

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def line(self, n: int) -> Optional[str]:
        """1-based line lookup."""
        if 1 <= n <= len(self.lines):
            return self.lines[n - 1]
        return None


@dataclass
class BookIndex:
    book_type: str
    filename: str
    page_count: int
    label_method: str                 # how printed pages were resolved
    offset: int                       # printed_page + offset == pdf_index (body only)
    pages: List[PageRecord]
    margin_note: str = "defaults"     # how the header/footer bands were derived
    chapters: List[dict] = field(default_factory=list)   # {id,title,start,end}
    chapter_method: str = "none"      # pdf_outline | heading_regex | none

    # ---- lookups -------------------------------------------------------
    def by_printed(self, printed) -> Optional[PageRecord]:
        key = str(printed).strip().lower()
        for p in self.pages:
            if p.printed_page.lower() == key:
                return p
        return None

    def by_index(self, idx: int) -> Optional[PageRecord]:
        if 0 <= idx < len(self.pages):
            return self.pages[idx]
        return None

    def line_at(self, printed, line_no: int) -> Optional[str]:
        page = self.by_printed(printed)
        return page.line(line_no) if page else None

    def body_pages(self) -> List[PageRecord]:
        return [p for p in self.pages if not p.is_front_matter]

    def chapter_pages(self, chapter_id: int) -> List[PageRecord]:
        return [p for p in self.pages if p.chapter_id == chapter_id]

    def find_chapter(self, needle: str) -> Optional[dict]:
        """Match 'IV', '4', 'Iteration' against chapter titles."""
        n = str(needle).strip().lower()
        for c in self.chapters:
            t = c["title"].lower()
            if n and (f"chapter {n}." in t or f"chapter {n} " in t
                      or t.startswith(f"chapter {n}") or n in t):
                return c
        # roman <-> arabic
        romans = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
                  "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii",
                  "xix", "xx", "xxi", "xxii", "xxiii", "xxiv", "xxv", "xxvi",
                  "xxvii", "xxviii", "xxix", "xxx"]
        if n.isdigit() and 1 <= int(n) <= len(romans):
            return self.find_chapter(romans[int(n) - 1])
        if n in romans:
            return self.find_chapter(str(romans.index(n) + 1))
        return None

    def chapter_text(self, chapter_id: int) -> str:
        return "\n".join(p.text for p in self.chapter_pages(chapter_id) if p.lines)

    # ---- persistence ---------------------------------------------------
    def path(self) -> str:
        return os.path.join(config.INDEX_DIR, f"{self.book_type}.json")

    def save(self) -> str:
        data = asdict(self)
        with open(self.path(), "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        return self.path()

    @staticmethod
    def load(book_type: str) -> Optional["BookIndex"]:
        p = os.path.join(config.INDEX_DIR, f"{book_type}.json")
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as fh:
            raw = json.load(fh)
        raw["pages"] = [PageRecord(**pg) for pg in raw["pages"]]
        return BookIndex(**raw)


# ---------------------------------------------------------------- extraction
def _lines_with_geometry(page) -> List[dict]:
    """Return rendered lines as {text, y0, y1, x0} sorted in reading order."""
    out = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        if block.get("type") != 0:          # 0 = text block
            continue
        for ln in block.get("lines", []):
            text = "".join(sp.get("text", "") for sp in ln.get("spans", []))
            text = text.replace("\xa0", " ").rstrip()
            if not text.strip():
                continue
            x0, y0, x1, y1 = ln["bbox"]
            out.append({"text": text, "y0": y0, "y1": y1, "x0": x0})
    out.sort(key=lambda d: (round(d["y0"], 1), d["x0"]))
    return out


def _split_margins(lines: List[dict], height: float,
                   top_cut_frac: float, bot_cut_frac: float):
    """Separate running head / folio from body text using vertical position."""
    top_cut = height * top_cut_frac
    bot_cut = height * bot_cut_frac

    header, footer, body = None, None, []
    for ln in lines:
        if ln["y1"] <= top_cut:
            header = ln["text"].strip() if header is None else header
        elif ln["y0"] >= bot_cut:
            footer = ln["text"].strip() if footer is None else footer
        else:
            body.append(ln["text"])
    return header, footer, body


def _calibrate_margins(pages_lines: List[List[dict]], heights: List[float]):
    """Work out where this book actually prints its folio, instead of assuming.

    A fixed "top 8% / bottom 8%" rule is a guess about one publisher's layout.
    Think Python puts the folio in the running head; Dracula puts it in the
    footer at 93.6% of page height - just outside a naive 92% cutoff, so the
    number leaks into the body and every line number shifts by one.

    So: find every line that is a bare folio, see where those lines actually
    sit, and set the margin band to just clear them.

    Use the MEDIAN, never the max. A near-empty page whose last line happens to
    be a bare number in the upper half will drag a max-based cutoff halfway down
    the page and silently delete half the book. The hard clamps below are a
    second safety net: a margin band has no business eating 20% of a page.
    """
    tops, bots = [], []
    seen = 0
    for lines, h in zip(pages_lines, heights):
        if not lines or not h:
            continue
        seen += 1
        edge = list(lines[:2]) + list(lines[-2:])   # folio may sit beside a running head
        for ln in edge:
            t = ln["text"].strip()
            if ARABIC_RE.match(t) or (ROMAN_RE.match(t) and len(t) <= 8):
                rel0, rel1 = ln["y0"] / h, ln["y1"] / h
                (tops if rel0 < 0.5 else bots).append((rel0, rel1))

    top_cut, bot_cut = config.HEADER_ZONE, config.FOOTER_ZONE
    note = "defaults"

    MAX_HEADER, MIN_FOOTER = 0.20, 0.80    # hard clamps

    if seen and len(tops) / seen >= 0.25:
        med = statistics.median(r1 for _, r1 in tops)
        cand = min(med + 0.005, MAX_HEADER)
        if cand > top_cut:
            top_cut = cand
            note = "calibrated (folio in header)"

    if seen and len(bots) / seen >= 0.25:
        med = statistics.median(r0 for r0, _ in bots)
        cand = max(med - 0.005, MIN_FOOTER)
        if cand < bot_cut:
            bot_cut = cand
            note = ("calibrated (folio in footer)" if note == "defaults"
                    else "calibrated (folio top and bottom)")

    return top_cut, bot_cut, note


def _folio_from(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = text.strip()
    if ARABIC_RE.match(t):
        return str(int(t))
    if ROMAN_RE.match(t) and len(t) <= 8:
        return t.lower()
    # folio embedded in a running head, e.g. "34   DRACULA"
    m = re.match(r"^(\d{1,4})\b", t) or re.search(r"\b(\d{1,4})$", t)
    if m:
        return str(int(m.group(1)))
    return None


def _labels_from_catalog(path: str) -> Optional[List[str]]:
    """Preferred method: read /PageLabels straight out of the PDF catalog.

    Careful: pypdf's ``page_labels`` SYNTHESISES sequential arabic labels when
    the catalog has no /PageLabels entry, so it never returns empty. Trusting it
    blindly means a PDF with no labels silently reports printed page == index+1,
    which is exactly the bug this whole module exists to prevent. Check the
    catalog key itself before believing the labels.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        root = reader.trailer.get("/Root", {})
        if "/PageLabels" not in root:
            return None
        labels = list(reader.page_labels)
        if labels and any(str(l).strip() for l in labels):
            return [str(l).strip() for l in labels]
    except Exception:
        pass
    return None


CHAPTER_HEADING_RE = re.compile(
    r"^\s*(chapter|part|section)\s+([0-9]{1,3}|[ivxlcdm]{1,8})\b", re.I)


def _chapters_from_outline(doc, total: int) -> List[dict]:
    """Preferred: read the PDF's own bookmarks. Deterministic, no guessing."""
    try:
        toc = doc.get_toc()
    except Exception:
        return []
    if not toc:
        return []

    marks = []
    for entry in toc:
        level, title, page1 = entry[0], entry[1], entry[2]
        if level != 1 or page1 < 1:
            continue
        if not CHAPTER_HEADING_RE.match(title) and not title.lower().startswith("note"):
            continue                      # skip Title Page / Contents / Index
        marks.append((title.strip(), page1 - 1))

    chapters = []
    for i, (title, start) in enumerate(marks):
        end = marks[i + 1][1] - 1 if i + 1 < len(marks) else total - 1
        chapters.append({"id": i, "title": title, "start": start, "end": end})
    return chapters


def _chapters_from_headings(raw_pages) -> List[dict]:
    """Fallback: a page whose first lines announce a chapter starts one."""
    marks = []
    for p in raw_pages:
        for ln in p["lines"][:3]:
            if CHAPTER_HEADING_RE.match(ln.strip()):
                marks.append((ln.strip()[:80], p["pdf_index"]))
                break
    chapters = []
    for i, (title, start) in enumerate(marks):
        end = marks[i + 1][1] - 1 if i + 1 < len(marks) else raw_pages[-1]["pdf_index"]
        chapters.append({"id": i, "title": title, "start": start, "end": end})
    return chapters


def build_index_from_vision(file_path: str, book_type: str, result: dict) -> BookIndex:
    """Build the page index for a comic from vision output instead of a text layer.

    Comics have no embedded text and no printed folios, so:
      * printed page number == sequential position (documented, not guessed)
      * every "line" is one extracted text item, which means page+line queries
        work on the comic exactly as they do on the two text books
    """
    hashes, cache = result["hashes"], result["cache"]
    pages = []
    for idx in sorted(hashes):
        entry = cache.get(hashes[idx], {})
        items = entry.get("items", [])
        pages.append(PageRecord(
            pdf_index=idx,
            printed_page=str(idx + 1),
            lines=page_lines_for(items),
            header=None,
            footer=None,
            folio=None,
            is_front_matter=False,
            is_index=False,
            source="vision",
        ))

    idx_obj = BookIndex(
        book_type=book_type,
        filename=os.path.basename(file_path),
        page_count=len(pages),
        label_method="sequential_image_pages",
        offset=0,
        pages=pages,
        margin_note="n/a (image pages)",
        chapters=[],
        chapter_method="none",
    )
    idx_obj.save()
    return idx_obj


def page_lines_for(items):
    import vision
    return vision.page_lines(items, include_sfx=True)


def build_index(file_path: str, book_type: str = "coding") -> BookIndex:
    """Extract every page's lines and resolve its printed page number."""
    doc = fitz.open(file_path)
    total = len(doc)
    if config.MAX_PAGES:
        total = min(total, config.MAX_PAGES)

    # pass 1: pull every line with its geometry, strip nothing yet
    pages_lines, heights = [], []
    for i in range(total):
        page = doc[i]
        heights.append(page.rect.height)
        pages_lines.append(_lines_with_geometry(page))

    outline_chapters = _chapters_from_outline(doc, total)
    doc.close()

    # calibrate the margin bands from where this book prints its folio
    top_cut, bot_cut, margin_note = _calibrate_margins(pages_lines, heights)

    # pass 2: split header / footer / body using the calibrated bands
    raw_pages = []
    for i, (lines, height) in enumerate(zip(pages_lines, heights)):
        header, footer, body = _split_margins(lines, height, top_cut, bot_cut)
        folio = _folio_from(footer) or _folio_from(header)
        raw_pages.append({
            "pdf_index": i, "lines": body, "header": header,
            "footer": footer, "folio": folio,
        })

    # ---- resolve printed page numbers -------------------------------
    catalog = _labels_from_catalog(file_path)
    method, offset = "fallback_sequential", 0

    if catalog and len(catalog) >= total:
        method = "pdf_page_labels"
        printed = catalog[:total]
        # offset is defined as: pdf_index (0-based) == printed_page + offset
        first_arabic = next((i for i, l in enumerate(printed) if l.isdigit()), None)
        offset = (first_arabic - int(printed[first_arabic])
                  if first_arabic is not None else 0)
    else:
        # derive a constant offset from whatever folios we could read
        deltas = Counter()
        for p in raw_pages:
            f = p["folio"]
            if f and f.isdigit():
                deltas[p["pdf_index"] - int(f)] += 1
        agreement = 0.0
        if deltas:
            best, count = deltas.most_common(1)[0]
            agreement = count / max(1, total)
            if agreement >= config.MIN_FOLIO_AGREEMENT:
                method, offset = "footer_regex_offset", best
        printed = []
        for p in raw_pages:
            if method == "footer_regex_offset":
                if p["folio"]:
                    printed.append(p["folio"])
                else:
                    n = p["pdf_index"] - offset
                    printed.append(str(n) if n >= 1 else f"[{p['pdf_index'] + 1}]")
            else:
                printed.append(str(p["pdf_index"] + 1))

    pages = []
    for p, label in zip(raw_pages, printed):
        pages.append(PageRecord(
            pdf_index=p["pdf_index"],
            printed_page=str(label),
            lines=p["lines"],
            header=p["header"],
            footer=p["footer"],
            folio=p["folio"],
            is_front_matter=not str(label).isdigit(),
            is_index=_index_ratio(p["lines"]) >= 0.40,
            source="text",
        ))

    # ---- chapters: outline first, heading regex as fallback ----------
    if outline_chapters:
        chapters, chapter_method = outline_chapters, "pdf_outline"
    else:
        chapters = _chapters_from_headings(raw_pages)
        chapter_method = "heading_regex" if chapters else "none"

    for c in chapters:
        for p in pages[c["start"]:c["end"] + 1]:
            p.chapter_id = c["id"]
            p.chapter_title = c["title"]

    idx = BookIndex(
        book_type=book_type,
        filename=os.path.basename(file_path),
        page_count=len(pages),
        label_method=method,
        offset=offset,
        pages=pages,
        margin_note=margin_note,
        chapters=chapters,
        chapter_method=chapter_method,
    )
    idx.save()
    return idx


def index_report(idx: BookIndex) -> str:
    """Human-readable summary - print this after ingest and during the demo."""
    body = idx.body_pages()
    lc = [len(p.lines) for p in body] or [0]
    rows = [
        f"  file            : {idx.filename}",
        f"  pages           : {idx.page_count} "
        f"({idx.page_count - len(body)} front matter, {len(body)} body)",
        f"  page numbering  : {idx.label_method}",
        f"  margin bands    : {idx.margin_note}",
        f"  chapters        : {len(idx.chapters)} ({idx.chapter_method})",
        f"  offset          : printed page N  ->  pdf index N+{idx.offset} (0-based)",
    ]
    if body:
        f = body[0]
        rows.append(f"  first body page : printed '{f.printed_page}' "
                    f"at pdf index {f.pdf_index}")
    rows.append(f"  lines/page      : min={min(lc)} max={max(lc)} "
                f"avg={sum(lc) // len(lc)}")
    return "\n".join(rows)
