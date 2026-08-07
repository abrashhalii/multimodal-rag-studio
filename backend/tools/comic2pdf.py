#!/usr/bin/env python3
"""
comic2pdf.py - turn CBZ/CBR comic archives into a single page-aligned PDF.

Why this is not just "zip the images into a PDF":

1. DOUBLE-PAGE SPREADS.
   Comic releases store spreads as ONE wide image named like "012_013.jpg".
   Treat it as one page and every page number after it is wrong. This script
   detects spreads by aspect ratio and splits them, with a small overlap so a
   speech balloon sitting across the gutter is captured in both halves rather
   than sliced in two.

2. COVERS ARE NOT STORY PAGES.
   Skipping them makes PDF page N line up exactly with comic page N, so
   "what happens on page 12" has an unambiguous answer.

3. SIZE. Full-res scans are ~2 MB/page. Vision models read fine at ~1500 px on
   the long edge, and smaller images mean fewer tokens and faster ingestion.

Usage
-----
  # one issue
  python comic2pdf.py "014 X-Men Alpha 01.cbz" -o alpha.pdf

  # combine several, in the order given
  python comic2pdf.py "014 ....cbz" "015 ....cbr" "027 ....cbr" -o volume.pdf

  # whole folder, sorted by the leading number in the filename
  python comic2pdf.py --dir "D:\\Comics\\AoA" -o volume.pdf

  # archival quality instead of pipeline-optimised
  python comic2pdf.py "014 ....cbz" -o alpha_full.pdf --max-edge 0 --quality 95

Requirements
------------
  pip install pillow
  CBR (RAR) support also needs:  pip install rarfile   + the `unrar` executable
  CBZ (ZIP) needs nothing extra.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import zipfile
from typing import List, Optional, Tuple

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
COVER_RE = re.compile(r"(^|[^0-9])0{2,}[a-z]?\b|cover|wraparound", re.I)
PAGENUM_RE = re.compile(r"(\d{1,4})(?:[_-](\d{1,4}))?\s*(?:\.[a-z]+)?$", re.I)


# ---------------------------------------------------------------- archives
def read_archive(path: str) -> List[Tuple[str, bytes]]:
    """Return [(member_name, bytes), ...] for images, in sorted order."""
    ext = os.path.splitext(path)[1].lower()

    if ext in (".cbz", ".zip"):
        opener = zipfile.ZipFile(path)
        names = opener.namelist()
        reader = opener.read
    elif ext in (".cbr", ".rar"):
        try:
            import rarfile
        except ImportError:
            raise SystemExit(
                f"\n{os.path.basename(path)} is a RAR archive.\n"
                "  pip install rarfile\n"
                "and install the 'unrar' executable (Windows: put UnRAR.exe on PATH).\n"
                "Tip: many .cbr files are secretly ZIPs - try renaming to .cbz first.\n"
            )
        opener = rarfile.RarFile(path)
        names = opener.namelist()
        reader = opener.read
    else:
        raise SystemExit(f"Unsupported archive type: {path}")

    imgs = [n for n in names if n.lower().endswith(IMAGE_EXT)]
    imgs.sort(key=lambda n: n.lower())
    return [(n, reader(n)) for n in imgs]


def looks_like_cover(member: str) -> bool:
    stem = os.path.splitext(os.path.basename(member))[0]
    tail = stem.split("-")[-1].strip()
    if re.fullmatch(r"0+[a-z]?(\s*\(.*\))?", tail, re.I):
        return True
    return bool(re.search(r"cover|wraparound", stem, re.I))


def declared_pages(member: str) -> Optional[Tuple[int, Optional[int]]]:
    """Pull the page number(s) encoded in the filename: '012_013' -> (12, 13).

    Careful with separators. Releases name pages as "<Title> NN-PPP.jpg" where
    NN is the ISSUE number, and spreads as "PPP_PPP". So only an UNDERSCORE
    denotes a spread - treating '-' as one makes "Alpha 01-005" parse as
    "pages 1 to 5", which silently destroys the whole page mapping.
    """
    stem = os.path.splitext(os.path.basename(member))[0]
    stem = re.sub(r"\s*\(.*?\)\s*$", "", stem).strip()

    m = re.search(r"(\d{1,4})_(\d{1,4})$", stem)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        # Releases sometimes typo the second half ("012_003" for pages 12-13).
        # A right-hand page is always left+1, so repair it rather than trust it.
        if b != a + 1:
            b = a + 1
        return (a, b)

    m = re.search(r"(\d{1,4})$", stem)
    return (int(m.group(1)), None) if m else None


# ---------------------------------------------------------------- pages
def split_spread(img: Image.Image, overlap_pct: float) -> List[Image.Image]:
    """Left half, right half - western reading order, with gutter overlap."""
    w, h = img.size
    mid = w // 2
    pad = int(w * overlap_pct)
    left = img.crop((0, 0, min(w, mid + pad), h))
    right = img.crop((max(0, mid - pad), 0, w, h))
    return [left, right]


def is_spread(img: Image.Image, member: str) -> bool:
    w, h = img.size
    if w <= h:
        return False
    dp = declared_pages(member)
    if dp and dp[1] is not None:
        return True
    return (w / h) > 1.15        # landscape enough to be two portrait pages


def prepare(img: Image.Image, max_edge: int) -> Image.Image:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if max_edge and max(img.size) > max_edge:
        scale = max_edge / max(img.size)
        img = img.resize((max(1, int(img.width * scale)),
                          max(1, int(img.height * scale))),
                         Image.LANCZOS)
    return img


# ---------------------------------------------------------------- main
def convert(inputs: List[str], out_pdf: str, max_edge: int, quality: int,
            keep_covers: bool, split: bool, overlap: float) -> dict:
    pages, manifest = [], []
    page_no = 0

    for src in inputs:
        members = read_archive(src)
        issue = os.path.splitext(os.path.basename(src))[0]
        print(f"\n{issue}")
        print(f"  {len(members)} images")

        skipped = spreads = 0
        for member, blob in members:
            if not keep_covers and looks_like_cover(member):
                skipped += 1
                continue

            img = Image.open(io.BytesIO(blob))
            img.load()

            if split and is_spread(img, member):
                spreads += 1
                halves = split_spread(img, overlap)
                dp = declared_pages(member)
                labels = [dp[0], dp[1]] if dp and dp[1] else [None, None]
                for half, lab in zip(halves, labels):
                    page_no += 1
                    pages.append(prepare(half, max_edge))
                    manifest.append({
                        "pdf_page": page_no, "issue": issue,
                        "source_image": os.path.basename(member),
                        "from_spread": True, "declared_page": lab,
                    })
            else:
                page_no += 1
                dp = declared_pages(member)
                pages.append(prepare(img, max_edge))
                manifest.append({
                    "pdf_page": page_no, "issue": issue,
                    "source_image": os.path.basename(member),
                    "from_spread": False,
                    "declared_page": dp[0] if dp else None,
                })

        print(f"  covers skipped : {skipped}")
        print(f"  spreads split  : {spreads}")

    if not pages:
        raise SystemExit("No pages produced - check your input files.")

    print(f"\nWriting {len(pages)} pages -> {out_pdf}")
    pages[0].save(out_pdf, "PDF", save_all=True, append_images=pages[1:],
                  resolution=150.0, quality=quality, optimize=True)

    man_path = os.path.splitext(out_pdf)[0] + "_manifest.json"
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump({"output": os.path.basename(out_pdf),
                   "page_count": len(pages),
                   "inputs": [os.path.basename(i) for i in inputs],
                   "pages": manifest}, fh, indent=1)

    # does PDF page N line up with the page number in the filenames?
    aligned = sum(1 for m in manifest
                  if m["declared_page"] is not None
                  and m["declared_page"] == m["pdf_page"])
    checkable = sum(1 for m in manifest if m["declared_page"] is not None)

    size_mb = os.path.getsize(out_pdf) / 1e6
    print(f"  size           : {size_mb:.1f} MB")
    print(f"  manifest       : {os.path.basename(man_path)}")
    if checkable:
        pct = 100 * aligned / checkable
        print(f"  page alignment : {aligned}/{checkable} ({pct:.0f}%) "
              f"PDF page == filename page")
    return {"pages": len(pages), "size_mb": size_mb,
            "aligned": aligned, "checkable": checkable}


def main():
    ap = argparse.ArgumentParser(description="CBZ/CBR -> page-aligned PDF")
    ap.add_argument("inputs", nargs="*", help="CBZ/CBR files, in reading order")
    ap.add_argument("--dir", help="folder of CBZ/CBR, sorted by leading number")
    ap.add_argument("-o", "--out", default="comic.pdf")
    ap.add_argument("--max-edge", type=int, default=1600,
                    help="longest edge in px; 0 = keep original (default 1600)")
    ap.add_argument("--quality", type=int, default=82, help="JPEG quality")
    ap.add_argument("--keep-covers", action="store_true",
                    help="include covers (breaks page alignment)")
    ap.add_argument("--no-split", action="store_true",
                    help="leave double-page spreads as single wide pages")
    ap.add_argument("--overlap", type=float, default=0.03,
                    help="gutter overlap when splitting spreads (default 3%%)")
    a = ap.parse_args()

    inputs = list(a.inputs)
    if a.dir:
        found = [os.path.join(a.dir, f) for f in os.listdir(a.dir)
                 if f.lower().endswith((".cbz", ".cbr"))]
        found.sort(key=lambda p: (int(re.match(r"\s*(\d+)", os.path.basename(p)).group(1))
                                  if re.match(r"\s*(\d+)", os.path.basename(p)) else 10**9,
                                  os.path.basename(p).lower()))
        inputs += found
    if not inputs:
        ap.error("give at least one archive, or --dir")

    convert(inputs, a.out, a.max_edge, a.quality,
            a.keep_covers, not a.no_split, a.overlap)


if __name__ == "__main__":
    main()
