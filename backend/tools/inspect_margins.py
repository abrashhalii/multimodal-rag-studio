"""Show exactly where a PDF prints its running head and folio.

    python tools/inspect_margins.py "book.pdf" [pdf_page_index]

Use this when the smoke test reports a folio leaking into the body text, or
when page numbering resolves to fallback_sequential. It prints the vertical
position of the top and bottom lines as a fraction of page height, which is
what the margin calibration keys off.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import fitz
import page_index

if len(sys.argv) < 2:
    raise SystemExit(__doc__)
path = sys.argv[1]
probe = int(sys.argv[2]) if len(sys.argv) > 2 else None

doc = fitz.open(path)
pages_lines, heights = [], []
for i in range(len(doc)):
    heights.append(doc[i].rect.height)
    pages_lines.append(page_index._lines_with_geometry(doc[i]))

top, bot, note = page_index._calibrate_margins(pages_lines, heights)
print(f"\npages          : {len(doc)}")
print(f"calibration    : {note}")
print(f"header cut     : {top:.4f} of page height")
print(f"footer cut     : {bot:.4f} of page height")

targets = [probe] if probe is not None else [len(doc)//4, len(doc)//2, 3*len(doc)//4]
for idx in targets:
    lines, h = pages_lines[idx], heights[idx]
    print(f"\n--- pdf index {idx} (height {h:.0f}) ---")
    for ln in lines[:3]:
        zone = "HEADER" if ln["y1"] <= h*top else ("FOOTER" if ln["y0"] >= h*bot else "body")
        print(f"  {ln['y0']/h:.3f}-{ln['y1']/h:.3f}  {zone:6}  {ln['text'][:56]!r}")
    print("   ...")
    for ln in lines[-3:]:
        zone = "HEADER" if ln["y1"] <= h*top else ("FOOTER" if ln["y0"] >= h*bot else "body")
        print(f"  {ln['y0']/h:.3f}-{ln['y1']/h:.3f}  {zone:6}  {ln['text'][:56]!r}")
doc.close()
