"""List the images inside a comic archive with their dimensions.

    python tools/inspect_cbr.py "path\\to\\issue.cbr"

Use when page alignment comes out wrong: it shows the naming convention and
which images are landscape (i.e. double-page spreads).
"""
import io, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import comic2pdf   # reuses its archive reader + UnRAR path setup

if len(sys.argv) < 2:
    raise SystemExit(__doc__)

for name, blob in comic2pdf.read_archive(sys.argv[1]):
    im = Image.open(io.BytesIO(blob))
    w, h = im.size
    base = os.path.basename(name)
    print(f"{base:52} {w}x{h}  ratio={w/h:.2f}"
          f"{'  <-- LANDSCAPE (spread?)' if w > h else ''}"
          f"{'  [cover]' if comic2pdf.looks_like_cover(name) else ''}"
          f"  declared={comic2pdf.declared_pages(name)}")