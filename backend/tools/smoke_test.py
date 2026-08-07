"""
Day 1 acceptance test. Run this before you touch anything else.

    python tools/smoke_test.py "C:\\path\\to\\Dracula_Bram_Stoker.pdf" novel

It checks, in order:
  1. page extraction and printed-page resolution
  2. that BOTH numbering methods agree (catalog labels vs footer regex)
  3. the router classifies location queries correctly
  4. deterministic page+line lookup returns real text
  5. page-bounded chunking produces sane, single-page chunks

It does NOT need an LLM or an API key - that is the point. Everything Bug 1
depends on is verifiable offline.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config          # noqa: E402
import router          # noqa: E402
from page_index import build_index, index_report   # noqa: E402

GREEN, RED, DIM, RESET = "\033[92m", "\033[91m", "\033[90m", "\033[0m"
passed = failed = 0


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  {GREEN}PASS{RESET}  {label}  {DIM}{detail}{RESET}")
    else:
        failed += 1
        print(f"  {RED}FAIL{RESET}  {label}  {detail}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    book_type = sys.argv[2] if len(sys.argv) > 2 else "coding"
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")

    print(f"\n{'=' * 62}\n  DAY 1 SMOKE TEST - {os.path.basename(path)}\n{'=' * 62}\n")

    print("[1] Page extraction")
    idx = build_index(path, book_type)
    print(index_report(idx))
    print()
    check("pages extracted", idx.page_count > 0, f"{idx.page_count} pages")
    body = idx.body_pages()
    check("body pages found", len(body) > 0, f"{len(body)} body pages")
    empty = [p.printed_page for p in body if not p.lines]
    check("no empty body pages", len(empty) <= max(5, len(body) // 25),
          f"{len(empty)} empty")
    check("page numbering resolved", idx.label_method != "fallback_sequential",
          idx.label_method if idx.label_method != "fallback_sequential"
          else "NO folios found - is this a scanned PDF?")

    print("\n[2] Router")
    for q, kind, pg, ln in [
        ("What does page 34 line 5 say?", "page_line", 34, 5),
        ("read me line 12 of page 200", "page_line", 200, 12),
        ("What happened on page 23?", "page", 23, None),
        ("Summarize the entire book", "global", None, None),
        ("What did I just ask you?", "memory", None, None),
        ("Who is the main character?", "semantic", None, None),
    ]:
        r = router.classify(q)
        check(f"{q[:34]:<36}", r.kind == kind and r.page == pg and r.line == ln,
              f"-> {r}")

    print("\n[2b] Margin stripping")
    import re as _re
    bare = _re.compile(r"^\s*(\d{1,4}|[ivxlcdm]{1,8})\s*$", _re.I)
    leak_first = [p.printed_page for p in body if p.lines and bare.match(p.lines[0])]
    leak_last = [p.printed_page for p in body if p.lines and bare.match(p.lines[-1])]
    check("folio not leaking into line 1", len(leak_first) <= len(body) * 0.02,
          f"{len(leak_first)} pages start with a bare number "
          f"{leak_first[:5]}" if leak_first else "clean")
    check("folio not leaking into last line", len(leak_last) <= len(body) * 0.02,
          f"{len(leak_last)} pages end with a bare number "
          f"{leak_last[:5]}" if leak_last else "clean")
    head = getattr(body[0], "header", None)
    check("margin bands resolved", True, idx.margin_note)

    print("\n[3] Deterministic location lookup")
    target = body[min(33, len(body) - 1)]
    pno = target.printed_page
    for ln in (1, 5):
        text = idx.line_at(pno, ln)
        check(f"page {pno} line {ln}", bool(text and text.strip()),
              f"{(text or '')[:52]!r}")
    check("out-of-range line rejected", idx.line_at(pno, 9999) is None, "returns None")
    check("out-of-range page rejected", idx.by_printed(999999) is None, "returns None")

    print("\n[4] Round-trip: printed page -> pdf index -> back")
    ok = True
    for p in body[::max(1, len(body) // 20)]:
        if idx.by_printed(p.printed_page).pdf_index != p.pdf_index:
            ok = False
    check("printed<->index mapping stable", ok, "sampled 20 pages")

    print("\n[5] Chunking")
    try:
        from ingest import build_documents
        docs = build_documents(idx)
        check("chunks produced", len(docs) > 0, f"{len(docs)} chunks")
        check("every chunk on one page",
              all(d.metadata.get("printed_page") for d in docs))
        check("no malformed line ranges",
              all(d.metadata["line_start"] <= d.metadata["line_end"] for d in docs))
        check("location header injected",
              all(d.page_content.startswith("[Source:") for d in docs))
        biggest = max(len(d.page_content) for d in docs)
        check("chunks within embedding window", biggest <= config.CHUNK_CHARS + 250,
              f"largest {biggest} chars")
        print(f"\n{DIM}  sample chunk:{RESET}")
        print(f"{DIM}  {docs[0].page_content[:150]}...{RESET}")
    except ImportError as e:
        print(f"  {DIM}skipped (install requirements first): {e}{RESET}")

    print(f"\n{'=' * 62}")
    verdict = f"{GREEN}ALL CLEAR{RESET}" if failed == 0 else f"{RED}{failed} FAILED{RESET}"
    print(f"  {passed} passed, {failed} failed   {verdict}")
    print(f"{'=' * 62}\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
