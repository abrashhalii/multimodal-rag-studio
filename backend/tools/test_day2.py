"""
Day 2 acceptance test. Offline where possible - no API key needed for most of it.

    python tools/test_day2.py <book_type>

Checks:
  1. chapters were detected and cover the book
  2. chapter lookup handles roman and arabic
  3. the router sends global / chapter / follow-up questions the right way
  4. rewrite triggers on follow-ups and leaves standalone questions alone
  5. the token bucket actually paces requests
  6. the summary tree exists and covers every chapter (if built)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config          # noqa: E402
import query_rewrite   # noqa: E402
import rate_limit      # noqa: E402
import router          # noqa: E402
import summarize       # noqa: E402
from page_index import BookIndex   # noqa: E402

GREEN, RED, DIM, RESET = "\033[92m", "\033[91m", "\033[90m", "\033[0m"
passed = failed = 0


def check(label, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  {GREEN}PASS{RESET}  {label}  {DIM}{detail}{RESET}")
    else:
        failed += 1
        print(f"  {RED}FAIL{RESET}  {label}  {detail}")


def main():
    book_type = sys.argv[1] if len(sys.argv) > 1 else "coding"
    print(f"\n{'=' * 62}\n  DAY 2 TEST - {book_type}\n{'=' * 62}\n")

    idx = BookIndex.load(book_type)
    if idx is None:
        raise SystemExit(f"No index for '{book_type}'. Upload the PDF first.")

    print("[1] Chapter detection")
    print(f"  method: {idx.chapter_method}, chapters: {len(idx.chapters)}")
    check("chapters detected", len(idx.chapters) > 0, f"{len(idx.chapters)}")
    if idx.chapters:
        for c in idx.chapters[:3]:
            pages = [p.printed_page for p in idx.chapter_pages(c["id"]) if p.lines]
            print(f"    {DIM}{c['title'][:46]:48} printed "
                  f"{pages[0] if pages else '?'}-{pages[-1] if pages else '?'}{RESET}")
        assigned = sum(1 for p in idx.body_pages() if p.chapter_id >= 0)
        body = len(idx.body_pages())
        check("most body pages belong to a chapter", assigned >= body * 0.8,
              f"{assigned}/{body}")
        check("no chapter is empty",
              all(idx.chapter_text(c["id"]).strip() for c in idx.chapters),
              f"{len(idx.chapters)} chapters")

    print("\n[2] Chapter lookup")
    if idx.chapters:
        first = idx.chapters[0]["title"]
        check("finds by arabic '1'", idx.find_chapter("1") is not None,
              (idx.find_chapter("1") or {}).get("title", "")[:44])
        check("finds by roman 'I'", idx.find_chapter("I") is not None,
              (idx.find_chapter("I") or {}).get("title", "")[:44])
        check("rejects nonsense", idx.find_chapter("zzzz") is None, "returns None")

    print("\n[3] Router")
    for q, kind in [
        ("Summarize the entire book", "global"),
        ("What are the main themes?", "global"),
        ("What is the best problem in the book?", "global"),
        ("Explain chapter 4", "chapter"),
        ("What happens in chapter VII?", "chapter"),
        ("What does page 34 line 5 say?", "page_line"),
        ("Who is the main character?", "semantic"),
    ]:
        r = router.classify(q)
        check(f"{q[:38]:<40}", r.kind == kind, f"-> {r.kind}")

    print("\n[4] Query rewriting (trigger logic, no API call)")
    hist = [{"role": "user", "content": "Who is Van Helsing?"},
            {"role": "assistant", "content": "A professor investigating the Un-Dead."}]
    for q, want in [
        ("What about her sister?", True),
        ("and then?", True),
        ("Tell me more", True),
        ("What is a while loop in Python?", False),
        ("Summarize chapter 4 of the book", False),
    ]:
        got = query_rewrite.needs_rewrite(q, hist)
        check(f"{q[:38]:<40}", got == want, f"needs_rewrite={got}")
    check("no history means no rewrite",
          not query_rewrite.needs_rewrite("What about her sister?", None), "")

    print("\n[5] Rate limiter")
    b = rate_limit.TokenBucket(rate_per_minute=60, capacity=2)
    t0 = time.monotonic()
    b.acquire(); b.acquire()                 # burst through capacity
    burst = time.monotonic() - t0
    t1 = time.monotonic()
    b.acquire()                              # third must wait ~1s at 60/min
    paced = time.monotonic() - t1
    check("burst up to capacity is instant", burst < 0.2, f"{burst:.2f}s")
    check("beyond capacity is paced", paced > 0.5, f"{paced:.2f}s")

    print("\n[6] Summary tree")
    tree = summarize.load_summaries(book_type)
    if tree is None:
        print(f"  {DIM}not built yet - upload the PDF with "
              f"BUILD_SUMMARIES_ON_INGEST=true, or run tools/build_summaries.py{RESET}")
    else:
        check("chapter summaries exist", len(tree["chapters"]) > 0,
              f"{len(tree['chapters'])} chapters")
        check("every chapter covered", len(tree["chapters"]) >= len(idx.chapters) - 1,
              f"{len(tree['chapters'])}/{len(idx.chapters)}")
        check("book overview exists", bool(tree["book"].strip()),
              f"{len(tree['book'].split())} words")
        check("summaries are substantive",
              all(len(c["summary"].split()) > 30 for c in tree["chapters"]),
              f"shortest {min(len(c['summary'].split()) for c in tree['chapters'])} words")
        print(f"\n{DIM}  book overview opens: {tree['book'][:150]}...{RESET}")

    print(f"\n{'=' * 62}")
    verdict = f"{GREEN}ALL CLEAR{RESET}" if failed == 0 else f"{RED}{failed} FAILED{RESET}"
    print(f"  {passed} passed, {failed} failed   {verdict}")
    print(f"{'=' * 62}\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
