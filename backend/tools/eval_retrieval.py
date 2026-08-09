"""
Retrieval evaluation - measure the stage that decides everything.

WHY THIS EXISTS
---------------
A RAG pipeline is retrieve-then-generate. If retrieval misses, the model cannot
recover: it answers from the wrong passages, fluently and confidently. So the
number worth knowing is not "do the answers look right" but "did the passage
that actually contains the answer make it into the context at all".

hit@k = fraction of questions where the correct page appears in the top k
        retrieved chunks
MRR   = mean of 1/rank of the first correct page; rewards ranking it 1st
        rather than 8th

This runs retrieval ONLY. No LLM call, no API cost, so it can be run as often
as you like - which is the point: a metric you can only afford occasionally is
a metric you will not use.

It would also have caught the silent embedding failure immediately. Retrieval
was returning arbitrary chunks while the system still produced plausible prose;
hit@5 would have been near zero from the first run.

    python tools/eval_retrieval.py                 # curated set
    python tools/eval_retrieval.py --auto 15       # + auto-generated recall set
    python tools/eval_retrieval.py --book novel    # one book only
    python tools/eval_retrieval.py --verbose       # show every miss
"""

import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config                                    # noqa: E402
import hybrid                                    # noqa: E402
from page_index import BookIndex                 # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = ("\033[92m", "\033[91m", "\033[93m",
                                  "\033[90m", "\033[0m")

NOT_INDEX = {"is_index": {"$eq": False}}

# ---------------------------------------------------------------- curated set
# Real questions a reader would ask, with the page that actually answers them
# verified against the printed book. Paraphrases, not copied sentences - the
# point is to test semantic retrieval, not string matching.
CURATED = [
    # ---- novel (Dracula) -------------------------------------------------
    ("novel", "What does Dracula make Harker write to reassure people back home?", ["34"]),
    ("novel", "How does the Count react when Harker cuts himself shaving?", ["24", "25", "26"]),
    ("novel", "What does Harker see the Count doing on the castle wall?", ["37", "38", "39"]),
    ("novel", "Who proposes to Lucy on the same day?", ["58", "59", "60", "61"]),
    ("novel", "What happens to the crew of the ship that reaches Whitby?", ["81", "82", "83", "84", "85"]),
    ("novel", "What marks are found on Lucy's throat?", ["102", "132"]),
    ("novel", "What does Van Helsing use to protect Lucy's room?", ["131", "132", "133"]),
    ("novel", "How does Van Helsing say Lucy's body must be dealt with?", ["213"]),
    ("novel", "What is the bloofer lady?", ["185", "186", "187", "188"]),
    ("novel", "What happens to Renfield near the end?", ["292", "293", "294"]),
    ("novel", "How does the group finally destroy the Count?", ["397", "398", "399", "400"]),

    # ---- coding (Think Python) -------------------------------------------
    ("coding", "How does the flow of execution work in a while statement?", ["65"]),
    ("coding", "How do you write a function that draws a circle with a turtle?", ["34"]),
    ("coding", "What is a stack diagram used for?", ["23", "24"]),
    ("coding", "What is the difference between a return value and a void function?", ["43", "44", "45"]),
    ("coding", "How do you read a word list from a file?", ["114", "115", "143"]),
    ("coding", "What is a dictionary and how is it different from a list?", ["103", "104", "105"]),
    ("coding", "What does it mean for a function to be pure?", ["157", "158"]),

    # ---- manga (vision-extracted comic) ----------------------------------
    ("manga", "What does Holocaust shout at the mutants?", ["130"]),
    ("manga", "What does Magneto say about humans and mutants living in harmony?", ["15"]),
    ("manga", "Who does Magneto compare Holocaust to?", ["240"]),
]


def load_cases(book_filter=None):
    return [c for c in CURATED if not book_filter or c[0] == book_filter]


# ---------------------------------------------------------------- auto set
def auto_cases(book_type: str, n: int, seed: int = 7):
    """Sample real pages and query with a distinctive phrase from each.

    Weaker than the curated set - a verbatim phrase is an easy target - so it
    is scored separately. What it DOES prove is index integrity across the whole
    corpus rather than the handful of pages a human bothered to write questions
    for, and it works on any book without hand-authoring.
    """
    idx = BookIndex.load(book_type)
    if idx is None:
        return []
    usable = [p for p in idx.pages
              if not p.is_front_matter and not p.is_index and len(p.lines) >= 4]
    if not usable:
        return []
    rng = random.Random(seed)
    picks = rng.sample(usable, min(n, len(usable)))

    out = []
    for p in picks:
        best = max(p.lines, key=lambda l: len(re.sub(r"[^A-Za-z ]", "", l)))
        phrase = re.sub(r"^\[[a-z]+\]\s*", "", best)          # drop [caption]/[sfx]
        phrase = re.sub(r"^[A-Za-z ]+:\s*", "", phrase)       # drop "Speaker: "
        phrase = " ".join(phrase.split()[:14]).strip()
        if len(phrase.split()) >= 5:
            out.append((book_type, phrase, [p.printed_page]))
    return out


# ---------------------------------------------------------------- scoring
def rank_of_hit(book_type, question, expected, k):
    # Goes through the SAME retrieval path the app uses, so the number measured
    # is the number that ships.
    docs = hybrid.search(book_type, question, k)
    pages = [str(d.metadata.get("printed_page")) for d in docs]
    want = {str(e) for e in expected}
    for i, pg in enumerate(pages, 1):
        if pg in want:
            return i, pages
    return None, pages


def run(cases, k, label, verbose=False):
    if not cases:
        return None
    hits = {1: 0, 3: 0, 5: 0, k: 0}
    rr_total = 0.0
    misses = []

    print(f"\n{label}  ({len(cases)} questions, k={k})")
    for book_type, question, expected in cases:
        rank, pages = rank_of_hit(book_type, question, expected, k)
        if rank:
            rr_total += 1.0 / rank
            for kk in hits:
                if rank <= kk:
                    hits[kk] += 1
            mark = (f"{GREEN}hit@{rank}{RESET}" if rank <= 5
                    else f"{YELLOW}hit@{rank}{RESET}")
        else:
            mark = f"{RED}MISS{RESET}"
            misses.append((book_type, question, expected, pages))
        print(f"  {mark:22} [{book_type}] {question[:58]}")

    n = len(cases)
    print(f"\n  {'metric':<12}{'score':>10}")
    for kk in sorted(hits):
        print(f"  hit@{kk:<9}{hits[kk]}/{n} = {hits[kk]/n:.2f}".rjust(0))
    print(f"  MRR         {rr_total/n:.3f}")

    if misses and verbose:
        print(f"\n  {DIM}misses:{RESET}")
        for bt, q, exp, got in misses:
            print(f"    [{bt}] {q}")
            print(f"      expected {exp}  got {got[:8]}")
    return {"n": n, "hits": hits, "mrr": rr_total / n, "misses": misses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", choices=list(config.BOOK_TYPES))
    ap.add_argument("--k", type=int, default=config.RETRIEVAL_K)
    ap.add_argument("--auto", type=int, default=0,
                    help="add N auto-generated recall questions per book")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--dense-only", action="store_true",
                    help="force dense retrieval, to A/B against hybrid")
    a = ap.parse_args()
    if a.dense_only:
        config.HYBRID_RETRIEVAL = False

    print(f"\n{'=' * 66}\n  RETRIEVAL EVALUATION\n{'=' * 66}")
    mode = ("hybrid (BM25 + dense, RRF)" if config.HYBRID_RETRIEVAL
            else "dense only")
    print(f"  embedding : {config.EMBEDDING_MODEL}")
    print(f"  retrieval : {mode}, k={a.k}, index pages excluded")
    if config.HYBRID_RETRIEVAL:
        print(f"  candidates: {config.CANDIDATE_K} per retriever before fusion")

    cur = run(load_cases(a.book), a.k,
              "CURATED - paraphrased questions, verified pages", a.verbose)

    auto_res = None
    if a.auto:
        cases = []
        for bt in ([a.book] if a.book else config.BOOK_TYPES):
            cases += auto_cases(bt, a.auto)
        auto_res = run(cases, a.k,
                       "AUTO - verbatim phrase recall (index integrity)", a.verbose)

    print(f"\n{'=' * 66}")
    if cur:
        h5 = cur["hits"][5]
        print(f"  Headline: correct page in top 5 for {h5}/{cur['n']} "
              f"paraphrased questions (hit@5 = {h5/cur['n']:.2f})")
    if auto_res:
        h5 = auto_res["hits"][5]
        print(f"            verbatim recall {h5}/{auto_res['n']} "
              f"(hit@5 = {h5/auto_res['n']:.2f})")
    print(f"{'=' * 66}\n")


if __name__ == "__main__":
    main()
