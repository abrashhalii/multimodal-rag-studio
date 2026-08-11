"""
Query router - the single most important architectural decision in this project.

"Vector blindness" is not an embedding-quality problem, so no amount of better
embeddings fixes it. "What is on page 34 line 5?" carries almost no semantic
signal matching the target text: cosine similarity has nothing to grip. The
question is a database lookup wearing a question's clothing.

So we classify intent BEFORE retrieval and send each class down a different
path. Location queries never touch the vector store at all.

Deliberately regex-first, no LLM: it is instant, free, deterministic, and
testable. An LLM classifier here would add latency and a failure mode to the
one route that must never fail during the live demo.
"""

import re
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------- patterns
# Plural matters ("pages 12-14"), and so does the optional filler word:
# readers write "page number 120" and "page no. 34" as often as "page 120".
_PG = r"(?:pages?|pgs?\.?|pp?\.)\s*(?:numbers?|nos?\.?|num|#)?\s*"
_LN = r"(?:line|ln\.?|l\.)\s*"
_NUM = r"(\d{1,4})"

PAGE_LINE_PATTERNS = [
    re.compile(_PG + _NUM + r"\s*,?\s*(?:and\s+)?" + _LN + _NUM, re.I),
    re.compile(_LN + _NUM + r"\s*(?:of|on|from)\s*" + _PG + _NUM, re.I),
]

PAGE_PATTERNS = [
    re.compile(_PG + _NUM, re.I),
    re.compile(r"\bon\s+page\s+" + _NUM, re.I),
]

# "pages 100 to 110", "page 100-110", "from page 100 through 110"
PAGE_RANGE_RE = re.compile(
    _PG + _NUM + r"\s*(?:to|through|until|\-|\u2013|\u2014)\s*(?:page\s*)?" + _NUM,
    re.I)

# "page 110, 64, 171" / "pages 110 and 64" - a list, not a range
PAGE_LIST_RE = re.compile(
    _PG + r"((?:\d{1,4})(?:\s*(?:,|and|&)\s*\d{1,4})+)", re.I)

GLOBAL_MARKERS = [
    r"\bentire book\b", r"\bwhole book\b", r"\bthe book\b.*\bsummar",
    r"\bsummar(?:y|ise|ize)\b.*\b(book|novel|volume|textbook|everything)\b",
    r"\boverall\b", r"\bin general\b", r"\bmain (?:themes|ideas|points|arguments)\b",
    r"\bkey takeaways\b", r"\bbest (?:problem|chapter|exercise|part)\b",
    r"\bacross the (?:book|novel|story)\b", r"\bevery chapter\b",
    r"\ball chapters\b", r"\bfrom start to finish\b", r"\bplot summary\b",
]
GLOBAL_RE = [re.compile(p, re.I) for p in GLOBAL_MARKERS]

CHAPTER_RE = re.compile(
    r"\bchapter\s+([0-9]{1,3}|[ivxlcdm]{1,8})\b", re.I)

# Guard rail: a huge span would overflow the context window, so past this the
# question is treated as a broad one rather than a literal page dump.
MAX_PAGE_SPAN = 30

MEMORY_RE = re.compile(
    r"\b(what did i (just )?(ask|say)|my (last|previous) (question|message)|"
    r"repeat (that|my question)|what was my)\b", re.I)


@dataclass
class Route:
    kind: str                      # page_line | page | pages | chapter | global | memory | semantic
    page: Optional[int] = None
    pages: Optional[list] = None   # several pages, for a range or an explicit list
    line: Optional[int] = None
    chapter: Optional[str] = None
    reason: str = ""

    def __str__(self):
        bits = [self.kind]
        if self.pages:
            bits.append(f"pages={self.pages[0]}..{self.pages[-1]} ({len(self.pages)})")
        elif self.page is not None:
            bits.append(f"page={self.page}")
        if self.line is not None:
            bits.append(f"line={self.line}")
        if self.chapter:
            bits.append(f"chapter={self.chapter}")
        return " ".join(bits)


def classify(question: str) -> Route:
    q = (question or "").strip()
    if not q:
        return Route("semantic", reason="empty question")

    # 1. page + line  -> exact deterministic lookup, no LLM needed
    for i, pat in enumerate(PAGE_LINE_PATTERNS):
        m = pat.search(q)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            page, line = (a, b) if i == 0 else (b, a)
            return Route("page_line", page=page, line=line,
                         reason="explicit page and line reference")

    # 2a. an explicit RANGE of pages
    m = PAGE_RANGE_RE.search(q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        if b - a <= MAX_PAGE_SPAN:
            return Route("pages", pages=list(range(a, b + 1)),
                         reason=f"page range {a}-{b}")

    # 2b. an explicit LIST of pages
    m = PAGE_LIST_RE.search(q)
    if m:
        nums = [int(x) for x in re.findall(r"\d{1,4}", m.group(1))]
        nums = sorted(dict.fromkeys(nums))
        if 2 <= len(nums) <= MAX_PAGE_SPAN:
            return Route("pages", pages=nums,
                         reason=f"explicit list of {len(nums)} pages")

    # 2c. page only -> load that page, let the LLM read it
    for pat in PAGE_PATTERNS:
        m = pat.search(q)
        if m:
            return Route("page", page=int(m.group(1)),
                         reason="explicit page reference")

    # 3. asking about the conversation itself
    if MEMORY_RE.search(q):
        return Route("memory", reason="question about the chat history")

    # 4. whole-book / large-scope questions
    for pat in GLOBAL_RE:
        if pat.search(q):
            return Route("global", reason="global scope marker")

    # 5. a named chapter -> scoped retrieval
    m = CHAPTER_RE.search(q)
    if m:
        return Route("chapter", chapter=m.group(1).upper(),
                     reason="named chapter")

    return Route("semantic", reason="default semantic retrieval")


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    cases = [
        "What does page 34 line 5 say?",
        "what is on page 34 line 5",
        "read me line 12 of page 200",
        "What happened on page 23?",
        "give summary from page 100 to 110",
        "what happened in page 110,64,171",
        "summarise pages 12-14",
        "the problem from page number 120",
        "what is on page no. 47",
        "p. 88 please",
        "Summarize the entire book",
        "What is the best problem in the book?",
        "give me the main themes",
        "What did I just ask you?",
        "Explain chapter IV",
        "Who is Van Helsing?",
        "how does the while loop work",
    ]
    for c in cases:
        print(f"{classify(c)!s:38} <- {c}")
