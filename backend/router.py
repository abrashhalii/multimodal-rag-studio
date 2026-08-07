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
_PG = r"(?:page|pg\.?|p\.)\s*"
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

MEMORY_RE = re.compile(
    r"\b(what did i (just )?(ask|say)|my (last|previous) (question|message)|"
    r"repeat (that|my question)|what was my)\b", re.I)


@dataclass
class Route:
    kind: str                      # page_line | page | chapter | global | memory | semantic
    page: Optional[int] = None
    line: Optional[int] = None
    chapter: Optional[str] = None
    reason: str = ""

    def __str__(self):
        bits = [self.kind]
        if self.page is not None:
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

    # 2. page only -> load that page, let the LLM read it
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
