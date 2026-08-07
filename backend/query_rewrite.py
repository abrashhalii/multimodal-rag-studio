"""
History-aware query rewriting - the half of "Amnesia" (Bug 3) that people skip.

Pasting the transcript into the prompt makes a model SOUND like it remembers.
It does not make retrieval work. "What about her sister?" embedded on its own
carries almost no signal: there is no "her" in the vector space. Retrieval runs
before generation, so by the time the transcript reaches the model the wrong
passages have already been fetched.

So the follow-up is rewritten into a standalone question FIRST, then that
rewritten question drives retrieval, and the original transcript still goes to
the model for tone and continuity.

Two economies, because the free tier allows 500 requests/day:
  * only rewrite when the question actually looks dependent on context
  * use the cheap model - this is mechanical work, not reasoning
"""

from __future__ import annotations

import re
from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
import rate_limit
from llm_provider import get_fast_llm

REWRITE_PROMPT = """Rewrite the reader's latest question so it stands alone, using the
conversation for context. Resolve pronouns and references ("it", "he", "that chapter",
"the second one") into explicit nouns taken from the conversation.

Rules:
- Output ONLY the rewritten question. No preamble, no explanation, no quotes.
- Keep it close to the original. Do not answer it, expand it, or add detail.
- If the question already stands alone, return it unchanged.
- Never invent names or facts that are not in the conversation.

Conversation:
{history}

Latest question: {question}

Standalone question:"""

# Signals that a question leans on earlier turns.
PRONOUNS = re.compile(
    r"\b(it|its|he|him|his|she|her|hers|they|them|their|this|that|these|those|"
    r"the (?:former|latter|first|second|third|last|next|previous|same)|"
    r"there|then|else|another|other one|one)\b", re.I)

CONTINUATION = re.compile(
    r"^\s*(and|but|so|also|what about|how about|why|why not|and then|then|ok|okay|"
    r"more|tell me more|go on|continue|explain more|elaborate)\b", re.I)


def needs_rewrite(question: str, history: Optional[List[dict]]) -> bool:
    if not history or not config.REWRITE_FOLLOWUPS:
        return False
    q = (question or "").strip()
    if not q:
        return False
    if len(q.split()) <= 4:            # "and the sister?" - almost certainly a follow-up
        return True
    if CONTINUATION.match(q):
        return True
    return bool(PRONOUNS.search(q))


def format_history(history: Optional[List[dict]], limit: int = None) -> str:
    limit = limit or config.HISTORY_TURNS
    if not history:
        return ""
    lines = []
    for turn in history[-limit:]:
        role = (turn.get("role") or "user").lower()
        who = "Reader" if role in ("user", "human") else "Tutor"
        content = (turn.get("content") or "").strip()
        if len(content) > 600:          # keep the window small; older turns get clipped
            content = content[:600] + " [...]"
        lines.append(f"{who}: {content}")
    return "\n".join(lines)


def rewrite(question: str, history: Optional[List[dict]]) -> str:
    """Return a standalone version of `question`, or the original if not needed."""
    if not needs_rewrite(question, history):
        return question
    try:
        chain = (ChatPromptTemplate.from_template(REWRITE_PROMPT)
                 | get_fast_llm() | StrOutputParser())
        rate_limit.throttle("query rewrite")
        out = chain.invoke({
            "history": format_history(history),
            "question": question,
        }).strip().strip('"').strip()
    except Exception as e:
        if config.DEBUG_LOG:
            print(f"[rewrite] failed, using original: {e}")
        return question

    # Guard against a model that ignores the instructions and writes an essay.
    if not out or len(out) > 400 or "\n" in out.strip():
        out = out.split("\n")[0][:400] if out else question
    if not out.strip():
        return question

    if config.DEBUG_LOG and out.lower() != question.lower():
        print(f"[rewrite] {question!r} -> {out!r}")
    return out
