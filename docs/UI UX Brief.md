# 04 — UI / UX Brief

**Project** Multimodal RAG Studio
**Author** Abrash Ali — AI Engineering Intern
**Organisation** Zenvyro Labs
**Repository** github.com/abrashhalii/multimodal-rag-studio
**Status** Delivered

---

## 1. Scope

The interface was supplied as part of the project scaffolding: a single-page
application with a sidebar and a chat panel. This brief documents the interaction
model as delivered, the one substantive change made to it, and the reasoning
behind the surfaces exposed for observation.

The visual design was not the subject of this project. The interaction model was,
in one specific respect.

## 2. Layout

```
┌────────────────┬──────────────────────────────────┐
│  Assistant     │  Conversation                    │
│  Mode          │                                  │
│   • Textbook   │    ┌──────────────────────────┐  │
│   • Novel      │    │  assistant message       │  │
│   • Comic      │    └──────────────────────────┘  │
│                │              ┌────────────────┐  │
│  Knowledge     │              │  user message  │  │
│  Base          │              └────────────────┘  │
│   [ Upload ]   │                                  │
│   status line  ├──────────────────────────────────┤
│                │  [ Ask a question…        ] [→]  │
└────────────────┴──────────────────────────────────┘
```

## 3. Interaction model

### 3.1 Mode as document scope

The three modes are not display filters. Each corresponds to a separate document
collection, and switching mode changes which corpus is queried.

This matters because the underlying failure it prevents is invisible: without
separation, a question about the novel can be answered from the textbook, and the
answer looks entirely plausible. The mode selector is therefore a correctness
control presented as a navigation control.

### 3.2 Upload

One button, one file. The status line reports the outcome in the system's own
terms — pages indexed, chunks produced, how page numbers were resolved, and the
offset between file position and printed page.

That last detail is deliberately surfaced rather than hidden. It is the clearest
possible signal that extraction understood the document, and if it is wrong the
user sees it immediately rather than discovering it three questions later.

### 3.3 Conversation

Free-text input. No mode switches, no query syntax, no operators. The user writes
the question they would ask a person, and the system classifies intent internally.

This is a design position: the router exists so that the interface does not need
one. Requiring a user to select "page lookup" before asking about a page would
move the system's internal structure into the user's head.

### 3.4 Per-mode conversation history

**The one substantive change made to the supplied interface.**

Each mode maintains its own conversation. Switching between them preserves both
transcripts separately.

The reason is functional rather than cosmetic. History is used to rewrite
follow-up questions before retrieval. Carrying a novel exchange into the textbook
context would corrupt that rewriting — a pronoun would resolve to a character from
the wrong book, and the resulting search would be confidently wrong.

History is recorded only after a successful response, so a failed request never
leaves an unanswered turn in the transcript for the rewriter to work from.

## 4. Feedback and state

| State | Signal |
|---|---|
| Processing upload | Spinner with status text |
| Upload complete | Green status line with extraction detail |
| Awaiting response | Typing indicator |
| Response received | Markdown-rendered message |
| Error | Inline message in the conversation, not a modal |

Errors appear in the conversation because that is where the user is looking and
because an error is information about the question they asked, not a system-level
interruption.

## 5. Answer presentation

Answers are rendered as markdown, so structure, emphasis and code blocks display
correctly. Three conventions are consistent across all routes:

**Page citations are inline**, using the printed page number — the number the
reader sees on the paper, not the position in the file. A citation the reader
cannot check is not a citation.

**Location answers show their provenance.** A page-and-line answer states the
printed page, the corresponding position in the file, and how many lines that page
contains — enough for the reader to verify it independently.

**Refusal is explicit.** When the retrieved context does not support an answer,
the system says so rather than producing something plausible. It will also
contradict a false premise: asked to compare a problem on a page that contains no
problem, it reports what the page actually contains.

## 6. Observability surface

`GET /api/status` returns the system's own account of its state: which documents
are loaded, how many pages carry text, which method resolved the page numbering,
whether margin bands were calibrated or defaulted, how chapters were detected,
whether extraction was textual or visual, whether a summary tree exists, and the
current behaviour of the rate limiter.

This is not a user-facing feature. It exists so that the system's behaviour can be
inspected rather than inferred — during development, during demonstration, and
when diagnosing a fault.

## 7. Known interface limitations

- **No streaming.** Responses appear complete. Streaming would reduce perceived
  latency substantially and is the highest-value remaining improvement.
- **No source panel.** Citations are textual; the retrieved passages are not shown
  alongside the answer.
- **History is session-only.** Refreshing the page clears it.
- **Single document per mode.** Uploading replaces rather than adds.
- **Script caching.** Browsers cache the application script aggressively enough
  that a version query string was required to force reload during development.
