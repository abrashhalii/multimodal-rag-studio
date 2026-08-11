# 03 — Application Flow

**Project** Multimodal RAG Studio
**Author** Abrash Ali — AI Engineering Intern
**Organisation** Zenvyro Labs
**Repository** github.com/abrashhalii/multimodal-rag-studio
**Status** Delivered

---

## 1. Flow A — Ingesting a text document

Triggered by `POST /api/upload` with a PDF and a document type.

```
1. File saved to temp storage
2. Every page's lines extracted with bounding boxes        (PyMuPDF)
3. Margin bands calibrated from folio positions            (median, clamped)
4. Header and footer removed by vertical position
5. Printed page numbers resolved                           (labels → regex → sequential)
6. Chapters detected                                       (outline → headings → none)
7. Page index persisted to disk
8. Pages chunked, never crossing a page boundary
9. Location metadata attached to each chunk
10. Existing collection cleared
11. Chunks embedded locally and stored
12. BM25 cache invalidated
13. Summary of the operation returned to the interface
```

Steps 3 to 5 are the substance. Everything before is mechanical and everything
after is standard.

**Failure modes.** A document with no text layer yields empty pages and reports
zero pages with text — a loud failure rather than a silent one. A document with no
detectable folio falls back to sequential numbering and records that it did so.

## 2. Flow B — Ingesting a comic

Triggered by the same endpoint with document type `manga`, or by the command-line
tool for large volumes.

```
1. Each page rendered to JPEG at 1400px longest edge
2. SHA-256 computed per rendered image
3. Cache consulted; already-extracted pages skipped entirely
4. For each remaining page:
     a. Rate limiter acquires a token
     b. Page image sent to the vision model with a structured prompt
     c. JSON response parsed, tolerating fences and preamble
     d. Balloon line-wrapping normalised
     e. Result written to cache immediately
5. Extracted items converted to numbered lines
6. Page index built with sequential numbering
7. Steps 8 to 13 of Flow A proceed unchanged
```

Step 4e is what makes the operation resumable. An interruption costs one page
rather than the run — which matters at 258 pages against a 500-request daily
quota.

Step 7 is the point of the whole design: from here the comic is indistinguishable
from a text document.

## 3. Flow C — Answering a question

Triggered by `POST /api/chat`.

```
1. Request received with message, document type, and conversation history
2. Router classifies intent by regular expression
3. If history exists and the question appears dependent:
       question rewritten into standalone form
       the rewritten form drives retrieval; the original is shown to the model
4. Dispatch by intent (see §4)
5. Prompt assembled with retrieved context and conversation history
6. Rate limiter acquires a token
7. Model generates
8. Answer returned
```

Step 3 is the part most implementations omit. Retrieval precedes generation, so a
transcript placed in the prompt cannot repair a query that already retrieved the
wrong passages.

## 4. Route dispatch

### 4.1 Page and line

```
"What does page 34 line 5 say?"
  → page index loaded
  → page located by printed number
  → line returned verbatim
```

No embedding, no retrieval, no model. Approximately one millisecond, and
structurally incapable of hallucination.

### 4.2 Single page

```
"What happens on page 132?"
  → page located by printed number
  → all lines numbered and passed to the model
  → model answers from that page alone
```

### 4.3 Multiple pages

```
"Summarise pages 100 to 110"     →  range expanded
"What happens on pages 110, 64, 171"  →  list parsed
  → each page fetched by number
  → pages labelled and passed together
  → pages not present reported explicitly
```

Fetched by number, not searched for, so no page can be dropped silently.

### 4.4 Chapter

```
"Explain chapter 4"
  → chapter resolved by name, arabic numeral or roman numeral
  → chapter summary retrieved
  → retrieval scoped to that chapter's pages
  → summary provides shape, passages provide detail
```

### 4.5 Global

```
"Summarise the entire book"
  → summary tree loaded
  → book overview plus every chapter summary passed to the model
```

If no tree exists the route degrades to wide retrieval and reports how to build
one. Building is never attempted inside a request: it is roughly fifty paced calls
and several minutes.

### 4.6 Memory

```
"What did I just ask?"
  → answered directly from the transcript
```

Retrieval is irrelevant — the question concerns the conversation, not the book.

### 4.7 Semantic

```
"How does a while loop work?"
  → dense retrieval returns 40 candidates
  → BM25 returns 40 candidates
  → reciprocal rank fusion merges by rank
  → top 12 passed to the model
```

## 5. Flow D — Building the summary tree

Run explicitly before a session, never inside a request.

```
For each chapter:
    chapter text assembled from its pages
    content hash computed
    if hash matches cache → reuse, no API call
    else → rate-limited call, result cached immediately
Compose all chapter summaries into a book overview
```

Content-hash caching means re-running on an unchanged document costs nothing,
which is what makes prompt iteration affordable.

## 6. Failure and recovery

| Condition | Behaviour |
|---|---|
| Quota rejection | Retry using the provider's stated delay, up to four attempts |
| Transient connection loss | Retried by the client with a bounded timeout |
| Malformed model JSON | Parsed leniently; on failure the page is recorded as empty rather than crashing the batch |
| Page not in document | Reported explicitly by number |
| No summary tree | Degrades to wide retrieval with an explanatory message |
| Provider unavailable | One configuration value switches to the local model |

## 7. Observability

Each request emits structured log lines that make the internal path visible:

```
[route]     which path was selected and why
[rewrite]   original question and its rewritten form
[hybrid]    candidate counts and how many were promoted by lexical match
[retrieve]  the pages that reached the model
[pages]     which pages were fetched for a multi-page query
[ratelimit] pacing delays and quota recoveries
```

`GET /api/status` reports loaded documents, how each was processed, and current
limiter state.
