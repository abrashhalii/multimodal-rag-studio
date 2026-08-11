# 02 — Technical Requirements Document

**Project** Multimodal RAG Studio
**Author** Abrash Ali — AI Engineering Intern
**Organisation** Zenvyro Labs
**Repository** github.com/abrashhalii/multimodal-rag-studio
**Status** Delivered

---

## 1. Technology stack

| Layer | Choice | Rationale |
|---|---|---|
| API | FastAPI + Uvicorn | Provided in the starter; async-capable, minimal |
| Orchestration | LangChain | Prompt templating and model abstraction only |
| Vector store | ChromaDB, persistent | Local, no server process, native metadata filtering |
| Embeddings | `BAAI/bge-small-en-v1.5` | 512-token window, runs locally, no API cost |
| Lexical retrieval | `rank_bm25` (BM25Okapi) | In-memory, built from the vector store's own documents |
| Generation | Gemini 3.5 Flash Lite | The only free-tier model with a workable daily quota |
| Offline fallback | Ollama (`llama3.2`) | Full offline operation via one config value |
| PDF handling | PyMuPDF, pypdf | Geometry-aware line extraction; page label reading |
| Image handling | Pillow | Comic archive conversion and page rendering |
| Config | python-dotenv | No hardcoded values anywhere |

## 2. Architecture

The system separates two phases with different cost profiles.

**Ingestion** runs once per document. It is slow, makes API calls, and produces
durable artifacts. **Querying** runs per question. It is fast and makes at most
two API calls, often zero.

Every expensive operation belongs to ingestion. This is the principle behind the
summary tree, the vision cache, and the page index.

### 2.1 Ingestion

```
document ──▶ extraction ──▶ page index ──┬──▶ chunking ──▶ vector store
                                          └──▶ chapters ──▶ summary tree
```

Two extraction paths converge on one representation:

- **Text path** — PyMuPDF returns lines with bounding boxes. Margin furniture is
  removed by vertical position, the printed page number is resolved, chapters are
  detected from the PDF outline or from headings.
- **Vision path** — each page is rendered to a JPEG and sent to a vision model,
  which returns structured JSON. Each extracted text item becomes a line.

The output of both is identical in shape: a list of pages, each with a printed
page number and a list of numbered lines. This is the single most important
design decision in the system, because it means no downstream component needs to
know which path produced its input.

### 2.2 Query

```
question ──▶ rewriter ──▶ router ──┬──▶ page + line   (deterministic, no model)
                                    ├──▶ page(s)       (exact fetch, then model)
                                    ├──▶ chapter       (scoped retrieval)
                                    ├──▶ global        (summary tree)
                                    ├──▶ memory        (transcript only)
                                    └──▶ semantic      (hybrid retrieval)
```

## 3. Component responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Single source of configuration, read from `.env` |
| `page_index.py` | Line extraction, margin calibration, printed page resolution, chapter detection, persistence |
| `vision.py` | Page rendering, batched vision extraction, JSON parsing, content-hash caching |
| `ingest.py` | Page-bounded chunking, metadata construction, embedding, storage |
| `hybrid.py` | BM25 index construction, dense retrieval, reciprocal rank fusion |
| `summarize.py` | Chapter and book summary construction with incremental caching |
| `query_rewrite.py` | Follow-up detection and standalone question rewriting |
| `router.py` | Intent classification by regular expression |
| `rate_limit.py` | Token bucket pacing and quota-aware retry |
| `llm_provider.py` | Provider abstraction and model instantiation |
| `rag_engine.py` | Route dispatch, prompt assembly, response generation |
| `main.py` | HTTP endpoints and static file serving |

## 4. Key technical decisions

### 4.1 Printed page resolution

A PDF's internal page index is not the number printed on the page. Front matter
displaces them. Resolution proceeds in order of reliability:

1. `/PageLabels` from the PDF catalog, **only if the catalog key exists**. The
   pypdf accessor synthesises sequential labels when it does not, so trusting it
   unconditionally reintroduces the bug it was meant to solve.
2. Folio detection by regular expression, taking the modal offset between file
   position and printed number.
3. Sequential position, used for image-only documents where no folio exists.

The method used is recorded on the index and exposed through the status endpoint.

### 4.2 Margin calibration

Header and footer removal cannot use a fixed proportion of page height, because
publishers differ. One source book prints its folio at 93.6% of page height; the
other prints it in the running head at 7.7–9.0%.

The extractor therefore locates every bare folio in the document, takes the
**median** of their vertical positions, and sets the margin band to just clear
them. The median rather than the extreme is required: a single anomalous page can
otherwise displace the band far into the body text. Hard clamps bound the result
regardless.

### 4.3 Chunking

Chunks are line-aligned and never cross a page boundary. A chunk spanning two
pages makes page attribution unprovable, which would undermine the precision
requirement.

Semantic chunking was rejected for the same reason: it places boundaries at
meaning transitions, which ignore page boundaries entirely.

Chunk size is bounded by the embedding model's context window. Exceeding it
causes silent truncation, not an error.

### 4.4 Location metadata

Chunk location is stored in metadata **and** re-attached to the chunk text at
retrieval time, not embedded with it.

Embedding an identical header into every chunk pulls all vectors toward a common
centroid and degrades separation, with the effect most severe on short chunks. But
metadata alone never reaches the model's context window. Storing it and
re-attaching it at retrieval satisfies both constraints.

### 4.5 Hybrid retrieval

Dense retrieval was measured at hit@12 = 0.71 on verbatim phrase recall, with
failures concentrated in the novel. A bi-encoder maps text to a single vector, so
a corpus of homogeneous prose produces vectors that cluster too tightly to
separate — while an exact phrase, the strongest available signal, is invisible to
it, because embeddings encode meaning and discard tokens.

BM25 is the complementary instrument. The two are fused by **reciprocal rank
fusion**:

```
score(document) = Σ over retrievers of 1 / (60 + rank)
```

Fusion uses rank rather than score because cosine distance and BM25 relevance are
not comparable quantities, and normalising them is fragile because their ranges
shift per query.

Measured result: hit@12 rose from 32/45 to 43/45.

### 4.6 Summarisation granularity

Chapter level, not page level. Page-level summaries would require 647 requests
for two books against a 500/day quota. Chapter level requires approximately 48.

Chapter boundaries derive from the PDF outline where present, from heading
patterns otherwise.

### 4.7 Vision extraction

Comic pages are rendered at 1400px on the longest edge and submitted individually.

Batching multiple pages per request was implemented and measured: it reduces
request count roughly fourfold at negligible token cost, and text extraction is
unaffected. However, speaker attribution fell from 46% to 19% named — batching
does not lose text, it loses the fine-grained visual work of tracing a balloon
tail to a figure. Since request count was not the binding constraint, single-page
requests were chosen.

Every page is cached by the SHA-256 of its rendered image, written after each
page, making the operation resumable and re-runs free.

### 4.8 Rate limiting

A token bucket with **capacity 1**. Sizing capacity to the rate is correct for
smoothing average load and wrong against a fixed provider window: a full bucket
discharges the entire minute's quota in seconds.

Effective rate is set below the stated ceiling, because the provider's window is
not aligned with the client's. Quota rejections are retried using the delay the
provider itself supplies rather than a guessed backoff.

## 5. Security posture

The corpus is trusted: public-domain fiction, a Creative Commons textbook, and a
commercial comic used locally. No defences against instructions embedded in
document content are implemented.

For untrusted input the following would be required, in order of leverage:
delimiting retrieved content and declaring it as data rather than instructions;
scanning at ingestion for directives aimed at a model; verifying cited pages
against the page index deterministically; and treating vision extraction output as
untrusted, since it is model-generated text re-entering a prompt on a second hop.

Generated artifacts contain the full text of source material and are excluded from
version control.

## 6. Testing strategy

| Suite | Coverage | API cost |
|---|---|---|
| `smoke_test.py` | 23 checks on extraction, numbering, margins, chunking, routing | None |
| `test_day2.py` | 25 checks on chapters, rewriting, rate limiting, summaries | None |
| `eval_retrieval.py` | hit@k and MRR over 66 questions, two sets scored separately | None |
| `diagnose_retrieval.py` | Failure isolation for a single query | None |

No test requires an API key. This is deliberate: a test suite that costs quota is
a suite that does not get run.

## 7. Configuration

All settings are read from `.env`, with defaults in `config.py`. Settings of
consequence:

| Setting | Default | Effect |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `ollama` runs fully offline |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local, 512-token window |
| `CHUNK_CHARS` | `900` | Must stay within the embedding window |
| `HYBRID_RETRIEVAL` | `true` | `false` reverts to dense retrieval |
| `RETRIEVAL_K` | `12` | Chunks passed to the model |
| `LLM_RPM` / `RATE_LIMIT_SAFETY` | `15` / `0.7` | Effective pacing |
| `VISION_BATCH_PAGES` | `1` | Pages per vision request |
| `COMIC_CHARACTERS` | — | Cast list for speaker attribution |
