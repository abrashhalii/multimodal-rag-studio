# Multimodal RAG Studio

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-8E75B2?style=flat-square&logo=googlegemini&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=flat-square&logo=ollama&logoColor=white)
![HuggingFace](https://img.shields.io/badge/BGE%20Embeddings-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![BM25](https://img.shields.io/badge/BM25-4B8BBE?style=flat-square)
![PyMuPDF](https://img.shields.io/badge/PyMuPDF-CB3837?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-2F4A6B?style=flat-square)

![hit@12](https://img.shields.io/badge/retrieval%20hit%4012-43%2F45-2E7D32?style=flat-square)
![Pages](https://img.shields.io/badge/pages%20indexed-910-2F4A6B?style=flat-square)
![Vision](https://img.shields.io/badge/vision%20extracted-3%2C121%20items-6A4C93?style=flat-square)
![429s](https://img.shields.io/badge/quota%20rejections-0-2E7D32?style=flat-square)

Page-precise retrieval over a coding textbook, a Victorian novel, and a 258-page
comic volume that contains no text layer at all.

Ask it *"what is on page 34, line 5?"* and it answers in about a millisecond
without calling a language model. Ask it *"what does Magneto say about humans?"*
and it answers from dialogue a vision model read out of drawings.

Built as the third project of the Zenvyro Labs internship.

---

## What makes it different from a tutorial RAG

Most RAG systems embed everything, retrieve the top *k* chunks, and hope. Three
decisions here depart from that, and each was forced by a problem rather than
chosen for style.

**Location queries never touch the vector store.** A page number carries almost no
semantic content, so cosine similarity has nothing to grip — this is the failure
usually called *vector blindness*, and a better embedding model does not fix it. A
router classifies question intent before retrieval runs, and page/line queries
become a deterministic lookup against an index built at ingestion. No model call,
so no hallucination and no rate limit.

**Whole-book questions read a summary tree, not a sample.** Top-*k* retrieval
samples a book; it does not survey it. Ranking the best exercise requires having
read every exercise. Chapter summaries are built once at ingestion and composed
into a book overview, so broad questions derive from every page.

**Every source normalises to one structure.** A PDF with a text layer and a comic
made of images both reduce to *pages made of numbered lines*. Once they look
identical, every feature works on every book with no special casing — `page 130
line 1` runs the same code whether the line came from typeset prose or from a
speech balloon.

---

## Measured, not asserted

Retrieval quality was A/B tested with a fixed question set and seed. Verbatim
phrase recall, 45 questions:

| Metric | Dense only | Hybrid (BM25 + dense) |
|---|---|---|
| Correct page ranked 1st | 16 / 45 | **28 / 45** |
| Correct page in top 5 | 26 / 45 | **38 / 45** |
| **Correct page reaches the model** | **32 / 45** | **43 / 45** |
| MRR | 0.465 | **0.737** |

`hit@12` is the operational metric because twelve chunks are passed to the model —
below that, ordering is cosmetic.

Request pacing, measured across three ingestion runs:

| Run | Calls | Duration | Throttled | Waited | 429s |
|---|---|---|---|---|---|
| Novel summaries | 29 | 174 s | 119× | 112.7 s | 0 |
| Textbook summaries | 20 | 119 s | 193× | 77.3 s | 0 |
| Comic vision ingest | 258 | ~20 min | 465× | 516.7 s | 0 |

---

## Corpus

| Book | Pages | Chunks | Page numbering | Chapters | Extraction |
|---|---|---|---|---|---|
| Think Python 2e | 244 | 663 | PDF page labels | 19 (heading regex) | text layer |
| Dracula | 408 | 1,203 | PDF page labels | 28 (PDF outline) | text layer |
| X-Men: Age of Apocalypse | 258 | 295 | sequential images | — | **vision** |

Three books, three page-numbering paths, two chapter-detection methods, two
extraction pipelines. Nothing hardcoded to a particular file — `GET /api/status`
reports how each was resolved.

The comic produced **3,121 text items** classified into speech, thought, caption
and sound effect, with roughly 80% of dialogue attributed to a named character.

---

## Architecture

```
                        ┌──────────────┐
   question  ──────────▶│   rewriter   │  resolves pronouns in follow-ups
                        └──────┬───────┘
                               ▼
                        ┌──────────────┐
                        │    router    │  regex, no model call
                        └──────┬───────┘
          ┌───────────────┬────┴─────┬────────────────┐
          ▼               ▼          ▼                ▼
    page + line     whole page    semantic         global
    ───────────     ──────────    ────────         ──────
    index lookup    one page      BM25 + dense     summary
    NO MODEL        to model      + RRF fusion     tree
```

Ingestion converges two very different sources on one structure:

```
   textbook ─┐
   novel   ──┴─▶ text layer  ─┐
                              ├─▶ page index ─┬─▶ chunks  ─▶ vector store
   comic     ──▶ vision model ┘  (pages made  └─▶ chapters ─▶ summary tree
                                  of lines)
```

---

## Quick start

```bash
git clone https://github.com/abrashhalii/multimodal-rag-studio.git
cd multimodal-rag-studio/backend

py -3.11 -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

copy .env.example .env         # then add your GEMINI_API_KEY
python main.py
```

Open `http://localhost:8000`, choose a mode, upload a PDF.

Then pre-build the summary tree so whole-book questions are instant:

```bash
python tools/build_summaries.py novel
```

Comics take a separate path, since extraction is the expensive step:

```bash
python tools/comic2pdf.py "issue1.cbz" "issue2.cbr" -o volume.pdf
python tools/build_manga.py volume.pdf --pages 8      # test on a slice first
python tools/build_manga.py volume.pdf                # then the full run
```

---

## Layout

### Pipeline

| Module | Responsibility |
|---|---|
| `config.py` | Every setting, from `.env`. Nothing hardcoded. |
| `page_index.py` | Page and line extraction, printed page resolution, chapter detection, margin calibration |
| `vision.py` | Comic path — render, extract, parse, cache by image hash |
| `ingest.py` | Page-bounded chunking, metadata, vector storage |
| `hybrid.py` | BM25 fused with dense retrieval via reciprocal rank fusion |
| `summarize.py` | Chapter and book summary trees, cached by content hash |
| `query_rewrite.py` | Turns follow-ups into standalone questions before retrieval |
| `router.py` | Question intent classification. Pure regex — instant, free, testable |
| `rate_limit.py` | Token bucket pacing, retry honouring the server's own delay |
| `llm_provider.py` | Gemini primary, Ollama offline fallback, one config line apart |
| `rag_engine.py` | Orchestration |
| `main.py` | FastAPI — upload, chat, status |

### Tools

| Tool | Purpose |
|---|---|
| `smoke_test.py` | 23 extraction checks. No API key needed. |
| `eval_retrieval.py` | hit@k and MRR. Retrieval only, zero API cost. `--dense-only` to A/B. |
| `diagnose_retrieval.py` | Why one query fails — four checks, one verdict |
| `inspect_margins.py` | Where a PDF prints its folio, as a fraction of page height |
| `comic2pdf.py` | CBR/CBZ archives to a page-aligned PDF |
| `inspect_cbr.py` | Images inside a comic archive, with dimensions |
| `build_summaries.py` | Pre-build the summary tree |
| `build_manga.py` | Vision ingest. Resumable, cached per page. |
| `list_models.py` | Which models your API key can actually reach |

---

## Configuration

Everything is read from `.env`; see `.env.example` for the full set.

| Setting | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `ollama` runs fully offline |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | 500 requests/day on the free tier; most alternatives allow 20 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local. 512-token window |
| `CHUNK_CHARS` | `900` | Stays inside the embedding window |
| `HYBRID_RETRIEVAL` | `true` | `false` reverts to dense only |
| `RETRIEVAL_K` | `12` | Chunks passed to the model |
| `LLM_RPM` / `RATE_LIMIT_SAFETY` | `15` / `0.7` | Paces at 70% of the stated ceiling |
| `VISION_BATCH_PAGES` | `1` | Batching cuts requests 4× but halves speaker attribution |
| `COMIC_CHARACTERS` | — | Cast list; narrows the label space for attribution |

---

## Engineering notes

Four bugs worth recording, because each one produced a system that **looked fine
and was not**.

### Embeddings that returned a constant vector

Every similarity distance came back exactly `0.0000`, and two unrelated queries
returned identical rankings in identical order. The model had loaded successfully
from a truncated cache and was returning the same vector for every input, so every
chunk was equidistant from every query and the ranking was insertion order. No
exception, no warning — the system produced fluent, page-cited answers throughout.

Caught by printing raw distances instead of trusting top-*k* output. Fixed by
clearing the HuggingFace cache.

### A token bucket sized to the rate

The bucket held 15 tokens for a 15 requests/minute limit. That is correct for
smoothing average load and wrong against a fixed provider window: a full bucket
fires all fifteen at once, spends the minute's quota in two seconds, and the
sixteenth request is rejected. **Capacity has to be 1.**

The same failure showed that the summary cache only persisted at the end of a run,
discarding sixteen completed calls out of a 500/day budget. It now saves after
every chapter.

### Margin bands assumed, not measured

Running heads were stripped with a fixed top-and-bottom 8% rule. Dracula prints its
folio at 93.6% of page height and was stripped by luck; Think Python prints it in
the running head at 7.7–9.0%, so it fell outside the band by one hundredth of a
page and leaked into the body as line 1 — shifting every line number on every page.

The tests passed, because they checked a line *existed*, not that it was the
*right* line. The first fix made it worse: calibrating from the maximum observed
folio position let one outlier page drag the cutoff to 45% of page height and
delete the top of all 244 pages. Use the median, and clamp it.

### A metric improving while correctness got worse

Comics rarely name the speaker in-panel, so the volume's cast list was supplied as
corpus metadata. Named attributions rose — but reading the output rather than the
counter showed it had started lifting names out of the dialogue: a line saying
*"THERE SHE IS, LORD UNUS!"* was attributed **to** Unus, who is being addressed
rather than speaking.

Fixed by making the balloon tail the only admissible evidence, with that exact
counter-example in the prompt.

---

## Known limitations

- **No summary tree for the comic.** It has no chapter boundaries — eight issues of
  continuous pages, no outline, no headings — so whole-volume questions fall back
  to wide retrieval. The conversion manifest already records issue-per-page, so
  treating each issue as a chapter would cost about nine API calls.
- **Speaker attribution sits near 80%.** Some of the remainder is genuinely
  unanswerable: balloons that point off-panel or at unnamed background figures. The
  prompt prefers an explicit `unknown` over a guess.
- **The router is regex-based.** Instant, free and predictable, but coverage is
  only as good as the phrasings anticipated. Four gaps surfaced within twenty
  minutes of running someone else's questions — plurals, ranges, comma-separated
  lists and filler words like "page number".
- **No prompt-injection defences.** The corpus is trusted — public-domain fiction,
  a CC-licensed textbook, a comic. For user-uploaded documents the retrieved
  content would need delimiting and declaring as data, and the vision output would
  need treating as untrusted, since it is model-generated text re-entering a prompt
  on a second hop.
- **Evaluation set is 66 questions.** Small, and some expected pages in the
  paraphrased half were inferred rather than verified line by line.

---

## Roadmap

- Cross-encoder reranking to improve ordering within the top 12
- Streaming responses — roughly 8× lower perceived latency
- Issue-as-chapter summary tree for comics
- Larger evaluation set with every expected page verified
- Delimiter-based prompt-injection boundary for untrusted corpora

---

## Source material

- **Think Python 2e** — Allen B. Downey, Green Tea Press. Creative Commons.
- **Dracula** — Bram Stoker, 1897. Public domain, via Project Gutenberg.
- **X-Men: Age of Apocalypse** — Marvel Comics, 1995. Used locally for evaluation
  only; no book files are committed to this repository.

`.gitignore` excludes `chroma_db/`, `page_index/` and `cache/` — they contain the
full extracted text of the source material.

---

## Documentation

Design documents are in [`docs/`](docs/):

| Document | Covers |
|---|---|
| [Architecture](docs/ARCHITECTURE.md) | How it works and why — routing, extraction, retrieval, quota handling |
| [Product Requirement Document](docs/Product%20Requirement%20Document.md) | Problem, goals, functional requirements, success criteria |
| [Technical Requirement Document](docs/Technical%20Requirement%20Document.md) | Stack, architecture, key technical decisions, testing |
| [Application Flow](docs/Application%20Flow.md) | Ingestion and query flows, route dispatch, failure handling |
| [UI UX Brief](docs/UI%20UX%20Brief.md) | Interaction model and observability surface |
| [Backend Schema](docs/Backend%20Schema.md) | Data structures, storage, caches, API contracts |
| [Implementation Plan](docs/Implementation%20Plan.md) | Sequencing, verification gates, what measurement changed |

---

## Credits

Built by **Abrash Ali** for the Zenvyro Labs internship.

Starter scaffolding — the frontend and the FastAPI skeleton — was provided by
Zenvyro Labs. The retrieval pipeline, extraction, routing, summarisation,
conversation memory, vision extraction, rate limiting and evaluation harness are
this project's work.

## License

MIT
