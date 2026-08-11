# 01 — Product Requirements Document

**Project** Multimodal RAG Studio
**Author** Abrash Ali — AI Engineering Intern
**Organisation** Zenvyro Labs
**Repository** github.com/abrashhalii/multimodal-rag-studio
**Status** Delivered

---

## 1. Overview

Multimodal RAG Studio is a retrieval-augmented question answering system that
operates over three structurally different books: a programming textbook, a
Victorian novel, and a comic volume that contains no machine-readable text.

A reader uploads a document and asks questions about it in natural language. The
system answers from the document alone, citing the printed page each claim comes
from. It handles four distinct kinds of question — exact location lookups,
single-page questions, whole-book questions, and open semantic questions — and
it does so identically regardless of whether the source was typeset text or
drawn artwork.

## 2. Problem statement

A language model knows its training data. It does not know the contents of a
specific book, and a book is too long to place in a prompt. Retrieval-augmented
generation solves this by fetching only the passages relevant to a question.

Standard implementations of that pattern fail in four specific ways, each of
which this project treats as a first-class requirement rather than an edge case.

**Location questions cannot be answered by similarity search.** "What is on page
34, line 5?" carries almost no semantic content. Embedding-based retrieval has
nothing to match against, so it returns arbitrary passages. Improving the
embedding model does not help, because the problem is not one of embedding
quality.

**Whole-book questions cannot be answered by sampling.** Retrieving the twelve
most similar passages tells you nothing about the other six hundred. Questions
that require surveying a work — which exercise is best, what are the main themes
— cannot be answered from a sample, at any sample size.

**Follow-up questions lose their referent.** "What happened to her?" contains no
retrievable signal. Storing conversation history and placing it in the prompt
does not fix this, because retrieval runs before generation.

**Image-only documents have no text to retrieve.** A comic page is a picture. No
amount of text-layer extraction produces anything.

## 3. Goals

| # | Goal | Measure of success |
|---|---|---|
| G1 | Answer exact location questions correctly | Line returned matches the printed page, verified by eye |
| G2 | Answer whole-book questions from complete coverage | Answer derives from every chapter, not a retrieved sample |
| G3 | Resolve follow-up questions correctly | Pronoun resolved before retrieval, observable in logs |
| G4 | Extract and answer from comic dialogue | Dialogue read from page images, page-cited |
| G5 | Operate within free-tier API limits | No quota rejections during normal operation |
| G6 | Make retrieval quality measurable | Reportable hit@k over a fixed question set |

## 4. Non-goals

- Multi-user accounts, authentication, or per-user document isolation
- Persistent conversation storage across sessions
- Model fine-tuning of any kind
- Editing, annotating or exporting the source documents
- Real-time collaborative use
- Defences against adversarial content in the corpus (see §8)

## 5. Users and use cases

**Primary user: a reader studying a specific book.** They want answers grounded
in that book, with page references they can check.

| Use case | Example question |
|---|---|
| Verify an exact location | "What does page 34, line 5 say?" |
| Understand one page | "What happens on page 132?" |
| Compare several pages | "Summarise pages 100 to 110" |
| Survey the whole work | "What are the main themes of this book?" |
| Understand a chapter | "Explain chapter 7" |
| Open question | "How does a while loop work?" |
| Follow up | "Who is Lucy?" then "What happened to her?" |
| Comic dialogue | "What does Magneto say about humans?" |

## 6. Functional requirements

### FR1 — Absolute precision

The system shall answer questions naming a page and line by returning that exact
line. Resolution shall use the **printed** page number as it appears on the page,
not the position of the page within the file, and the system shall report which
method it used to establish that mapping.

Location answers shall be produced without invoking a language model.

### FR2 — Global summarisation

The system shall answer questions about a work as a whole from a summary
structure derived from every page, built at ingestion rather than at query time.

### FR3 — Conversation memory

The system shall resolve references to earlier turns. Resolution shall occur
**before** retrieval, so that the corrected question drives the search. History
shall be maintained separately per document.

### FR4 — Multimodal extraction

The system shall extract dialogue from comic page images, classifying each item
as speech, thought, caption, or sound effect, and attributing a speaker where
visual evidence supports one. Extracted text shall be indexed in the same
structure as text-layer documents, so all other features apply unchanged.

### FR5 — Request pacing

The system shall pace outbound model requests below the provider's stated limit,
recover from quota rejections using the provider's own stated delay, and persist
completed work incrementally so an interruption does not discard it.

### FR6 — Observability

The system shall expose its own state: which documents are loaded, how each was
processed, and the current behaviour of the rate limiter.

## 7. Success criteria

| Criterion | Target | Achieved |
|---|---|---|
| Location answers verified against printed pages | Exact match | Verified on all three books |
| Whole-book coverage | Every chapter summarised | 19 + 28 chapters |
| Follow-up resolution | Observable in logs | Verified end to end |
| Comic extraction | Dialogue read from images | 3,121 items over 258 pages |
| Retrieval quality | Measured and reported | hit@12 = 43/45 |
| Quota rejections | Zero in normal operation | Zero across 307 calls |

## 8. Constraints

**API quota.** The free tier permits 500 requests per day and 15 per minute. This
constraint shaped the architecture directly: page-level summarisation would have
required 647 requests for two books and was therefore impossible, which is why
summarisation operates at chapter level.

**Local hardware.** Embeddings run locally on consumer hardware. Model choice was
bounded by that.

**Trusted corpus.** All source material is known and controlled. The system does
not defend against instructions embedded in document content. This is a stated
limitation rather than an oversight; §5 of the TRD describes what would change
for untrusted input.

## 9. Out of scope

Reranking of retrieved passages, response streaming, summary structures for
documents without chapter boundaries, and an expanded evaluation set are all
identified improvements deliberately deferred. They are recorded in the
implementation plan rather than abandoned.
