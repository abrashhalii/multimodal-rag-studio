# 06 — Implementation Plan

**Project** Multimodal RAG Studio
**Author** Abrash Ali — AI Engineering Intern
**Organisation** Zenvyro Labs
**Repository** github.com/abrashhalii/multimodal-rag-studio
**Status** Delivered

---

## 1. Sequencing rationale

The four requirements were not independent, and the order they were built in
followed from that.

Absolute precision came first because everything else depends on the page index it
produces. Global summarisation came second because it needs chapter boundaries,
which the same extraction step provides. Conversation memory came third because it
modifies retrieval, and retrieval had to be correct before it could be modified.
Vision extraction came last, deliberately: it is the largest single item, it
consumes the most API quota, and it needed the page-and-line infrastructure
already working so that the extracted dialogue could plug into it unchanged.

Hardening was scheduled after all four, on the principle that a change made on the
final day can only lose points that already exist.

## 2. Delivery

### Phase 1 — Foundations and absolute precision

Environment, configuration through `.env`, and a full audit of the supplied
scaffolding before writing anything.

That audit found two defects not listed in the brief. One was fatal: the query
function referenced an undefined variable, so no question could execute at all.
The other was architectural: all three document types were written into a single
vector collection, and the document-type parameter was accepted and then never
used — so the novel would answer from the textbook, and re-uploading a document
silently duplicated it.

Then the extraction engine: line extraction with geometry, printed page
resolution, page-bounded chunking, and the query router.

*Exit criterion — a page-and-line query returns the correct line, verified against
the printed page.*

### Phase 2 — Summarisation and memory

Chapter detection, the summary tree, query rewriting, conversation history end to
end, and the rate limiter that the summary build required.

*Exit criterion — a whole-book question answers from every chapter, and a
pronoun-only follow-up resolves before retrieval.*

### Phase 3 — Multimodal extraction

Comic archive conversion, page rendering, batched and single-page vision
extraction, structured output parsing, and content-hash caching.

*Exit criterion — dialogue extracted from 258 image pages, page-cited, with the
precision route working on the comic identically to the text books.*

### Phase 4 — Hardening

Telemetry, the retrieval evaluation harness, and hybrid retrieval — added only
after measurement showed it was needed.

*Exit criterion — retrieval quality measured, improved, and the improvement
verified against the same question set.*

## 3. Verification gates

Each phase had an automated gate that ran without an API key, so it could be run
continuously rather than sparingly.

| Gate | Checks | Outcome |
|---|---|---|
| Extraction | 23 | Passing on both text documents |
| Summarisation and memory | 25 | Passing on both text documents |
| Retrieval quality | 66 questions | hit@12 = 43/45 |

The design principle behind all three: **a test that costs money is a test that
does not get run.**

## 4. Risks and how they were handled

| Risk | Mitigation | Outcome |
|---|---|---|
| API quota exhaustion | Chapter-level summarisation; content-hash caching; incremental persistence | 307 calls, no rejections |
| Silent extraction errors | Automated gates asserting correctness, not just presence | Three silent failures caught |
| Provider unavailability | Local model behind one configuration value | Verified working |
| Late changes breaking working features | Release tagged before each change; every enhancement reversible by configuration | No regressions |
| Losing expensive artifacts | Caches backed up outside the project | Recoverable |

## 5. What measurement changed

Three decisions were altered by evidence rather than reasoning, and all three
would have gone the other way without it.

**Vision batching.** Sending four pages per request reduces API calls roughly
fourfold at negligible token cost, and was the obvious choice under a quota
constraint. Measurement showed text extraction was unaffected but speaker
attribution fell by more than half — batching loses the fine-grained visual work
of tracing a balloon tail to a figure, not the text. Since request count was not
actually the binding constraint, the accuracy was worth the cost.

**Hybrid retrieval.** Scheduled as an optional enhancement. Evaluation showed
dense retrieval failing to find pages when queried with text copied verbatim from
them, so it became a fix rather than an improvement. Correct page reaching the
model rose from 32 of 45 to 43 of 45.

**Speaker attribution.** Supplying the volume's cast list raised the count of
named speakers. Reading the output rather than the counter showed it had begun
lifting names out of the dialogue and attributing lines to the character being
addressed. The metric had improved while correctness got worse.

## 6. Late corrections

Running the evaluation questions supplied by the assessor — rather than my own —
exposed four gaps in the question classifier within twenty minutes. Page ranges,
comma-separated page lists, plural forms, and filler words such as "page number"
all fell through to general search instead of being treated as location queries.

A separate correction was needed for questions asking the system to write code.
The answer instructions required grounding in the document and prohibited outside
knowledge, and the model had reasonably concluded that writing a solution
constitutes outside knowledge. The line was redrawn: the **specification** must
come from the document, not the solution.

The common cause in every case is the same, and it is worth recording. The
patterns were written for how I would phrase a question rather than how a reader
does. That is the standing trade-off with a rule-based classifier — instant, free
and entirely predictable, but its coverage is exactly as good as the range of
phrasings anticipated.

## 7. Remaining work

| Item | Value | Cost |
|---|---|---|
| Response streaming | Large — roughly 8× lower perceived latency | Low |
| Cross-encoder reranking | Improves ordering within the retrieved set | Moderate |
| Summary tree for image documents | Enables whole-volume questions on the comic | ~9 API calls |
| Expanded evaluation set | Higher confidence in the reported figures | Moderate |
| Prompt-injection boundary | Required for untrusted documents | Low |
| Screenshots in the repository | Presentation | Low |

## 8. Retrospective

Four principles were adopted partway through and materially changed the work:
think before coding, prefer the simpler construction, keep changes surgical, and
define a verifiable success criterion before starting.

The fourth is the one that mattered. Three separate times the system produced
fluent, well-formatted, correctly-cited answers while something underneath was
completely broken — embeddings returning an identical vector for every input, line
numbers shifted across every page, a quality metric improving while the results
degraded. None of those were visible in the output. Every one was caught by
measuring something rather than reading the results.

The transferable conclusion is narrow and holds generally: **in a retrieval
system, fluency is not evidence of correctness.** The output of a broken pipeline
and the output of a working one look the same. Only measurement distinguishes
them.
