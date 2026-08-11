# Architecture

**Project** Multimodal RAG Studio
**Author** Abrash Ali — AI Engineering Intern
**Organisation** Zenvyro Labs
**Repository** github.com/abrashhalii/multimodal-rag-studio

A narrative walkthrough of how the system works and why it is built this way.
For requirements see the [Product Requirements Document](Product%20Requirement%20Document.md);
for the specification see the [Technical Requirements Document](Technical%20Requirement%20Document.md).

---

## 1. The problem this system exists to solve

A language model knows its training data. It does not know the contents of a
particular book, and a book is too long to place in a prompt — a 400-page novel is
roughly 200,000 tokens, and even where that fits, attention thins across it.

Retrieval-augmented generation fetches only the passages relevant to a question.
The analogy is an open-book exam: the student does not memorise the textbook, they
flip to the right pages and answer from them. Retrieval is the flipping.

Ingestion splits each book into chunks and converts them to vectors — 384 numbers
each. Text with similar meaning produces nearby vectors, so a question can find
its answer by proximity rather than by keyword. Proximity is measured by cosine
similarity: the angle between two vectors, where 1.0 means near-identical meaning
and 0.0 means unrelated.

That is the standard pattern. It fails in four specific ways, and this system
treats each as a first-class problem rather than an edge case.

## 2. The idea that holds the system together

Three books enter this system. One is a programming textbook full of code
listings. One is a Victorian novel of homogeneous first-person prose. One is a
258-page comic volume containing **no machine-readable text at all**.

All three reduce to the same structure:

> A list of pages. Each page has a printed page number and a list of numbered
> lines.

A PDF with a text layer produces that by reading the text layer, with margin
furniture stripped by vertical position. A comic produces it by sending each page
image to a vision model and turning every speech balloon into a line.

Once both look identical, **every feature works on every book with no special
casing**. The question `page 130, line 1` runs the same code whether the answer
came from typeset prose or from a drawing of a character shouting.

This is the single most consequential decision in the system. Everything else
follows from it.

## 3. Vector blindness, and why routing rather than retrieval

`What is on page 34, line 5?` carries almost no semantic content. A page number
has no meaning for cosine similarity to match against. Embedding-based retrieval
returns arbitrary passages, and — this is the important part — **a better
embedding model does not help**, because the problem is not one of embedding
quality.

The fix is to classify the question before retrieval runs. A regular expression
inspects the query, and if it names a page and a line, the query never touches the
vector store. It becomes a dictionary lookup against an index built at ingestion.

Consequences worth stating:

- It cannot hallucinate — no model is involved
- It cannot be rate limited — no API call
- It returns in about a millisecond

The routing table has six destinations:

| Intent | Path |
|---|---|
| Page and line | Deterministic index lookup, no model |
| Single page | Exact fetch, then the model reads that page |
| Multiple pages | Range or list fetched by number, passed together |
| Chapter | Chapter summary plus retrieval scoped to that chapter |
| Global | Summary tree — derived from every page |
| Conversation | Answered from the transcript itself |
| Anything else | Hybrid retrieval |

The router is deliberately regex-based rather than model-based: instant, free,
deterministic, and testable. The trade-off is that its coverage is exactly as good
as the range of phrasings anticipated — see §9.

## 4. Printed page numbers are not file positions

A PDF's internal page index is not the number printed on the page. Front matter
displaces them: in the novel, printed page 34 is the 39th page of the file.

Resolution proceeds in order of reliability.

**PDF page labels.** Read from the document catalog where present. One trap: the
pypdf accessor **synthesises** sequential labels when the catalog has no such
entry, and never returns empty. Trusting it unconditionally silently reports
`printed = index + 1`, which is precisely the bug the module exists to prevent. The
code checks for the catalog key before believing the labels.

**Folio regex.** Locate the bare page number in the margin, take the modal offset
between file position and printed number.

**Sequential position.** For image-only documents where no folio exists. Recorded
explicitly as the method used, so the assumption is visible rather than implied.

### Margin calibration

Header and footer removal cannot use a fixed proportion of page height. The two
text books differ:

| Book | Folio position | Consequence of a fixed 8% band |
|---|---|---|
| Novel | 93.6% of page height | Stripped correctly, by luck |
| Textbook | 7.7–9.0%, in the running head | **Leaked into the body as line 1** |

The textbook's folio fell outside the band by one hundredth of a page height. The
boundary test compares the line's lower edge, so `0.090` against a `0.080` cutoff
failed — and every line number on every page shifted by one.

So the extractor locates every bare folio in the document, takes the **median** of
their vertical positions, and sets the band to just clear them. The median rather
than the maximum is essential: a single anomalous page whose last line happens to
be a bare number in the upper half will otherwise drag the cutoff to 45% of page
height and delete the top of every page. Hard clamps bound the result regardless.

## 5. Chunking, and where location metadata lives

Chunks are line-aligned and **never cross a page boundary**. A chunk spanning two
pages makes page attribution unprovable, which would undermine the entire
precision requirement.

This is also why semantic chunking was rejected. It places boundaries at meaning
transitions, which ignore page boundaries entirely — fashionable, and wrong for
this problem.

Location metadata presents a genuine tension. Every chunk needs a header like:

```
[Source: PDF Viewer Page 39 | Printed Page 34 | Lines 1-14 | novel]
```

Embedding that into the chunk text pulls every vector toward a common centroid,
degrading separation — the effect is worst on short chunks, where the identical
header is a large fraction of the content. But metadata alone never reaches the
model's context window; only text does.

The resolution: store the header in metadata, and re-attach it to the text at
retrieval time. Both constraints satisfied.

## 6. Whole-book questions cannot be sampled

Retrieving the twelve most similar passages tells you nothing about the other six
hundred. Ranking the best exercise in a textbook requires having read every
exercise. **No value of k fixes a survey problem** — top-k samples a book, it does
not survey it.

The fix is a hierarchical summary tree, built at ingestion:

```
chapter text  →  chapter summary     (one call per chapter)
all summaries →  whole-book overview (one call)
```

Global questions route to the tree, so the answer derives from every page.

### Why chapter level and not page level

Budget. Page-level summaries would cost 244 calls for the textbook plus 403 for
the novel — 647 against a 500-per-day quota. The tree could never finish. Chapter
level costs approximately 48.

The rate limit did not merely constrain the implementation. It selected the right
granularity: chapters are the unit readers ask about anyway.

Chapter boundaries come from the PDF outline where present, from heading patterns
otherwise. The two books happen to exercise both paths, which is useful evidence
that neither is hardcoded.

## 7. Memory has to change retrieval, not just the prompt

`What happened to her?` carries almost no retrievable signal. There is no "her" in
the vector space.

Storing conversation history and placing it in the prompt makes a model *sound*
like it remembers. It does not make memory work, because **retrieval runs before
generation**. By the time the transcript reaches the model, the wrong passages
have already been fetched.

So the follow-up is rewritten into a standalone question first, and the rewritten
form drives retrieval. The original transcript still reaches the model separately,
for tone and continuity:

```
[rewrite] 'What happened to her?' -> 'What happened to Lucy Westenra?'
```

Two economies, because the daily budget is finite. Rewriting fires only when the
question looks dependent — pronouns, continuation words, very short queries — and
it is mechanical work, so it does not need the strong model.

History is tracked per document. Carrying one book's turns into another would
corrupt the rewriting: a pronoun would resolve to a character from the wrong book,
and the resulting search would be confidently wrong.

## 8. Reading a book that has no text

Comic pages are rendered to JPEG at 1400px on the longest edge and sent to a
vision model, which returns structured JSON: panel, text type, speaker, text.

Four text types, and they are not interchangeable:

| Type | Identified by |
|---|---|
| `speech` | Rounded balloon with a tail |
| `thought` | Cloud-shaped balloon |
| `caption` | Rectangular box, **no tail** |
| `sfx` | Lettering drawn into the artwork |

Classification is by **container, not content** — if it has no tail it is
narration, not dialogue. That rule was arrived at after a looser prompt filed a
caption box as speech and missed a large sound effect entirely. Sound effects need
calling out explicitly, because they are drawn as art rather than placed on the
page, and a model scanning for balloons does not register them as text.

Sound effects are stored but excluded from semantic indexing: they match on
phonetics rather than meaning. They remain answerable through the deterministic
route.

### Batching, and why it was rejected

Sending four pages per request reduces API calls roughly fourfold at negligible
token cost — the binding free-tier constraint is requests per minute, not tokens,
and a downscaled page is only ~1,300 tokens against a 250,000/minute ceiling.

It was implemented, then measured on the same eight pages:

| | Batch of 4 | Single page |
|---|---|---|
| Text items extracted | 74 | 75 |
| Speakers named | 19% | **46%** |
| Distinct characters found | 2 | **4** |

Batching does not lose text. It loses the fine-grained visual work of tracing a
balloon tail to a figure and recognising who that figure is. Since request count
was not actually the binding constraint, single-page requests were chosen.

Every page is cached by the SHA-256 of its rendered image, written after each
page. The operation is therefore resumable — an interruption costs one page — and
re-runs cost nothing, which is what makes prompt iteration affordable.

### Speaker attribution

Comics rarely name the speaker in-panel; a human reader recognises the costume. So
the volume's cast list is supplied as corpus metadata to narrow the label space.

That change raised the count of named speakers and **made correctness worse**. The
model began lifting names out of the dialogue: a line reading *"THERE SHE IS, LORD
UNUS!"* was attributed **to** Unus, who is being addressed rather than speaking.

The fix makes the balloon tail the only admissible evidence, with that exact
counter-example written into the prompt. Final result: 3,121 text items across 258
pages, with roughly 80% of dialogue attributed to a named character.

A high rate of `unknown` is partly **correct**. Many balloons genuinely point
off-panel or at unnamed background figures, and the prompt prefers an explicit
`unknown` to a guess.

## 9. Retrieval: two instruments with non-overlapping failures

Dense retrieval was measured before being trusted. Querying with a 14-word phrase
copied verbatim from a page should find that page nearly every time — it is the
easiest possible query. Dense-only managed the correct page reaching the model on
32 of 45 questions.

The failures clustered in the novel, and the reason is structural. A bi-encoder
maps text to a single vector. When 1,200 chunks are all anxious first-person
narration about illness and dread, those vectors sit close together and cosine
similarity cannot separate them. Meanwhile the exact phrase — the strongest signal
a query could carry — is **invisible** to it, because embeddings encode meaning and
discard tokens.

BM25 is the complementary instrument: it scores rare terms heavily and matches
phrases exactly. Proper nouns, code identifiers and quoted dialogue are precisely
what it is good at, and precisely where dense retrieval blurs.

Their failure modes do not overlap, which is what makes combining them worthwhile
rather than redundant.

### Fusion by rank, not by score

Cosine distance and BM25 relevance are not comparable quantities, and normalising
them is fragile because their ranges shift per query. Reciprocal rank fusion uses
only position:

```
score(document) = Σ over retrievers of 1 / (60 + rank)
```

No normalisation, no tuning, no assumption about score distribution.

### Measured result

45 verbatim-recall questions, fixed seed, one variable changed:

| | Dense only | Hybrid |
|---|---|---|
| Correct page ranked 1st | 16 / 45 | **28 / 45** |
| Correct page in top 5 | 26 / 45 | **38 / 45** |
| **Correct page reaches the model** | **32 / 45** | **43 / 45** |
| MRR | 0.465 | **0.737** |

`hit@12` is the operational metric because twelve chunks are passed to the model.
Below that, ordering is largely cosmetic — the model reads all twelve.

On paraphrased questions the improvement is marginal, which is expected rather
than disappointing: a reworded question shares no tokens with the passage, so BM25
has nothing to exploit.

## 10. Operating inside a hard quota

The free tier permits 500 requests per day and 15 per minute. That constraint
shaped the architecture more than any other single factor.

### The token bucket capacity bug

The bucket initially held 15 tokens for a 15-per-minute limit. That is correct for
smoothing average load and **wrong against a fixed provider window**: a full bucket
fires all fifteen at once, spends the entire minute's quota in about two seconds,
and the sixteenth request is rejected with a 39-second penalty.

Capacity has to be **1** — one request every 60/RPM seconds, no burst. The
effective rate also runs below the stated ceiling, because the provider's window
is not aligned with the client's, so pacing exactly at the limit still collides at
boundaries.

Retries parse the delay the provider itself supplies rather than guessing at
exponential backoff — the difference between recovering in 26 seconds and
hammering a closed door.

### Incremental persistence

The same failure revealed that the summary cache only persisted at the end of a
run, so a crash on the seventeenth chapter discarded sixteen completed API calls.
At 500 requests per day that is a meaningful fraction of the budget. Everything
now saves after each unit of work.

### Measured behaviour

| Run | Calls | Duration | Throttled | Waited | Rejections |
|---|---|---|---|---|---|
| Novel summaries | 29 | 174 s | 119× | 112.7 s | 0 |
| Textbook summaries | 20 | 119 s | 193× | 77.3 s | 0 |
| Comic vision ingest | 258 | ~20 min | 465× | 516.7 s | 0 |

## 11. Silent failure is the real enemy

Three times during development the system produced fluent, well-formatted,
correctly-cited answers while something underneath was completely broken. None of
it was visible in the output.

**Embeddings returning a constant vector.** Every similarity distance came back
exactly `0.0000`, and two unrelated queries returned identical rankings in
identical order. The model had loaded "successfully" from a truncated cache and
was returning the same vector for every input, so every chunk was equidistant from
every query and the ranking was insertion order. No exception, no warning. Caught
only by printing raw distances instead of trusting the top-k output.

**Line numbers shifted across every page.** The margin band bug in §4. The test
suite passed throughout, because it checked that a line *existed*, not that it was
the *right* line.

**A metric improving while correctness degraded.** The speaker attribution
regression in §8, caught by reading the output rather than the counter.

The transferable conclusion is narrow and holds generally: **in a retrieval
system, fluency is not evidence of correctness.** The output of a broken pipeline
and the output of a working one look the same. Only measurement distinguishes
them — which is why every test suite in this project runs without an API key, on
the principle that a test which costs money is a test that does not get run.

## 12. Known limitations

- **No summary tree for the comic.** It has no chapter boundaries — eight issues
  of continuous pages, no outline, no headings — so whole-volume questions fall
  back to wide retrieval. The conversion manifest records issue-per-page, so
  treating each issue as a chapter would cost roughly nine API calls.
- **Speaker attribution near 80%**, with some of the remainder genuinely
  unanswerable.
- **The router is regex-based.** Four coverage gaps surfaced within twenty minutes
  of running questions written by someone else — plurals, page ranges,
  comma-separated lists, and filler words such as "page number". The patterns had
  been written for how the author would phrase a question rather than how a reader
  does.
- **No prompt-injection defences.** The corpus is trusted. For user-uploaded
  documents the retrieved content would need delimiting and declaring as data, and
  the vision output would itself need treating as untrusted, since it is
  model-generated text re-entering a prompt on a second hop.
- **Evaluation set is 66 questions.** Small, and some expected pages in the
  paraphrased half were inferred rather than verified line by line.
