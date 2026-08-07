# Day 1 — Foundations + Bug 1 (Absolute Precision)

Target: by tonight, `"What does page 34 line 5 say?"` returns the exact correct
line, and you can prove it against the printed PDF.

---

## Step 0 — Layout

Replace the starter `backend/` folder with this one. The `frontend/` folder is
unchanged today (it changes on Day 2 for Bug 3).

```
project/
├── backend/
│   ├── config.py          # all settings via .env
│   ├── page_index.py      # page + line extraction, printed-page resolution
│   ├── ingest.py          # page-bounded chunking, metadata injection, Chroma
│   ├── router.py          # query intent classification
│   ├── llm_provider.py    # Gemini / Ollama switch
│   ├── rag_engine.py      # orchestration
│   ├── main.py            # FastAPI
│   ├── requirements.txt
│   ├── .env.example
│   ├── .gitignore
│   └── tools/
│       ├── smoke_test.py  # Day 1 acceptance test
│       └── list_models.py # verify Gemini model names
└── frontend/              # unchanged today
```

---

## Step 1 — Environment (20 min)

```bash
cd project/backend
py -3.11 -m venv venv
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

First install pulls torch + sentence-transformers, ~2 GB. Start it now and read
the rest of this while it runs.

```bash
copy .env.example .env
```

Open `.env` and set `GEMINI_API_KEY`. Get one free at
`https://aistudio.google.com/apikey`.

---

## Step 2 — Confirm your model names (5 min)

Model names change between releases and a wrong one is a 404 that looks like a
code bug. Do not guess:

```bash
pip install google-generativeai
python tools/list_models.py
```

Copy a Flash model into `GEMINI_MODEL` and a Flash-Lite into `GEMINI_FAST_MODEL`.
While you are in AI Studio, **write down your actual RPM / RPD / images-per-minute
limits** — you need those numbers for the Day 4 rate limiter and the interview.

---

## Step 3 — The two bugs the README never mentions

Both are already fixed in this code. You need to be able to explain them, because
"I found two bugs you didn't document" is a strong opening in the interview.

### 3a. `NameError` on every single query

Original `rag_engine.py`, last line:

```python
def query_rag_system(user_message: str, book_type: str = "coding") -> str:
    ...
    return rag_chain.invoke(user_question)   # user_question is not defined
```

Nothing worked at all until this was fixed.

### 3b. All three books shared one collection

```python
vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings,
                                  persist_directory=DB_DIR)
```

`from_documents` **appends**. The frontend sends `book_type`, `main.py` forwards
it, and `query_rag_system` accepted it and never used it. So the Novel tab
retrieved Think Python chunks, and re-uploading a book silently doubled it.

Fixed with one collection per book type plus a reset before re-ingest:

```python
def get_vector_db(book_type):
    return Chroma(collection_name=f"book_{book_type}", ...)

def reset_collection(book_type):
    get_vector_db(book_type).delete_collection()
```

---

## Step 4 — Understand the Bug 1 fix before you demo it

Three pieces, and the mentor will probe all three.

**1. Printed page ≠ PDF index.** Front matter shifts them. In our Dracula build,
printed page 34 is PDF page 39 (index 38). `page_index.py` resolves the printed
number two independent ways:

- `/PageLabels` from the PDF catalog (structured, exact)
- footer/header regex, taking the most common `index − folio` offset

Watch out: `pypdf.page_labels` **synthesises** sequential labels when the catalog
has no `/PageLabels` entry — it never returns empty. Trusting it blindly means a
label-less PDF silently reports `printed == index + 1`, which is the exact bug
you are trying to fix. We check for the catalog key before believing the labels.
That subtlety is a great thing to mention live.

**2. Header and footer are stripped by geometry, not guesswork.** Lines in the
top 8% / bottom 8% of the page are margin furniture. Strip them, and line 1 is
the first real line of prose. Leave them in and every line number is off by one.

**3. Location goes into `page_content`, not just metadata.** Chroma metadata
never reaches the LLM — only the text does:

```
[Source: PDF Viewer Page 39 | Printed Page 34 | Lines 1-14 | novel]
"Then write now, my young friend," he said, laying a heavy hand on
...
```

**And the part that actually wins the points:** a page+line query never touches
the vector store. `router.py` catches it and `page_index.py` answers it directly
from the line index — no embeddings, no LLM, no hallucination possible. Say this
sentence out loud in the interview:

> A page-number query is a database lookup wearing a question's clothing.
> Better embeddings cannot fix vector blindness, because the query carries no
> semantic signal to match. The fix is routing, not retrieval.

---

## Step 5 — Run the smoke test (10 min)

No API key needed — everything Bug 1 depends on is verifiable offline.

```bash
python tools/smoke_test.py "D:\path\to\Dracula_Bram_Stoker.pdf" novel
```

Expect **20 passed, 0 failed**, and in the report:

```
page numbering  : pdf_page_labels
offset          : printed page N  ->  pdf index N+4 (0-based)
PASS  page 34 line 5   'the thought.'
```

Then run it on Think Python:

```bash
python tools/smoke_test.py "D:\path\to\thinkpython2.pdf" coding
```

Think Python has no `/PageLabels`, so it should report `footer_regex_offset`.
**Getting two different resolution methods on two different books is the single
best thing you can show live** — it proves the system adapts rather than
hardcoding one publisher's convention.

If it reports `fallback_sequential`, no folios were found: check whether the PDF
actually prints page numbers, and widen `HEADER_ZONE` / `FOOTER_ZONE` in `.env`.

---

## Step 6 — Start the server and ingest for real (30 min)

```bash
python main.py
```

Open `http://localhost:8000`, pick the **Novel** tab, upload the Dracula PDF.
Ingest is ~400 pages; first run also downloads the embedding model.

Check `http://localhost:8000/api/status` — it reports which book is loaded in
which mode, the resolution method, and the offset. Useful during the demo.

---

## Step 7 — Verify against the printed page (15 min)

Ask, in the Novel tab:

| Question | Expected |
|---|---|
| `What does page 34 line 5 say?` | `the thought.` — instant, no LLM call |
| `read me line 12 of page 200` | exact line, cites viewer page 204 |
| `What happened on page 23?` | LLM summarises page 23 only |
| `Who is Van Helsing?` | semantic answer citing printed pages |

Open the PDF, go to printed page 34, count five lines down. It must match
character for character. **Do not accept "close enough" today** — every later
feature sits on this foundation.

Cross-check a few more against `Dracula_ground_truth.json` if you want a second
opinion. That file is a test fixture only — never wire it into the RAG itself,
because you will not have one for Think Python or the Marvel volume.

---

## Step 8 — Commit

```bash
git init
git add .
git commit -m "Day 1: page-precise indexing, query routing, per-book collections"
```

`.env` is gitignored. `.env.example` is committed.

---

## Done when

- [ ] `smoke_test.py` passes on Dracula **and** Think Python
- [ ] the two books resolve page numbers by *different* methods
- [ ] `page 34 line 5` matches the printed PDF exactly
- [ ] Novel tab and Coding tab return content from the right book
- [ ] re-uploading the same book does not duplicate chunks
- [ ] committed

---

## Tomorrow (Day 2)

Bug 2 (global summarisation via a hierarchical summary tree built at ingest) and
Bug 3 (chat history end to end, plus history-aware query rewriting — the piece
that makes memory actually work rather than merely look like it works).
