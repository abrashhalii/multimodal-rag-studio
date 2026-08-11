# 05 — Backend Schema

**Project** Multimodal RAG Studio
**Author** Abrash Ali — AI Engineering Intern
**Organisation** Zenvyro Labs
**Repository** github.com/abrashhalii/multimodal-rag-studio
**Status** Delivered

---

## 1. Core data model

Everything in the system reduces to one structure. A document is a list of pages;
a page is a printed number and a list of numbered lines. Text-layer extraction and
vision extraction both produce exactly this, which is why no downstream component
needs to know which produced its input.

### 1.1 `PageRecord`

| Field | Type | Meaning |
|---|---|---|
| `pdf_index` | int | Zero-based position within the file |
| `printed_page` | str | The number printed on the page — `"34"`, `"iv"` |
| `lines` | list[str] | Body text, margin furniture removed, 1-based when addressed |
| `header` | str, optional | Running head, if one was found |
| `footer` | str, optional | Footer content, if any |
| `folio` | str, optional | The bare page number located in the margin |
| `is_front_matter` | bool | True where the printed number is not arabic |
| `is_index` | bool | True for back-of-book index pages |
| `chapter_id` | int | Index into the document's chapter list, `-1` if none |
| `chapter_title` | str, optional | Chapter this page belongs to |
| `source` | str | `text` or `vision` |

`printed_page` is a string rather than an integer because front matter is numbered
in roman numerals. Storing it as text avoids a lossy conversion.

### 1.2 `BookIndex`

| Field | Type | Meaning |
|---|---|---|
| `book_type` | str | `coding`, `novel` or `manga` |
| `filename` | str | Source file |
| `page_count` | int | Pages indexed |
| `label_method` | str | `pdf_page_labels`, `footer_regex_offset`, `sequential_image_pages` |
| `offset` | int | `pdf_index = printed_page + offset` |
| `margin_note` | str | Whether margin bands were calibrated or defaulted |
| `chapters` | list[dict] | `{id, title, start, end}` by file position |
| `chapter_method` | str | `pdf_outline`, `heading_regex` or `none` |
| `pages` | list[PageRecord] | The pages themselves |

Lookups provided: by printed page, by file position, by chapter, and by line
within a page. Chapter lookup resolves arabic and roman numerals
interchangeably.

Persisted as JSON at `page_index/<book_type>.json`.

## 2. Chunk schema

Chunks are line-aligned and never cross a page boundary.

### 2.1 Content

The chunk text is the body **only**. The location header is stored in metadata and
re-attached at retrieval time.

Embedding an identical header into every chunk pulls all vectors toward a common
centroid, degrading separation most severely on short chunks. But metadata alone
never reaches the model's context window. Storing and re-attaching satisfies both.

### 2.2 Metadata

| Key | Type | Purpose |
|---|---|---|
| `location_header` | str | Re-attached at retrieval so the model can cite |
| `book_type` | str | Redundant with the collection; useful in diagnostics |
| `filename` | str | Provenance |
| `pdf_index` | int | Position in the file |
| `pdf_viewer_page` | int | One-based, matches what a PDF reader displays |
| `printed_page` | str | The number on the page |
| `printed_page_num` | int | Numeric form, `-1` for front matter, for range filters |
| `line_start` / `line_end` | int | Line span within the page |
| `is_front_matter` | bool | Filterable |
| `is_index` | bool | Excluded from semantic retrieval |
| `chapter_id` | int | Enables chapter-scoped retrieval |
| `chapter_title` | str | Display |
| `source` | str | `text` or `vision` |

The location header format:

```
[Source: PDF Viewer Page 39 | Printed Page 34 | Lines 1-14 | novel]
```

Both page numbers appear because they differ, and because the reader needs the
printed one while a person checking the file needs the other.

## 3. Storage layout

```
backend/
├── chroma_db/                    vector store, one collection per document type
├── page_index/
│   ├── coding.json               BookIndex, serialised
│   ├── novel.json
│   └── manga.json
├── cache/
│   ├── summaries_coding.json     chapter and book summaries
│   ├── summaries_novel.json
│   └── vision_manga.json         extracted comic dialogue
└── temp/                         uploaded files
```

Collections are named `book_<book_type>`. Separation is physical rather than by
filter: a missing filter clause would leak one document's content into another's
answers, and that failure is invisible in the output.

All four directories are excluded from version control. They contain the full text
of the source material.

## 4. Cache schemas

### 4.1 Summary cache

```json
{
  "chapters": {
    "0": {
      "hash": "9f2c1a…",
      "title": "Chapter I. Jonathan Harker's Journal",
      "pages": "1-14",
      "summary": "…"
    }
  },
  "book": { "hash": "4d81e0…", "summary": "…" }
}
```

Keyed by SHA-256 of the source text. Written after **every** chapter — an
interruption must not discard completed API calls, which at 500 requests per day
is a meaningful fraction of the budget.

### 4.2 Vision cache

```json
{
  "<sha256 of rendered image>": {
    "pdf_index": 129,
    "items": [
      {"panel": 1, "type": "caption", "speaker": "",         "text": "MEANWHILE…"},
      {"panel": 1, "type": "speech",  "speaker": "Magneto",  "text": "We must strike now."},
      {"panel": 2, "type": "sfx",     "speaker": "",         "text": "KRAKOOM"}
    ]
  }
}
```

Keyed by the rendered image rather than by page number, so re-ingesting an
identical document costs nothing regardless of ordering.

Four text types: `speech`, `thought`, `caption`, `sfx`. Sound effects are stored
but excluded from semantic indexing — they match on phonetics rather than meaning.

## 5. Line rendering from vision output

Extracted items become numbered lines, which is what allows the precision route to
work on a comic:

```
1  [caption] MEANWHILE, IN LATVERIA…
2  Magneto: We must strike now.
3  Rogue (thinking): I cannot touch him.
4  [sfx] KRAKOOM
```

## 6. Route model

| Field | Type | Meaning |
|---|---|---|
| `kind` | str | `page_line`, `page`, `pages`, `chapter`, `global`, `memory`, `semantic` |
| `page` | int, optional | Single page reference |
| `pages` | list[int], optional | Range or explicit list |
| `line` | int, optional | Line within a page |
| `chapter` | str, optional | Chapter identifier as written |
| `reason` | str | Why this route was chosen — logged, not shown |

## 7. API contracts

### `POST /api/upload`

```
multipart:  file, book_type
returns:    { "message": "Processed <file> as <type>. Indexed N pages
                          into M chunks (<method>, offset K)." }
```

### `POST /api/chat`

```json
{
  "message": "What happened to her?",
  "book_type": "novel",
  "history": [
    { "role": "user",      "content": "Who is Lucy Westenra?" },
    { "role": "assistant", "content": "Lucy is Mina's friend…" }
  ]
}
```

```json
{ "answer": "…" }
```

History is optional, so an older client or a direct request still works.

### `GET /api/status`

Returns the active models, the rate limiter's live state, effective configuration,
and per-document detail: pages, pages carrying text, page-numbering method, margin
band treatment, offset rule, chapter count and detection method, extraction path,
and whether a summary tree exists.

## 8. Retrieval filters

| Filter | Applied to | Reason |
|---|---|---|
| `is_index = false` | All semantic retrieval | Index rows are dense keyword lists — strong matches, useless answers |
| `chapter_id = N` | Chapter-scoped queries | Confines retrieval to the named chapter |

Front matter is deliberately **not** excluded: the table of contents is often the
most useful chunk in a book for structural questions.

ChromaDB requires `$and` for multiple conditions; a two-key dictionary is
rejected.
