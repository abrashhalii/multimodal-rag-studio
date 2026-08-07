"""
Find out why a query is not retrieving the passage you know is in the book.

    python tools/diagnose_retrieval.py coding "how does a while loop work" "while statement"

  arg1 = book_type
  arg2 = the query as a user would type it
  arg3 = a literal string you KNOW appears in the correct passage

It answers four questions in order:
  1. Do chunks containing the literal string even exist in the collection?
  2. Are they excluded by a metadata filter (front matter / index)?
  3. Where do they rank for the user's query, and what beats them?
  4. Does the book's own phrasing retrieve them when the user's phrasing does not?

Question 4 is the important one. If the book's wording works and the user's
wording does not, the embeddings are fine and the problem is vocabulary
mismatch - which needs lexical search or query rewriting, not a better model.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config                     # noqa: E402
from ingest import get_vector_db, get_embeddings   # noqa: E402

NOT_JUNK = {"$and": [{"is_front_matter": {"$eq": False}},
                     {"is_index": {"$eq": False}}]}


def short(t, n=64):
    return t[:n].replace("\n", " ")


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    book_type, query, literal = sys.argv[1], sys.argv[2], sys.argv[3]

    db = get_vector_db(book_type)
    col = db._collection
    total = col.count()
    print(f"\ncollection : {config.collection_name(book_type)}  ({total} chunks)")
    print(f"query      : {query!r}")
    print(f"literal    : {literal!r}\n")

    # ---- 1. does the text exist at all? --------------------------------
    raw = col.get(include=["documents", "metadatas"])
    hits = [(m, d) for d, m in zip(raw["documents"], raw["metadatas"])
            if literal.lower() in (d or "").lower()]
    print(f"[1] chunks containing the literal string : {len(hits)}")
    if not hits:
        print("    -> the text is NOT in the collection. This is an INGEST problem,")
        print("       not a retrieval problem. Check extraction for those pages.")
        return
    for m, d in hits[:6]:
        flags = []
        if m.get("is_front_matter"):
            flags.append("FRONT_MATTER")
        if m.get("is_index"):
            flags.append("INDEX")
        print(f"    page {str(m.get('printed_page')):>5}  "
              f"lines {m.get('line_start')}-{m.get('line_end')}  "
              f"{'/'.join(flags) or 'ok':12} {short(d)!r}")

    # ---- 2. are they filtered out? -------------------------------------
    excluded = [m for m, _ in hits if m.get("is_front_matter") or m.get("is_index")]
    print(f"\n[2] of those, excluded by filters : {len(excluded)}/{len(hits)}")
    if len(excluded) == len(hits):
        print("    -> every relevant chunk is being filtered out. Loosen the filter.")

    target_pages = {str(m.get("printed_page")) for m, _ in hits
                    if not m.get("is_front_matter") and not m.get("is_index")}
    print(f"    target pages: {sorted(target_pages)[:12]}")

    # ---- 3. where do they rank for the user's query? --------------------
    print(f"\n[3] ranking for the USER's phrasing")
    scored = db.similarity_search_with_score(query, k=25, filter=NOT_JUNK)
    rank_of_target = None
    for i, (doc, dist) in enumerate(scored, 1):
        pg = str(doc.metadata.get("printed_page"))
        mark = "  <== TARGET" if pg in target_pages else ""
        if pg in target_pages and rank_of_target is None:
            rank_of_target = i
        if i <= 8 or mark:
            print(f"    {i:>2}. dist={dist:.4f} page {pg:>5}  {short(doc.page_content, 52)!r}{mark}")
    print(f"    first target page appears at rank: {rank_of_target or 'NOT IN TOP 25'}")

    # ---- 4. does the book's own phrasing work? -------------------------
    print(f"\n[4] ranking for the BOOK's phrasing ({literal!r})")
    scored2 = db.similarity_search_with_score(literal, k=8, filter=NOT_JUNK)
    ok = False
    for i, (doc, dist) in enumerate(scored2, 1):
        pg = str(doc.metadata.get("printed_page"))
        mark = "  <== TARGET" if pg in target_pages else ""
        if mark:
            ok = True
        print(f"    {i:>2}. dist={dist:.4f} page {pg:>5}  {short(doc.page_content, 52)!r}{mark}")

    # ---- verdict --------------------------------------------------------
    print("\n" + "=" * 62)
    if rank_of_target and rank_of_target <= 12:
        print("VERDICT: retrieval IS finding it - the prompt is too strict.")
    elif ok:
        print("VERDICT: vocabulary mismatch. The book's wording retrieves it, the")
        print("         user's wording does not. Embeddings are fine. Fix with")
        print("         lexical (BM25) search alongside vectors, and/or query")
        print("         rewriting. A better embedding model will NOT fix this.")
    else:
        print("VERDICT: neither phrasing retrieves it. Suspect the embedding path")
        print("         itself - check that query and document embeddings use the")
        print("         same model and that the BGE prefix is applied to queries only.")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
