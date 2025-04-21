from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import requests
import os
import re
from datetime import datetime
import hashlib

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION = "legal_chunks"
SOLR_URL = os.getenv("SOLR_URL", "http://solr:8983/solr/opensemanticsearch")
K = 100
RRF_K = 60
SNIPPET_SIZE = 300


def get_snippet(content: str, query: str, max_length: int = SNIPPET_SIZE) -> str:
    # Simple snippet generation with query term highlighting
    if not content:
        return ""

    # Find position of query terms
    query_terms = set(re.findall(r"\w+", query.lower()))
    content_lower = content.lower()

    # Find best snippet position (prioritize query term matches)
    best_pos = 0
    max_matches = 0

    words = content.split()
    for i in range(0, len(words) - max_length // 10, max_length // 10):
        snippet_words = words[i : i + max_length // 10]
        snippet_text = " ".join(snippet_words).lower()
        matches = sum(1 for term in query_terms if term in snippet_text)
        if matches > max_matches:
            max_matches = matches
            best_pos = i

    # Extract snippet
    start = best_pos
    end = min(start + max_length // 10, len(words))
    snippet = " ".join(words[start:end])

    if start > 0:
        snippet = "... " + snippet
    if end < len(words):
        snippet = snippet + " ..."

    # Highlight query terms
    for term in query_terms:
        pattern = re.compile(f"({term})", re.IGNORECASE)
        snippet = pattern.sub(r"<em>\1</em>", snippet)

    return snippet


def rrf(list1, list2):
    scores = {}
    for rank, doc in enumerate(list1, 1):
        scores.setdefault(doc["id"], 0)
        scores[doc["id"]] += 1 / (RRF_K + rank)
        doc["matchType"] = "Exact"
    for rank, doc in enumerate(list2, 1):
        scores.setdefault(doc["id"], 0)
        scores[doc["id"]] += 1 / (RRF_K + rank)
        doc["matchType"] = "Semantic"
    id2doc = {d["id"]: d for d in list1 + list2}
    return sorted(
        [id2doc[i] | {"score": s} for i, s in scores.items()], key=lambda x: -x["score"]
    )


@app.get("/fusion")
def hybrid_search(q: str):
    # Solr BM25
    r1 = requests.get(
        SOLR_URL + "/select",
        params={
            "q": q,
            "rows": K,
            "fl": "id,title,content,file_modified_dt,Content-Length_i,content_type_ss,author_ss",
        },
    ).json()["response"]["docs"]

    # Qdrant vector
    vec = model.encode(q).tolist()
    r2 = requests.post(
        f"http://qdrant:6333/collections/{COLLECTION}/points/search",
        json={"vector": vec, "top": K, "with_payload": True},
    ).json()["result"]

    semantic_docs = []
    for p in r2:
        doc_path = p["payload"].get("doc_path", "")
        content = p["payload"].get("text", "")

        # Generate a file ID that matches Solr format
        file_id = f"file:///var/opensemanticsearch/documents/{doc_path}"

        # Create document with all necessary fields
        doc = {
            "id": file_id,
            "title": doc_path,
            "content": content,
            "content_txt": content,  # For snippet generation
            "file_modified_dt": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Content-Length_i": len(content),
            "content_type_ss": ["Text"],
            "author_ss": [],  # Could be extracted from metadata if available
            "container_s": os.path.dirname(doc_path),
            "file_path_s": doc_path,
            "file_path_basename_s": os.path.basename(doc_path),
        }

        # Add highlighting/snippets
        doc["highlighting"] = {"content_txt": [get_snippet(content, q)]}

        semantic_docs.append(doc)

    results = rrf(r1, semantic_docs)

    # Add highlighting to results
    for doc in results:
        if "highlighting" not in doc and "content" in doc:
            doc["highlighting"] = {"content_txt": [get_snippet(doc["content"], q)]}

    return results
