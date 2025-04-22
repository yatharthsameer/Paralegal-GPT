from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import requests
import os
import re
from datetime import datetime
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION = "legal_chunks"
SOLR_URL = os.getenv("SOLR_URL", "http://solr:8983/solr/opensemanticsearch")
K = 100
RRF_K = 60
SNIPPET_SIZE = 300


def normalize_content(content):
    """Ensure content is a string regardless of input type."""
    if content is None:
        return ""
    if isinstance(content, list):
        if len(content) > 0:
            return content[0] if isinstance(content[0], str) else str(content[0])
        return ""
    return str(content)


def get_snippet(content, query: str, max_length: int = SNIPPET_SIZE) -> str:
    """Generate a snippet with highlighted query terms."""
    content = normalize_content(content)
    if not content:
        return ""

    # Find position of query terms
    query_terms = set(re.findall(r"\w+", query.lower()))
    content_lower = content.lower()

    # Find best snippet position (prioritize query term matches)
    words = content.split()
    if len(words) <= max_length // 10:
        # Content is short enough, use it all
        snippet = content
    else:
        # Find the best section with query term matches
        best_pos = 0
        max_matches = 0
        for i in range(
            0, max(1, len(words) - max_length // 10), max(1, max_length // 20)
        ):
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
        if len(term) > 2:  # Only highlight meaningful terms
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            snippet = pattern.sub(r"<mark>\g<0></mark>", snippet)

    return snippet


def rrf(list1, list2):
    """Combine results using Reciprocal Rank Fusion."""
    scores = {}
    # Process exact matches (Solr results)
    for rank, doc in enumerate(list1, 1):
        scores.setdefault(doc["id"], 0)
        scores[doc["id"]] += 1 / (RRF_K + rank)
        doc["matchType"] = "Exact"

    # Process semantic matches
    for rank, doc in enumerate(list2, 1):
        scores.setdefault(doc["id"], 0)
        scores[doc["id"]] += 1 / (RRF_K + rank)
        # If score from the semantic match already exists, use it
        if "score" in doc:
            doc["semantic_score"] = doc["score"]
        doc["matchType"] = "Semantic"

    # Combine documents
    id2doc = {d["id"]: d for d in list1 + list2}

    # Sort by combined RRF score
    combined_results = sorted(
        [id2doc[i] | {"score": s} for i, s in scores.items()], key=lambda x: -x["score"]
    )

    # Group by original document ID (without chunk) and keep only the top N chunks per document
    if combined_results:
        # Only attempt grouping if any results have original_id
        if "original_id" in combined_results[0]:
            MAX_CHUNKS_PER_DOC = 3  # Maximum chunks to show from one document

            # Group by original document
            docs_by_original = {}
            for doc in combined_results:
                if "original_id" in doc:
                    original_id = doc["original_id"]
                    if original_id not in docs_by_original:
                        docs_by_original[original_id] = []
                    docs_by_original[original_id].append(doc)

            # For each document, keep only the top chunks
            filtered_results = []
            for original_id, chunks in docs_by_original.items():
                # Sort chunks by score
                sorted_chunks = sorted(chunks, key=lambda x: -x["score"])
                # Keep only top chunks
                filtered_results.extend(sorted_chunks[:MAX_CHUNKS_PER_DOC])

            # Sort overall results again by score
            filtered_results = sorted(filtered_results, key=lambda x: -x["score"])

            # Add any remaining non-chunked results
            for doc in combined_results:
                if "original_id" not in doc and doc not in filtered_results:
                    filtered_results.append(doc)

            return filtered_results

    # Default return all results if no chunking
    return combined_results

@app.get("/fusion")
def hybrid_search(q: str):
    """
    Search API that combines BM25 and semantic search results.

    The function:
    1. Gets BM25 results from Solr
    2. Gets semantic results from Qdrant
    3. Normalizes both result sets to a consistent format
    4. Combines them using RRF
    5. Ensures proper highlighting for preview
    """
    logger.info(f"Processing search query: {q}")

    try:
        # 1. Get Solr BM25 results with highlighting
        solr_response = requests.get(
            SOLR_URL + "/select",
            params={
                "q": q,
                "rows": K,
                "fl": "id,title_txt,content,content_txt,file_modified_dt,Content-Length_i,content_type_ss,author_ss,container_s,file_path_s,file_path_basename_s",
                "hl": "true",
                "hl.fl": "content_txt",
                "hl.snippets": 3,
                "hl.fragsize": SNIPPET_SIZE,
                "hl.simple.pre": "<mark>",
                "hl.simple.post": "</mark>",
            },
        ).json()

        # Extract documents and highlighting
        exact_docs = solr_response["response"]["docs"]
        solr_highlights = solr_response.get("highlighting", {})

        # Normalize Solr results
        for doc in exact_docs:
            # Ensure consistent ID format
            doc_id = doc["id"]

            # Normalize content fields
            if "content" in doc:
                doc["content"] = normalize_content(doc["content"])
            if "content_txt" in doc and "content" not in doc:
                doc["content"] = normalize_content(doc["content_txt"])
            elif "content_txt" not in doc and "content" in doc:
                doc["content_txt"] = doc["content"]

            # Add highlighting from Solr if available, otherwise generate
            if doc_id in solr_highlights and "content_txt" in solr_highlights[doc_id]:
                doc["highlighting"] = {
                    "content_txt": solr_highlights[doc_id]["content_txt"]
                }
            else:
                doc["highlighting"] = {
                    "content_txt": [get_snippet(doc.get("content", ""), q)]
                }

        # 2. Get semantic search results from Qdrant
        try:
            vec = model.encode(q).tolist()
            qdrant_response = requests.post(
                f"http://qdrant:6333/collections/{COLLECTION}/points/search",
                json={"vector": vec, "top": K, "with_payload": True},
            ).json()
            semantic_results = qdrant_response.get("result", [])
        except Exception as e:
            logger.error(f"Error fetching semantic results: {str(e)}")
            semantic_results = []

        # 3. Normalize semantic results to match Solr format
        semantic_docs = []
        for p in semantic_results:
            try:
                payload = p.get("payload", {})
                doc_path = payload.get("doc_path", "")
                chunk_id = payload.get("chunk_id", 0)
                content = payload.get("text", "")

                # Get document title from payload if available
                document_title = payload.get("document_title", "")

                # Use just the basename for the ID, but include chunk info to keep chunks separate
                basename = os.path.basename(doc_path)
                # Append chunk ID to make each chunk result unique
                file_id = f"file:///var/opensemanticsearch/documents/{basename}#chunk_{chunk_id}"

                # Create title - use document title from payload if available, otherwise fallback to filename
                if document_title:
                    title = document_title
                else:
                    title = os.path.splitext(basename)[0]

                # Create document with all necessary fields
                doc = {
                    "id": file_id,
                    "original_id": f"file:///var/opensemanticsearch/documents/{basename}",
                    "title_txt": title,
                    "content": normalize_content(content),
                    "content_txt": normalize_content(content),
                    "file_modified_dt": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "Content-Length_i": len(normalize_content(content)),
                    "content_type_ss": ["Text"],
                    "author_ss": [],
                    "container_s": "",
                    "file_path_s": basename,
                    "file_path_basename_s": basename,
                    "chunk_id": chunk_id,
                    "highlighting": {"content_txt": [get_snippet(content, q)]},
                    "matchType": "Semantic",
                    "score": p.get("score", 0),
                }

                semantic_docs.append(doc)
            except Exception as e:
                logger.error(f"Error processing semantic doc: {str(e)}")

        # 4. Combine results using RRF
        results = rrf(exact_docs, semantic_docs)

        # 5. Final check to ensure all results have necessary fields
        for doc in results:
            # Ensure highlighting exists
            if "highlighting" not in doc or not doc["highlighting"].get("content_txt"):
                doc["highlighting"] = {
                    "content_txt": [get_snippet(doc.get("content", ""), q)]
                }

            # Ensure content fields exist
            if "content" not in doc and "content_txt" in doc:
                doc["content"] = doc["content_txt"]
            elif "content_txt" not in doc and "content" in doc:
                doc["content_txt"] = doc["content"]

        return results

    except Exception as e:
        logger.error(f"Error in hybrid_search: {str(e)}", exc_info=True)
        # Return empty results rather than 500 error
        return []
