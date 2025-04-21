#!/usr/bin/env python3
import os
import time
import math
import requests
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Configuration
SOLR_HOST = os.environ.get("SOLR_HOST", "http://localhost:8983")
SOLR_CORE = os.environ.get("SOLR_CORE", "opensemanticsearch")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "legal_chunks")
MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
API_PORT = int(os.environ.get("API_PORT", 8000))

# Reciprocal Rank Fusion constant
RRF_K = 60  # Common value for RRF

# Initialize the model (load only once when the API starts)
model = SentenceTransformer(MODEL_NAME)

app = FastAPI(title="Hybrid Search API", version="1.0.0")


class SearchResult(BaseModel):
    id: str
    title: str
    snippet: str
    score: float
    source: str  # 'exact' or 'semantic'


class SearchResponse(BaseModel):
    total: int
    took_ms: int
    results: List[SearchResult]


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="The search query"),
    rows: int = Query(10, description="Number of results to return"),
    exact_weight: float = Query(0.5, description="Weight for exact search (0-1)"),
    semantic_weight: float = Query(0.5, description="Weight for semantic search (0-1)"),
    filter_query: Optional[str] = Query(None, description="Solr filter query"),
):
    start_time = time.time()

    # Perform both searches in parallel (in real implementation, this would be async)
    exact_results = get_solr_results(q, rows=rows * 3, filter_query=filter_query)
    semantic_results = get_qdrant_results(q, limit=rows * 3)

    # Combine results using Reciprocal Rank Fusion
    combined_results = reciprocal_rank_fusion(
        exact_results,
        semantic_results,
        exact_weight=exact_weight,
        semantic_weight=semantic_weight,
        k=RRF_K,
    )

    # Format the response
    results = []
    for i, item in enumerate(combined_results[:rows]):
        results.append(
            SearchResult(
                id=item["id"],
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                score=item.get("score", 0.0),
                source=item.get("source", ""),
            )
        )

    took_ms = int((time.time() - start_time) * 1000)

    return SearchResponse(total=len(combined_results), took_ms=took_ms, results=results)


def get_solr_results(
    query: str, rows: int = 30, filter_query: Optional[str] = None
) -> List[Dict[Any, Any]]:
    """Get results from Solr"""
    params = {
        "q": query,
        "rows": rows,
        "wt": "json",
        "fl": 'id,title,content:[value v=""],score',
    }

    if filter_query:
        params["fq"] = filter_query

    try:
        response = requests.get(f"{SOLR_HOST}/solr/{SOLR_CORE}/select", params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for doc in data.get("response", {}).get("docs", []):
            snippet = ""
            if "content" in doc:
                snippet = (
                    doc["content"][:200] + "..."
                    if len(doc["content"]) > 200
                    else doc["content"]
                )

            results.append(
                {
                    "id": doc.get("id", ""),
                    "title": doc.get("title", ""),
                    "snippet": snippet,
                    "score": doc.get("score", 0),
                    "source": "exact",
                }
            )

        return results

    except Exception as e:
        print(f"Error getting Solr results: {e}")
        return []


def get_qdrant_results(query: str, limit: int = 30) -> List[Dict[Any, Any]]:
    """Get results from Qdrant based on semantic similarity"""
    try:
        # Generate embedding for the query
        query_vector = model.encode(query).tolist()

        # Search Qdrant
        response = requests.post(
            f"{QDRANT_HOST}/collections/{QDRANT_COLLECTION}/points/search",
            json={"vector": query_vector, "limit": limit, "with_payload": True},
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("result", []):
            payload = item.get("payload", {})
            results.append(
                {
                    "id": payload.get("source_id", str(item.get("id", ""))),
                    "title": payload.get("title", ""),
                    "snippet": (
                        payload.get("text", "")[:200] + "..."
                        if len(payload.get("text", "")) > 200
                        else payload.get("text", "")
                    ),
                    "score": item.get("score", 0),
                    "source": "semantic",
                }
            )

        return results

    except Exception as e:
        print(f"Error getting Qdrant results: {e}")
        return []


def reciprocal_rank_fusion(
    exact_results: List[Dict],
    semantic_results: List[Dict],
    exact_weight: float = 0.5,
    semantic_weight: float = 0.5,
    k: int = 60,
) -> List[Dict]:
    """
    Combine results using Reciprocal Rank Fusion with weighted sources
    """
    # Normalize weights to sum to 1.0
    total_weight = exact_weight + semantic_weight
    exact_weight = exact_weight / total_weight
    semantic_weight = semantic_weight / total_weight

    # Create a map of document ID to its combined score
    doc_scores = {}

    # Process exact match results
    for rank, doc in enumerate(exact_results):
        doc_id = doc["id"]
        rrf_score = 1.0 / (rank + k) * exact_weight

        if doc_id not in doc_scores:
            doc_scores[doc_id] = {"doc": doc, "score": 0}

        doc_scores[doc_id]["score"] += rrf_score

    # Process semantic search results
    for rank, doc in enumerate(semantic_results):
        doc_id = doc["id"]
        rrf_score = 1.0 / (rank + k) * semantic_weight

        if doc_id not in doc_scores:
            doc_scores[doc_id] = {"doc": doc, "score": 0}
        elif doc_scores[doc_id]["doc"]["source"] == "exact":
            # Document exists in both result sets, stick with the exact match
            # but use the semantic score to boost it
            doc_scores[doc_id]["score"] += rrf_score
            continue

        doc_scores[doc_id]["score"] += rrf_score

    # Sort by combined score and convert back to list
    combined_results = []
    for doc_id, data in sorted(
        doc_scores.items(), key=lambda x: x[1]["score"], reverse=True
    ):
        doc = data["doc"].copy()
        doc["score"] = data["score"]
        combined_results.append(doc)

    return combined_results


if __name__ == "__main__":
    uvicorn.run("hybrid_search_api:app", host="0.0.0.0", port=API_PORT, reload=True)
