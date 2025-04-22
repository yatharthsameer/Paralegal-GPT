#!/usr/bin/env python3
from sentence_transformers import SentenceTransformer
import requests
import uuid
import os
from pathlib import Path
import argparse
from bs4 import BeautifulSoup

model = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION = "legal_chunks"


def extract_text_from_html(html_content):
    """Extract text content from HTML."""
    soup = BeautifulSoup(html_content, "html.parser")
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text(separator=" ", strip=True)


def process_documents(docs_path):
    # Process both .txt and .html files
    for extension in ["*.txt", "*.html"]:
        docs = Path(docs_path).rglob(extension)

        for path in docs:
            print(f"Processing {path}")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                # If it's an HTML file, extract the text
                if path.suffix.lower() == ".html":
                    content = extract_text_from_html(content)

                chunks = [content[i : i + 512] for i in range(0, len(content), 400)]
                payload = []
                for i, chunk in enumerate(chunks):
                    if not chunk.strip():  # Skip empty chunks
                        continue
                    vec = model.encode(chunk).tolist()
                    payload.append(
                        {
                            "id": str(uuid.uuid4()),
                            "vector": vec,
                            "payload": {
                                "text": chunk,
                                "doc_path": str(path),
                                "chunk_id": i,
                            },
                        }
                    )
                if payload:  # Only send request if there are non-empty chunks
                    response = requests.put(
                        f"http://localhost:6333/collections/{COLLECTION}/points",
                        json={"points": payload},
                    )
                    print(
                        f"  Added {len(chunks)} chunks with status: {response.status_code}"
                    )
            except Exception as e:
                print(f"  Error processing {path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index documents into Qdrant")
    parser.add_argument("docs_path", help="Path to directory containing documents")
    args = parser.parse_args()

    process_documents(args.docs_path)
