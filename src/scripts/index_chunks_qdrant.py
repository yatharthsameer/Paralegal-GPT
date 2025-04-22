#!/usr/bin/env python3
from sentence_transformers import SentenceTransformer
import requests
import uuid
import os
from pathlib import Path
import argparse
from bs4 import BeautifulSoup
import re

model = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION = "legal_chunks"


def extract_text_from_html(html_content):
    """Extract text content from HTML."""
    soup = BeautifulSoup(html_content, "html.parser")
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text(separator=" ", strip=True)


def extract_title_from_html(html_content):
    """Extract title from HTML document."""
    soup = BeautifulSoup(html_content, "html.parser")
    # Try to get title tag content
    if soup.title and soup.title.string:
        return soup.title.string.strip()

    # If no title tag, try to find h1 or h2 with class="doc_title"
    doc_title = soup.find(class_="doc_title")
    if doc_title:
        return doc_title.get_text().strip()

    # If still no title, try any h1
    h1 = soup.find("h1")
    if h1:
        return h1.get_text().strip()

    return None


def extract_title_from_txt(txt_content, filename):
    """Try to extract title from first few lines of text file."""
    # Try to find a title in the first few lines
    lines = txt_content.split("\n")[:10]  # Look at first 10 lines

    # Look for lines that might be titles (not too long, not too short)
    for line in lines:
        line = line.strip()
        if (
            10 < len(line) < 200
            and not line.startswith("#")
            and not re.match(r"^[0-9.]+$", line)
        ):
            return line

    # If no good title found, use filename
    return os.path.splitext(filename)[0]


def process_documents(docs_path):
    # Process both .txt and .html files
    for extension in ["*.txt", "*.html"]:
        docs = Path(docs_path).rglob(extension)

        for path in docs:
            print(f"Processing {path}")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Extract title based on file type
                filename = path.name
                document_title = None

                # If it's an HTML file, extract the text and title
                if path.suffix.lower() == ".html":
                    document_title = extract_title_from_html(content)
                    content = extract_text_from_html(content)
                else:
                    # Try to extract title from text file content
                    document_title = extract_title_from_txt(content, filename)

                # Fallback to filename if no title was found
                if not document_title:
                    document_title = os.path.splitext(filename)[0]

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
                                "document_title": document_title,
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
