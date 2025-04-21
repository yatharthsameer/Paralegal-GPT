#!/usr/bin/env python3
import os
import sys
import json
import requests
import argparse
from sentence_transformers import SentenceTransformer
import hashlib

# Configuration
QDRANT_HOST = os.environ.get("QDRANT_HOST", "http://localhost:6333")
COLLECTION_NAME = "legal_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"

# Initialize the model
model = SentenceTransformer(MODEL_NAME)


def get_document_id_hash(file_path):
    """Generate a consistent hash ID from file path"""
    return hashlib.md5(file_path.encode()).hexdigest()


def get_document_content(file_path):
    """Extract content from file - this is a simplified version"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def embed_and_store(doc_id, text, metadata=None):
    """Generate embedding and store in Qdrant"""
    if not text:
        return False

    # Generate embedding
    embedding = model.encode(text).tolist()

    # Prepare metadata
    if metadata is None:
        metadata = {}

    # Add document to Qdrant
    response = requests.put(
        f"{QDRANT_HOST}/collections/{COLLECTION_NAME}/points",
        json={
            "points": [
                {
                    "id": doc_id,
                    "vector": embedding,
                    "payload": {
                        "text": text[
                            :1000
                        ],  # Store only the first 1000 chars in payload
                        "source_id": metadata.get("source_id", ""),
                        "source_path": metadata.get("source_path", ""),
                        "title": metadata.get("title", ""),
                    },
                }
            ]
        },
    )

    return response.status_code == 200


def process_file(file_path, metadata=None):
    """Process a single file - read, embed, and store"""
    doc_id = get_document_id_hash(file_path)
    text = get_document_content(file_path)

    if not text:
        return False

    if metadata is None:
        metadata = {"source_path": file_path, "title": os.path.basename(file_path)}

    return embed_and_store(doc_id, text, metadata)


def process_directory(directory_path, metadata=None):
    """Process all files in a directory recursively"""
    for root, _, files in os.walk(directory_path):
        for file in files:
            # Skip hidden files and certain extensions
            if file.startswith(".") or file.endswith((".tmp", ".bak", ".swp")):
                continue

            file_path = os.path.join(root, file)
            file_metadata = (
                metadata.copy()
                if metadata
                else {"source_path": file_path, "title": os.path.basename(file_path)}
            )
            process_file(file_path, file_metadata)
            print(f"Processed {file_path}")


def read_metadata_file(metadata_file):
    """Read metadata from a JSON file"""
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading metadata file {metadata_file}: {e}")
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed documents and store in Qdrant")
    parser.add_argument("path", help="File or directory path to process")
    parser.add_argument("--metadata", help="JSON file containing metadata")

    args = parser.parse_args()

    metadata = None
    if args.metadata:
        metadata = read_metadata_file(args.metadata)

    if os.path.isfile(args.path):
        success = process_file(args.path, metadata)
        print(f"Processed file {args.path}: {'Success' if success else 'Failed'}")
    elif os.path.isdir(args.path):
        process_directory(args.path, metadata)
        print(f"Processed directory {args.path}")
    else:
        print(f"Path {args.path} does not exist")
        sys.exit(1)
