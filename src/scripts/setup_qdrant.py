#!/usr/bin/env python3
import requests

COLLECTION = "legal_chunks"
response = requests.put(
    "http://localhost:6333/collections/" + COLLECTION,
    json={"vectors": {"size": 384, "distance": "Cosine"}},
)
print(response.json())
