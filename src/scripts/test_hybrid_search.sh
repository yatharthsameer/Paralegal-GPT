#!/bin/bash

echo "Setting up Qdrant collection..."
python3 setup_qdrant.py

echo "Generating embeddings for sample documents..."
python3 index_chunks_qdrant.py ../Sample_ID_742

echo "Testing fusion API..."
curl -X GET "http://localhost:8001/fusion?q=test"

echo "Open your browser to http://localhost:8080 to use the search UI"
echo "Remember to check 'Include semantically similar results' in search options to use hybrid search"
echo "Done!" 