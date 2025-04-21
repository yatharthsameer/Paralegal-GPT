#!/bin/bash

echo "Setting up Qdrant collection..."
python3 setup_qdrant.py

echo "Testing hybrid search API..."
curl -X GET "http://localhost:8000/search?q=test"

echo "Done!" 