#!/bin/bash
set -e

echo "Starting default-cmd.sh"

# Download data
python ./scripts/download_sample_codesearchnet.py

# Upload to Pinecone (test connection)
python ./scripts/upload_to_pinecone.py
