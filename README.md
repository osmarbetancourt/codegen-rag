# CodeGen-RAG: Data Preparation for Retrieval-Augmented Generation with Pinecone, BGE, and Mistral

This project demonstrates a modern, scalable pipeline for preparing code/documentation data for Retrieval-Augmented Generation (RAG) use cases. It automates the process of downloading, embedding, and uploading code data to Pinecone, making it ready for downstream RAG applications (e.g., with Mistral-7B-Instruct or other LLMs).

**Note:** This repository focuses on data preparation and vector database ingestion. It does not include the retrieval or LLM inference steps of a full RAG pipeline.

## Features
- Automated download and sampling of 100K+ Python code examples from CodeSearchNet
- Local embedding generation using SentenceTransformers (BGE-base)
- Efficient, batch upload of embeddings and metadata to Pinecone (with per-vector and per-request size validation)
- GPU acceleration if available, but fully compatible with CPU-only systems
- All steps (download, embedding, upload) run automatically via Docker Compose
- Robust error handling for Pinecone API limits (batch size, metadata size)

## Quickstart

1. **Clone the repo and set up your Pinecone API key in a `.env` file:**
   ```env
   PINECONE_API_KEY=your-pinecone-key
   ```

2. **Build and run the pipeline:**
   ```sh
   docker compose down  # (optional, to clean up)
   docker compose up --build
   ```
   This will:
   - Download and sample 100,000 Python code examples
   - Generate BGE-base embeddings (using GPU if available)
   - Upload vectors to Pinecone in safe batches

3. **Monitor progress:**
   - Console logs show progress, warnings for skipped vectors, and upsert status
   - Check your Pinecone dashboard to see the new index and vectors

## Project Structure
```
codegen-rag/
├── data/                  # Downloaded and sampled data (ignored by git)
├── scripts/
│   ├── download_sample_codesearchnet.py   # Downloads and samples CodeSearchNet data
│   ├── upload_to_pinecone.py              # Embeds and uploads vectors to Pinecone (with batching and validation)
│   └── calculate_estimated_wus.py         # Tool to estimate record size and WUs for Pinecone
├── Dockerfile.base         # PyTorch + CUDA + sentence-transformers base image
├── Dockerfile              # App image, runs the pipeline
├── docker-compose.yml      # Orchestrates the workflow, requests GPU if available
├── default-cmd.sh          # Entrypoint script for the pipeline
├── requirements.txt        # Python dependencies
├── .env                    # Pinecone API key (not committed)
└── README.md               # Project documentation
```

## Notes & Best Practices
- Pinecone free tier: 2GB storage, 1M WUs/month, 100K+ records supported (with small code snippets)
- Per-vector metadata limit: 40,960 bytes (vectors exceeding this are skipped with a warning)
- Batch upserts: 100 vectors per request (to avoid API payload limits)
- Fully reproducible: all steps run automatically in Docker
- Compatible with both CPU and GPU environments
- For estimating your Pinecone usage, see `scripts/calculate_estimated_wus.py` — this tool helps you measure average record size and WUs, and now includes a tree-structure summary for visualizing your dataset's impact on Pinecone limits.

## Credits
- CodeSearchNet dataset: https://huggingface.co/datasets/claudios/code_search_net
- BGE-base embedding model: https://huggingface.co/BAAI/bge-base-en-v1.5
- Pinecone vector database: https://www.pinecone.io/
- Mistral-7B-Instruct: https://mistral.ai/

---

For questions or improvements, open an issue or PR!
