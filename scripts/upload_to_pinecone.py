import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import json

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

print("Testing Pinecone connection...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to an index
INDEX_NAME = "codegen-demo"
DIM = 768  # BGE-base embedding size

if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"Index '{INDEX_NAME}' already exists.")

index = pc.Index(INDEX_NAME)

# Load embedding model
MODEL_NAME = "BAAI/bge-base-en-v1.5"
print(f"Loading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
if torch.cuda.is_available():
    print("CUDA is available! Using GPU for embeddings.")
else:
    print("CUDA is NOT available. Using CPU for embeddings.")

# Load 10 samples from your dataset
sample_path = "data/sample_codesearchnet.jsonl"
df = pd.read_json(sample_path, lines=True)

# Prepare and upsert embeddings
vectors = []
MAX_METADATA_SIZE = 40960  # Pinecone per-vector metadata limit in bytes
for idx, row in df.iterrows():
    code = row.get("func_code_string") or row.get("whole_func_string") or str(row)
    embedding = model.encode(code, normalize_embeddings=True)
    vector_id = f"sample-{idx}"
    metadata = {"code": code}
    metadata_bytes = len(json.dumps(metadata).encode("utf-8"))
    if metadata_bytes > MAX_METADATA_SIZE:
        print(f"Warning: Skipping vector {vector_id} (metadata size {metadata_bytes} bytes exceeds {MAX_METADATA_SIZE} bytes limit)")
        continue
    vectors.append({"id": vector_id, "values": embedding.tolist(), "metadata": metadata})
    if (idx + 1) % 1000 == 0:
        print(f"Prepared {idx + 1} vectors...")

BATCH_SIZE = 100
print(f"Upserting {len(vectors)} vectors to Pinecone in batches of {BATCH_SIZE}...")
for i in range(0, len(vectors), BATCH_SIZE):
    batch = vectors[i:i+BATCH_SIZE]
    index.upsert(batch)
    print(f"Upserted {min(i+BATCH_SIZE, len(vectors))} vectors to Pinecone...")
print("Upsert complete! Check your Pinecone UI.")
