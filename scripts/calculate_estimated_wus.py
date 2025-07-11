import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import json

# Use BGE-base embedding model for RAG with Mistral
MODEL_NAME = "BAAI/bge-base-en-v1.5"

print(f"Loading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

# Load 10 samples from your dataset
sample_path = "data/sample_codesearchnet.jsonl"
df = pd.read_json(sample_path, lines=True)
sample_df = df.head(10)

sizes = []
for idx, row in sample_df.iterrows():
    code = row.get("func_code_string") or row.get("whole_func_string") or str(row)
    embedding = model.encode(code, normalize_embeddings=True)
    record = {
        "id": f"sample-{idx}",
        "values": embedding.tolist(),
        "metadata": {"code": code}
    }
    record_json = json.dumps(record)
    size_bytes = len(record_json.encode("utf-8"))
    sizes.append(size_bytes)
    wus = (size_bytes + 1023) // 1024  # 1 WU per KB, round up
    print(f"Sample {idx} record size: {size_bytes} bytes ({size_bytes/1024:.2f} KB), WUs: {wus}")

avg_size = sum(sizes) / len(sizes)
avg_wus = sum((s + 1023) // 1024 for s in sizes) / len(sizes)
total_wus = int(avg_wus * 5000)
print(f"Average record size: {avg_size:.2f} bytes ({avg_size/1024:.2f} KB)")
print(f"Average WUs per record: {avg_wus:.2f}")
print(f"Estimated total for 5,000 records: {avg_size*5000/1024/1024:.2f} MB")
print(f"Estimated total WUs for 5,000 records: {total_wus}")
