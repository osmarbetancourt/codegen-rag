import os
from datasets import load_dataset
import pandas as pd

# Parameters
LANGUAGE = "python"  # Change this to another language if needed
SAMPLE_SIZE = 100000
OUTPUT_PATH = "data/sample_codesearchnet.jsonl"

# Load dataset from Hugging Face
print(f"Loading CodeSearchNet dataset for language: {LANGUAGE}")
ds = load_dataset("claudios/code_search_net", LANGUAGE,split="train")

# Convert to pandas DataFrame for easy sampling
print("Converting to DataFrame and sampling...")
df = pd.DataFrame(ds)
print(df.columns)  # Print columns to verify structure
# Filter out rows with missing code or docstring
filtered = df.dropna(subset=["func_code_string", "func_documentation_string"])

# Sample up to SAMPLE_SIZE rows
sampled = filtered.sample(n=min(SAMPLE_SIZE, len(filtered)), random_state=42)

# Save to JSONL
print(f"Saving {len(sampled)} samples to {OUTPUT_PATH}")
sampled.to_json(OUTPUT_PATH, orient="records", lines=True)

print("Done.")
