# App Dockerfile, builds on top of ML base
FROM codegen-rag-base:latest

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/
COPY data/ ./data/
COPY default-cmd.sh ./

# Make sure the default-cmd.sh script is executable
RUN chmod +x default-cmd.sh

# Run both scripts: download data and then generate embeddings
CMD ["bash", "./default-cmd.sh"]
