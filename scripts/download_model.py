#!/usr/bin/env python3
"""Download BGE embedding model to local project directory.

Usage:
    .\\venv\\Scripts\\python scripts\\download_model.py

Model will be saved to models/bge-small-zh-v1.5/
"""

import os
import sys

# Use HuggingFace mirror for users in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "bge-small-zh-v1.5")
MODEL_NAME = "BAAI/bge-small-zh-v1.5"


def main():
    print(f"Downloading model: {MODEL_NAME}")
    print(f"Save to: {MODEL_DIR}")
    print(f"Mirror: {os.environ['HF_ENDPOINT']}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)
    model.save(MODEL_DIR)
    print(f"Model saved to: {MODEL_DIR}")

    emb = model.encode("test embedding")
    print(f"Embedding dimension: {len(emb)} (expected 384)")
    print("Done!")


if __name__ == "__main__":
    main()
