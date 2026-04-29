"""
ingest.py
---------
Reads problem description + label pairs from Large_Scale_Or_Files/Other_example/
Embeds the problem descriptions using sentence-transformers (local)
Stores vectors + metadata into Qdrant (on your server)

Usage:
    python ingest.py
"""

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── Config ────────────────────────────────────────────────────────────────────
QDRANT_HOST = "192.168.195.69"
QDRANT_PORT = 6333
COLLECTION_NAME = "bpo_problems"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # small, fast, runs locally
DATA_DIR = Path("Large_Scale_Or_Files/Other_example")
# ──────────────────────────────────────────────────────────────────────────────


def load_problems(data_dir: Path) -> list[dict]:
    """Walk each numbered subfolder and load problem description + label."""
    problems = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue

        desc_file = folder / "problem description.txt"
        label_file = folder / "label.txt"

        if not desc_file.exists() or not label_file.exists():
            print(f"  Skipping {folder.name} — missing description or label")
            continue

        description = desc_file.read_text(encoding="utf-8").strip()
        label = label_file.read_text(encoding="utf-8").strip()

        problems.append({
            "id": folder.name,
            "description": description,
            "label": label,
        })
        print(f"  Loaded problem {folder.name}")

    return problems


def setup_collection(client: QdrantClient, vector_size: int):
    """Create Qdrant collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists — recreating...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created with vector size {vector_size}")


def ingest():
    print("=== LEAN-LLM-OPT Mini Ingest ===\n")

    # 1. Load problems
    print(f"Loading problems from {DATA_DIR}...")
    problems = load_problems(DATA_DIR)
    print(f"\nLoaded {len(problems)} problems\n")

    if not problems:
        print("No problems found. Check your DATA_DIR path.")
        return

    # 2. Embed descriptions
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    descriptions = [p["description"] for p in problems]
    print("Embedding problem descriptions...")
    embeddings = model.encode(descriptions, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}\n")

    # 3. Connect to Qdrant
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    setup_collection(client, vector_size=embeddings.shape[1])

    # 4. Upload points
    print("Uploading to Qdrant...")
    points = []
    for i, (problem, vector) in enumerate(zip(problems, embeddings)):
        points.append(PointStruct(
            id=i,
            vector=vector.tolist(),
            payload={
                "problem_id": problem["id"],
                "description": problem["description"],
                "label": problem["label"],
            }
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"\n✅ Successfully ingested {len(points)} problems into '{COLLECTION_NAME}'")

    # 5. Quick sanity check
    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"✅ Qdrant collection now has {count} vectors")


if __name__ == "__main__":
    ingest()
