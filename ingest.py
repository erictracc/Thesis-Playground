"""
ingest.py - v2
--------------
Reads problem description + label pairs from Large_Scale_Or_Files/Other_example/
Embeds the problem descriptions using sentence-transformers (local)
Stores vectors + metadata into Qdrant (on your server)

Key fix from v1:
  - Reads QDRANT_HOST, QDRANT_PORT from .env (no hardcoded IPs)
  - Stores problem_type in payload so the Instructor agent can do
    type-filtered RAG lookups correctly

Usage:
    python ingest.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# Config - all from .env
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "bpo_problems"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATA_DIR        = Path("Large_Scale_Or_Files/Other_example")

# Problem type keyword detection
# Must match PROBLEM_TYPES in agents.py exactly
PROBLEM_TYPE_KEYWORDS = {
    "Facility Location Problem (FLP)": [
        "facility location", "plant", "warehouse", "opening cost",
        "fixed cost", "service centre", "service center",
    ],
    "Transportation Problem (TP)": [
        "transportation", "supply", "demand", "shipment", "route",
        "source", "destination", "truck", "cargo",
    ],
    "Assignment Problem (AP)": [
        "assignment", "machine", "task", "worker", "job",
        "one-to-one", "bipartite",
    ],
    "Network Revenue Management (NRM)": [
        "revenue management", "booking", "capacity allocation",
        "fare", "airline", "flight", "network revenue",
    ],
    "Resource Allocation (RA)": [
        "resource allocation", "budget", "allocation",
        "capacity", "knapsack",
    ],
    "Scheduling Problem (SP)": [
        "scheduling", "schedule", "makespan", "due date",
        "processing time", "machine scheduling",
    ],
}


def detect_problem_type(description: str) -> str:
    desc_lower = description.lower()
    scores = {}
    for ptype, keywords in PROBLEM_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in desc_lower)
        if score > 0:
            scores[ptype] = score
    if scores:
        return max(scores, key=scores.get)
    return "Other"


def load_problems(data_dir: Path) -> list[dict]:
    problems = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        desc_file  = folder / "problem description.txt"
        label_file = folder / "label.txt"
        if not desc_file.exists() or not label_file.exists():
            print(f"  Skipping {folder.name} - missing description or label")
            continue
        description  = desc_file.read_text(encoding="utf-8").strip()
        label        = label_file.read_text(encoding="utf-8").strip()
        problem_type = detect_problem_type(description)
        problems.append({
            "id":           folder.name,
            "description":  description,
            "label":        label,
            "problem_type": problem_type,
        })
        print(f"  Loaded problem {folder.name:>8} - type: {problem_type}")
    return problems


def setup_collection(client: QdrantClient, vector_size: int):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists - recreating...")
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created (vector size: {vector_size})")


def ingest():
    print("=== LEAN-LLM-OPT Mini Ingest v2 ===\n")
    print(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Data:   {DATA_DIR}\n")

    print(f"Loading problems from {DATA_DIR}...")
    problems = load_problems(DATA_DIR)
    print(f"\nLoaded {len(problems)} problems\n")

    if not problems:
        print("No problems found. Check your DATA_DIR path.")
        return

    from collections import Counter
    type_counts = Counter(p["problem_type"] for p in problems)
    print("Problem type distribution:")
    for ptype, count in sorted(type_counts.items()):
        print(f"  {ptype}: {count}")
    print()

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model        = SentenceTransformer(EMBEDDING_MODEL)
    descriptions = [p["description"] for p in problems]
    print("Embedding problem descriptions...")
    embeddings   = model.encode(descriptions, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}\n")

    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        timeout=30,
        check_compatibility=False,
    )
    setup_collection(client, vector_size=embeddings.shape[1])

    print("Uploading to Qdrant...")
    points = []
    for i, (problem, vector) in enumerate(zip(problems, embeddings)):
        points.append(PointStruct(
            id=i,
            vector=vector.tolist(),
            payload={
                "problem_id":   problem["id"],
                "description":  problem["description"],
                "label":        problem["label"],
                "problem_type": problem["problem_type"],  # KEY FIX - was missing in v1
            }
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"\nSuccessfully ingested {len(points)} problems into '{COLLECTION_NAME}'")

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"Qdrant collection now has {count} vectors")

    print("\nVerifying type-filtered retrieval...")
    test_vector = embeddings[0].tolist()
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    for ptype in list(type_counts.keys())[:3]:
        q_filter = Filter(must=[FieldCondition(
            key="problem_type",
            match=MatchValue(value=ptype)
        )])
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=test_vector,
            limit=3,
            with_payload=True,
            query_filter=q_filter,
        )
        print(f"  Filter '{ptype}': {len(results.points)} results OK")

    print("\nIngest complete - Instructor agent type filtering will now work correctly")


if __name__ == "__main__":
    ingest()