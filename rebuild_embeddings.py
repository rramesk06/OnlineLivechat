import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_FILE = DATA_DIR / "documents.json"
FAISS_DIR = DATA_DIR / "faiss"

MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

if not DOCS_FILE.exists():
    raise RuntimeError(f"Documents file not found: {DOCS_FILE}")

client = OpenAI(api_key=API_KEY)
documents = json.loads(DOCS_FILE.read_text(encoding="utf-8"))

if not isinstance(documents, list):
    raise RuntimeError("data/documents.json must contain a list of documents")

FAISS_DIR.mkdir(parents=True, exist_ok=True)

for document in documents:
    document_id = document["document_id"]
    chunks = document.get("chunks", [])
    texts = [chunk["text"] for chunk in chunks]

    if not texts:
        print(f"SKIP {document_id}: no chunks")
        continue

    print(f"Rebuilding {document.get('filename', document_id)} ({len(texts)} chunks)...")

    response = client.embeddings.create(
        model=MODEL,
        input=texts,
    )

    ordered = sorted(response.data, key=lambda item: item.index)
    vectors = np.asarray(
        [[float(value) for value in item.embedding] for item in ordered],
        dtype="float32",
    )

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    path = FAISS_DIR / f"{document_id}.index"
    faiss.write_index(index, str(path))

    document["faiss_path"] = str(path.relative_to(ROOT))
    print(f"OK: {path}")

DOCS_FILE.write_text(json.dumps(documents, indent=2), encoding="utf-8")
print("Embedding/index rebuild completed.")
