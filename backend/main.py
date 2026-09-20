import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import fitz
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import OpenAI
from pydantic import BaseModel, Field
from tavily import TavilyClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FAISS_DIR = DATA_DIR / "faiss"
DOCS_FILE = DATA_DIR / "documents.json"
GUARDRAILS_DIR = ROOT / "guardrails"

for directory in (UPLOAD_DIR, FAISS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
NEO4J_URI = os.getenv("NEO4J_URI", "").strip()
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is required")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
neo4j_driver = (
    GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    if NEO4J_URI and NEO4J_PASSWORD
    else None
)

app = FastAPI(title="PDF RAG + Graph RAG API", version="1.0.0")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)


def load_documents() -> List[Dict[str, Any]]:
    if not DOCS_FILE.exists():
        return []
    try:
        data = json.loads(DOCS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("documents"), list):
            return data["documents"]
    except Exception:
        pass
    return []


def save_documents(documents: List[Dict[str, Any]]) -> None:
    DOCS_FILE.write_text(json.dumps(documents, indent=2), encoding="utf-8")


def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    return next((d for d in load_documents() if d.get("document_id") == document_id), None)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    pages = []
    with fitz.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            text = normalize_text(page.get_text("text"))
            if text:
                pages.append({"page": page_no, "text": text})
    return pages


def build_chunks(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for page in pages:
        pieces = splitter.split_text(page["text"])
        for piece_index, text in enumerate(pieces):
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "page": page["page"],
                    "chunk_index": piece_index,
                    "text": text,
                }
            )
    return chunks


def create_embeddings(texts: List[str]):
    if not texts:
        return []
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    items = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in items]


def save_faiss(document_id: str, chunks: List[Dict[str, Any]]) -> Path:
    texts = [c["text"] for c in chunks]
    vectors = np.array(
        create_embeddings(texts), dtype="float32"
    )
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    path = FAISS_DIR / f"{document_id}.index"
    faiss.write_index(index, str(path))
    return path


def load_faiss(document_id: str):
    path = FAISS_DIR / f"{document_id}.index"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Vector index not found")
    return faiss.read_index(str(path))


def vector_search(document_id: str, question: str, top_k: int) -> List[Dict[str, Any]]:
    document = get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = document.get("chunks", [])
    if not chunks:
        return []
    index = load_faiss(document_id)
    query = np.array(
        create_embeddings([question]), dtype="float32"
    )
    faiss.normalize_L2(query)
    scores, ids = index.search(query, min(top_k, len(chunks)))
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[int(idx)]
        results.append({**chunk, "vector_score": round(float(score), 6)})
    return results


def extract_concepts(question: str) -> List[str]:
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Extract concise entities or technology concepts from the question. Return JSON: {\"concepts\":[\"...\"]}. Max 8 concepts.",
            },
            {"role": "user", "content": question},
        ],
    )
    try:
        values = json.loads(response.choices[0].message.content or "{}").get("concepts", [])
        return [str(x).strip() for x in values if str(x).strip()][:8]
    except Exception:
        return []


def graph_search(document_id: str, question: str, top_k: int) -> List[Dict[str, Any]]:
    if not neo4j_driver:
        return []
    concepts = extract_concepts(question)
    if not concepts:
        return []
    cypher = """
    MATCH (d:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:Chunk)
    OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
    WHERE any(concept IN $concepts WHERE toLower(e.name) CONTAINS toLower(concept))
    WITH c, collect(DISTINCT e.name) AS entities
    WHERE size(entities) > 0
    RETURN c.chunk_id AS chunk_id, c.text AS text, c.page AS page,
           entities, size(entities) AS graph_score
    ORDER BY graph_score DESC
    LIMIT $top_k
    """
    with neo4j_driver.session() as session:
        rows = session.run(cypher, document_id=document_id, concepts=concepts, top_k=top_k)
        return [dict(row) for row in rows]


def graph_store(document_id: str, chunks: List[Dict[str, Any]]) -> None:
    if not neo4j_driver:
        return

    cypher = """
    MERGE (d:Document {document_id: $document_id})
    SET d.updated_at = datetime()
    WITH d
    UNWIND $chunks AS item
    MERGE (c:Chunk {chunk_id: item.chunk_id})
    SET c.document_id = $document_id, c.text = item.text, c.page = item.page,
        c.chunk_index = item.chunk_index
    MERGE (d)-[:HAS_CHUNK]->(c)
    WITH c, item
    UNWIND item.entities AS entity_name
    MERGE (e:Entity {name: entity_name})
    MERGE (c)-[:MENTIONS]->(e)
    """
    with neo4j_driver.session() as session:
        session.run(cypher, document_id=document_id, chunks=chunks)


def extract_entities(text: str) -> List[str]:
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Extract named technologies, frameworks, databases, protocols, cloud services, products, and important technical concepts. Return JSON {\"entities\":[\"...\"]}. Max 15. Do not invent.",
            },
            {"role": "user", "content": text[:6000]},
        ],
    )
    try:
        entities = json.loads(response.choices[0].message.content or "{}").get("entities", [])
        return list(dict.fromkeys(str(x).strip() for x in entities if str(x).strip()))[:15]
    except Exception:
        return []


def hybrid_search(document_id: str, question: str, top_k: int) -> List[Dict[str, Any]]:
    vector = vector_search(document_id, question, max(top_k, 5))
    graph = graph_search(document_id, question, max(top_k, 5))
    graph_map = {item["chunk_id"]: item for item in graph}

    merged = []
    for item in vector:
        graph_item = graph_map.get(item["chunk_id"], {})
        graph_score = float(graph_item.get("graph_score", 0))
        vector_score = float(item.get("vector_score", 0))
        normalized_graph = min(graph_score / 3.0, 1.0)
        hybrid = 0.7 * max(vector_score, 0.0) + 0.3 * normalized_graph
        merged.append(
            {
                **item,
                "graph_score": round(normalized_graph, 6),
                "hybrid_score": round(hybrid, 6),
                "entities": graph_item.get("entities", []),
            }
        )

    # Graph-only hits are included so graph retrieval is genuinely independent.
    existing = {x["chunk_id"] for x in merged}
    for item in graph:
        if item["chunk_id"] in existing:
            continue
        merged.append(
            {
                **item,
                "vector_score": 0.0,
                "graph_score": round(min(float(item.get("graph_score", 0)) / 3.0, 1.0), 6),
                "hybrid_score": round(0.3 * min(float(item.get("graph_score", 0)) / 3.0, 1.0), 6),
            }
        )

    merged.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
    return merged[:top_k]


def heuristic_relevance(question: str, results: List[Dict[str, Any]]) -> bool:
    if not results:
        return False
    q_terms = set(re.findall(r"[a-zA-Z0-9]{3,}", question.lower()))
    context = " ".join(r.get("text", "") for r in results).lower()
    overlap = sum(1 for term in q_terms if term in context)
    top_score = max(float(r.get("hybrid_score", 0)) for r in results)
    return top_score >= 0.25 or overlap >= max(1, min(3, len(q_terms)))


def is_context_relevant(question: str, results: List[Dict[str, Any]]) -> bool:
    if not results:
        return False
    context = "\n\n".join(f"[{i+1}] {r['text']}" for i, r in enumerate(results))[:12000]
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Decide whether the supplied document passages contain enough relevant information to answer the user's question. Return only JSON {\"relevant\":true/false}. Do not use outside knowledge.",
                },
                {"role": "user", "content": f"Question: {question}\n\nPassages:\n{context}"},
            ],
        )
        return bool(json.loads(response.choices[0].message.content or "{}").get("relevant", False))
    except Exception:
        return heuristic_relevance(question, results)


def is_greeting(question: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    greetings = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "how are you doing",
    }
    return normalized in greetings


def generate_greeting_answer(question: str) -> str:
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "You are a friendly PDF RAG assistant. Respond naturally to a simple greeting. Do not perform document retrieval or web search.",
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content or "Hello! How can I help you?"


def generate_document_answer(question: str, results: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(f"[Source {i+1}, page {r.get('page', '?')}] {r['text']}" for i, r in enumerate(results))
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": "Answer using only the provided document context. If the context does not support a claim, say that it is not stated. Cite supporting passages as [Source N].",
            },
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
        ],
    )
    return response.choices[0].message.content or "I could not generate an answer."


def web_search(question: str) -> List[Dict[str, Any]]:
    if not tavily_client:
        return []
    result = tavily_client.search(query=question, search_depth="advanced", max_results=5)
    return result.get("results", [])


def generate_web_answer(question: str, results: List[Dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"[{i+1}] {r.get('title', '')}\n{r.get('content', '')}\nURL: {r.get('url', '')}"
        for i, r in enumerate(results)
    )
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": "Answer from the web search evidence provided. Clearly say that the document did not contain enough relevant information. Include source URLs when available.",
            },
            {"role": "user", "content": f"Question: {question}\n\nWeb evidence:\n{context}"},
        ],
    )
    return response.choices[0].message.content or "I could not generate a web answer."


# NeMo Guardrails is intentionally initialized once. If unavailable, a deterministic
# local safety filter remains active so the app does not silently lose input protection.
guardrails = None
try:
    from nemoguardrails import LLMRails, RailsConfig
    guardrails = LLMRails(RailsConfig.from_path(str(GUARDRAILS_DIR)))
except Exception:
    guardrails = None


BLOCKED_PATTERNS = [
    r"ignore (all|any|the) (previous|prior|system) instructions",
    r"reveal (your|the) system prompt",
    r"show me (your|the) hidden prompt",
    r"bypass (the )?(safety|guardrail|security)",
    r"jailbreak",
]


async def validate_input(question: str) -> None:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, question, flags=re.IGNORECASE):
            raise HTTPException(status_code=400, detail="Request blocked by safety guardrails.")
    if guardrails is not None:
        try:
            from nemoguardrails.rails.llm.options import RailStatus, RailType
            result = await guardrails.check_async(
                [{"role": "user", "content": question}], rail_types=[RailType.INPUT]
            )
            if result.status == RailStatus.BLOCKED:
                raise HTTPException(status_code=400, detail="Request blocked by safety guardrails.")
        except HTTPException:
            raise
        except Exception:
            # Fail closed only for explicit guardrail failures is too disruptive for a demo;
            # deterministic filters above remain active and the request can continue.
            pass


def build_document_metadata(document_id: str, filename: str, file_hash: str, chunks: List[Dict[str, Any]], pdf_path: Path, faiss_path: Path):
    return {
        "document_id": document_id,
        "filename": filename,
        "sha256": file_hash,
        "chunk_count": len(chunks),
        "pdf_path": str(pdf_path.relative_to(ROOT)),
        "faiss_path": str(faiss_path.relative_to(ROOT)),
        "chunks": chunks,
    }


@app.get("/health")
def health():
    neo4j_ok = False
    if neo4j_driver:
        try:
            neo4j_driver.verify_connectivity()
            neo4j_ok = True
        except Exception:
            pass
    return {
        "status": "ok",
        "neo4j": neo4j_ok,
        "tavily": bool(tavily_client),
        "guardrails": bool(guardrails),
    }


@app.get("/documents")
def list_documents():
    docs = []
    for d in load_documents():
        docs.append({k: d.get(k) for k in ["document_id", "filename", "sha256", "chunk_count"]})
    return {"documents": docs}


@app.get("/documents/{document_id}")
def document_details(document_id: str):
    document = get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return {k: document.get(k) for k in ["document_id", "filename", "sha256", "chunk_count"]}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty")

    file_hash = sha256_bytes(content)
    existing = next((d for d in load_documents() if d.get("sha256") == file_hash), None)
    if existing:
        return {
            "status": "duplicate",
            "document_id": existing["document_id"],
            "filename": existing["filename"],
            "chunk_count": existing.get("chunk_count", 0),
            "message": "This PDF is already indexed.",
        }

    document_id = str(uuid.uuid4())
    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"
    pdf_path.write_bytes(content)

    try:
        pages = extract_pdf_pages(pdf_path)
        chunks = build_chunks(pages)
        if not chunks:
            pdf_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="No extractable text found in PDF")
        for chunk in chunks:
            chunk["entities"] = extract_entities(chunk["text"])
        faiss_path = save_faiss(document_id, chunks)
        graph_store(document_id, chunks)
        metadata = build_document_metadata(document_id, file.filename, file_hash, chunks, pdf_path, faiss_path)
        documents = load_documents()
        documents.append(metadata)
        save_documents(documents)
        return {
            "status": "created",
            "document_id": document_id,
            "filename": file.filename,
            "chunk_count": len(chunks),
            "message": "PDF indexed successfully.",
        }
    except HTTPException:
        raise
    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        (FAISS_DIR / f"{document_id}.index").unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {exc}")


@app.post("/retrieve")
def retrieve(request: RetrieveRequest):
    # Retrieve endpoint is useful for debugging/demo evidence.
    documents = load_documents()
    if not documents:
        raise HTTPException(status_code=404, detail="No documents indexed")
    document_id = documents[0]["document_id"]
    results = hybrid_search(document_id, request.question, request.top_k)
    return {
        "document_id": document_id,
        "vector_results": vector_search(document_id, request.question, request.top_k),
        "graph_results": graph_search(document_id, request.question, request.top_k),
        "hybrid_results": results,
    }


@app.post("/documents/{document_id}/chat")
async def chat(document_id: str, request: ChatRequest):
    if not get_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")
    await validate_input(request.question)

    if is_greeting(request.question):
        return {
            "answer": generate_greeting_answer(request.question),
            "source": "greeting",
            "web_used": False,
            "retrieval": [],
        }

    results = hybrid_search(document_id, request.question, request.top_k)
    relevant = is_context_relevant(request.question, results)

    if relevant:
        answer = generate_document_answer(request.question, results)
        return {
            "answer": answer,
            "source": "hybrid",
            "web_used": False,
            "retrieval": results,
        }

    web_results = web_search(request.question)
    if web_results:
        answer = generate_web_answer(request.question, web_results)
        return {
            "answer": answer,
            "source": "web",
            "web_used": True,
            "retrieval": results,
            "web_results": web_results,
        }

    return {
        "answer": "The document does not contain enough relevant information to answer this question, and web search is not configured or returned no results.",
        "source": "none",
        "web_used": False,
        "retrieval": results,
    }
