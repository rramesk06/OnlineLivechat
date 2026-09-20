# PDF RAG + Graph RAG AI Assistant

A production-style learning project that combines PDF RAG, FAISS vector retrieval, a Neo4j knowledge graph, hybrid retrieval, relevance-based web fallback, NeMo Guardrails, DeepEval evaluation, FastAPI, and Streamlit.

## Architecture

```text
                    ┌─────────────────────┐
                    │     Streamlit UI    │
                    └──────────┬──────────┘
                               │ HTTP
                    ┌──────────▼──────────┐
                    │      FastAPI        │
                    │                     │
                    │  Input Guardrails   │
                    └──────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
         ┌──────▼─────┐ ┌──────▼──────┐ ┌────▼─────┐
         │   FAISS    │ │   Neo4j     │ │  Tavily  │
         │Vector RAG  │ │  Graph RAG  │ │ Web      │
         └──────┬─────┘ └──────┬──────┘ └────┬─────┘
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                       Hybrid / Fallback
                               │
                               ▼
                           OpenAI LLM
```

## Features

- Upload a PDF once; duplicate PDFs are detected using SHA-256.
- Extract PDF text with PyMuPDF.
- Split text with recursive chunking.
- Generate Sentence Transformer embeddings.
- Store vectors in FAISS using cosine-style inner product with normalized embeddings.
- Extract technical entities and store document/chunk/entity relationships in Neo4j.
- Run vector retrieval and graph retrieval independently.
- Combine them using a transparent 70/30 hybrid score baseline.
- Judge whether retrieved context is relevant before answering.
- Fall back to Tavily web search when the PDF context is not sufficient.
- Run NeMo Guardrails input checks plus deterministic prompt-injection filters.
- Evaluate the RAG pipeline with DeepEval metrics.
- Streamlit UI keeps indexed documents available after restart.
- Docker Compose runs backend and frontend separately.

## Project layout

```text
backend/main.py
frontend/app.py
guardrails/config.yml
evaluation/evaluate.py
evaluation/test_cases.json
data/uploads/
data/faiss/
requirements.txt
.env.example
Dockerfile files
docker-compose.yml
```

## Environment

Copy `.env.example` to `.env` and fill in:

- `OPENAI_API_KEY`
- `TAVILY_API_KEY`
- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

Never commit `.env`.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

Backend:

```bash
uvicorn backend.main:app --reload --port 8000
```

Frontend:

```bash
streamlit run frontend/app.py
```

Open the Streamlit URL shown by the terminal.

## Docker

```bash
docker compose build
docker compose up -d
```

Backend: `http://localhost:8000`

Frontend: `http://localhost:8501`

## Evaluation

The evaluation suite is intentionally separate from production request handling. Once the API is running and a PDF is indexed:

```bash
python evaluation/evaluate.py
```

DeepEval uses LLM-as-a-judge metrics for answer relevancy, faithfulness, contextual relevancy, contextual precision, and contextual recall.

## Retrieval behavior

1. The user asks a question.
2. Input is checked by deterministic safety filters and NeMo Guardrails.
3. FAISS retrieves semantic matches.
4. Neo4j retrieves chunks connected to relevant entities.
5. Results are merged and ranked with a hybrid score.
6. A relevance decision determines whether document context is sufficient.
7. If sufficient, the LLM answers from document context only.
8. If insufficient, Tavily searches the web and the LLM answers from the returned evidence.
9. The UI displays the selected source and retrieval evidence.

## Important design choice

This implementation does **not** make the application dependent on web search for normal document questions. The PDF remains the primary knowledge source. Web search is a fallback only when the retrieved document context is judged insufficient.

## Assignment completion checklist

- [x] PDF upload
- [x] Duplicate prevention
- [x] PDF text extraction
- [x] Chunking
- [x] Embeddings
- [x] FAISS vector DB/index
- [x] Neo4j knowledge graph
- [x] Independent vector retrieval
- [x] Independent graph retrieval
- [x] Hybrid retrieval
- [x] Web fallback
- [x] NeMo Guardrails integration
- [x] DeepEval evaluation suite
- [x] Streamlit final UI
- [x] Docker
- [x] GitHub-ready `.gitignore`

Deployment to a VPS should expose Streamlit through Nginx/HTTPS and keep FastAPI internal where possible.
