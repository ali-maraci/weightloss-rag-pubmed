# WeightLoss RAG — GLP-1 & Weight Loss Research Assistant

A conversational RAG (Retrieval-Augmented Generation) system for searching, analysing, and synthesising PubMed literature on GLP-1 medications and weight loss. Ask questions about semaglutide, tirzepatide, Ozempic, Wegovy, Mounjaro, side effects, safety, and nutrition — and get evidence-backed answers with inline citations linked directly to PubMed.

---

## Architecture Overview

The system is split into three decoupled stages: data pipeline, backend API, and frontend.

```
PubMed (NCBI API)
      │
      ▼
  [1] Ingestion          batch_ingest_weightloss.py
      Raw XML / JSON  →  data/raw/weightloss/
      │
      ▼
  [2] Chunking           run_chunking.py + AuraChunker
      Processed JSON  →  data/processed/weightloss/
      │
      ▼
  [3] Embedding          embed_to_qdrant.py + AuraEmbedder
      Qdrant Cloud    ←  Dense (OpenAI) + Sparse (BM25)
      │
      ▼
  [4] Query Pipeline     FastAPI + LangChain
      4-stage hybrid retrieval → SSE stream
      │
      ▼
  [5] Frontend           React 19 + Vite + MUI
      http://localhost:5173
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | React 19, TypeScript, Vite, MUI v7, SCSS |
| **Backend** | Python 3.12, FastAPI, Uvicorn |
| **LLM** | OpenAI `gpt-4o-mini` |
| **Embeddings** | OpenAI `text-embedding-3-small` (dense), FastEmbedSparse BM25 (sparse) |
| **Vector DB** | Qdrant Cloud |
| **Orchestration** | LangChain |
| **Data source** | PubMed / NCBI Entrez API |
| **Hosting** | Firebase Hosting (frontend) |

---

## Data Pipeline

### Step 1 — Ingest raw papers from PubMed

```bash
export PYTHONPATH=$(pwd)
python scripts/batch_ingest_weightloss.py
```

Downloads abstracts and full-text XMLs from PubMed using targeted query groups split by era and topic:

- GLP-1 drugs 2010–2016 (exenatide, liraglutide, dulaglutide)
- GLP-1 drugs 2017–2020 (semaglutide, Ozempic added)
- GLP-1 drugs 2021–2023 (tirzepatide, Wegovy, Mounjaro)
- GLP-1 drugs 2024–2026 (latest research)
- GLP-1 side effects & safety (cardiovascular, pancreatitis, thyroid, tolerability)
- Nutrition & muscle during GLP-1 (sarcopenia, lean body mass, dietary protein)

Output: `data/raw/weightloss/` — one JSON per PMID. Script is resumable; already-downloaded PMIDs are skipped.

### Step 2 — Chunk and preprocess

```bash
export PYTHONPATH=$(pwd)
python scripts/run_chunking.py --folder weightloss
```

Produces two indices per article:
- **Index A (Abstracts)** — full abstract as a single document for conceptual matching
- **Index B (Full Text Body)** — split by Markdown headers (Results, Methods, etc.), then recursive character chunking

Output: `data/processed/weightloss/`

### Step 3 — Embed and upload to Qdrant

```bash
# One-time: create Qdrant collections
export PYTHONPATH=$(pwd)
python scripts/setup_qdrant_collections.py

# Upload embeddings
python scripts/embed_to_qdrant.py --folder weightloss 2>&1 | tee logs/embed.log
```

Creates two Qdrant collections:
- `aura_index_a_abstracts` — 1536-dim cosine + BM25 sparse
- `aura_index_b_bodies` — 1536-dim cosine + BM25 sparse

Duplicate-safe: checks existing Qdrant points before uploading.

---

## Running the System

### Prerequisites

- Python 3.12 (3.13+ not supported due to `onnxruntime`)
- Node.js 18+
- API keys: OpenAI, NCBI Entrez, Qdrant Cloud

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Where to get it |
|---|---|
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `NCBI_API_KEY` / `NCBI_EMAIL` | [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/) — key is optional but raises rate limit |
| `QDRANT_URL` / `QDRANT_API_KEY` | [cloud.qdrant.io](https://cloud.qdrant.io) — free 1GB cluster |
| `LANGCHAIN_*` | [smith.langchain.com](https://smith.langchain.com) — optional, set `LANGCHAIN_TRACING_V2=false` to disable |

### Backend

```bash
source venv/bin/activate
export PYTHONPATH=$(pwd)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

### First-time Python setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -e .
```

---

## RAG Query Pipeline (4 stages)

Every user query goes through four retrieval stages before reaching the LLM:

1. **LLM Query Parsing** — `gpt-4o-mini` expands the query for hybrid search and strips metadata filters (author, year) from the semantic query
2. **Abstract Discovery** — hybrid search on Index A to identify candidate PMIDs
3. **Deep Chunk Search** — semantic + keyword search on Index B, scoped to candidate PMIDs only
4. **Reranking + Diversity Filtering** — boosts by study type (RCT > case report), section (Results > Introduction), and recency; caps chunks per paper to prevent single papers from dominating

Responses are streamed token-by-token over SSE with inline citations formatted as `(Author, Year) [PMID: 12345]` — each citation is a clickable PubMed link.

---

## Project Layout

```
weightloss-rag-pubmed/
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── api/                       # REST + SSE endpoints
│   ├── core/                      # RAG logic: retriever, chat engine, QA chain
│   ├── db/                        # Qdrant vector store + NCBI client wrappers
│   ├── models/                    # Pydantic schemas
│   └── utils/                     # Config (pydantic-settings)
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Main chat component
│   │   ├── App.scss               # Component styles
│   │   ├── index.css              # Global styles (citation links, markdown)
│   │   └── services/
│   │       └── chatService.ts     # SSE streaming client
│   ├── vite.config.ts
│   ├── package.json
│   └── firebase.json              # Firebase Hosting config (public: dist)
├── scripts/
│   ├── batch_ingest_weightloss.py # Step 1: PubMed ingestion
│   ├── run_chunking.py            # Step 2: chunk + preprocess
│   ├── setup_qdrant_collections.py # One-time Qdrant setup
│   ├── embed_to_qdrant.py         # Step 3: embed + upload
│   ├── run_chat.py                # CLI chat interface
│   ├── run_query.py               # Single-shot query
│   └── run_evaluation.py          # LLM-as-a-judge evaluation
├── data/
│   ├── raw/weightloss/            # Downloaded PubMed JSONs
│   └── processed/weightloss/      # Chunked article JSONs
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

---

## Deploy to Firebase

```bash
cd frontend
npm run build
firebase deploy --only hosting
```

Ensure `firebase.json` has `"public": "dist"` (Vite output, not Angular's `dist/frontend/browser`).

---

## How Answers Are Generated

This is a **retrieval-augmented** system, not a pure LLM chatbot. Here is what each component actually does:

| Component | Role |
|---|---|
| **Qdrant vector database** | Stores only the PubMed papers you ingested — no external knowledge |
| **Retrieved chunks** | Injected verbatim into every prompt as the `{context}` block |
| **`gpt-4o-mini`** | Reads the retrieved chunks and writes the response in natural language |
| **Citations** | Pulled directly from retrieved document metadata (real PMIDs) |

**The LLM is instructed to answer exclusively from the provided context.** The system prompt explicitly states:

> *"You must answer the user's question based explicitly on the Context Provided. If the context does not contain the answer, state clearly: 'I couldn't find sufficient evidence in the literature to answer this question.' Do not attempt to guess or use outside knowledge."*

If no relevant papers are found in either Index A or the fallback Index B search, the system will say so rather than fabricating an answer.

**Important caveat:** This constraint is enforced by prompt instructions, not by technical guardrails. `gpt-4o-mini` still carries its own training knowledge, and like all instruction-following LLMs, could theoretically ignore the constraint in edge cases. Citations always link to real PubMed articles that were retrieved — if a citation appears in the response, that paper exists in the database and was used as a source.

---

## Required API Accounts

| Service | Purpose | Free tier |
|---|---|---|
| [OpenAI](https://platform.openai.com) | LLM (`gpt-4o-mini`) + embeddings | Pay-per-use |
| [NCBI Entrez](https://www.ncbi.nlm.nih.gov/account/) | PubMed paper download | Free (key increases rate limit) |
| [Qdrant Cloud](https://cloud.qdrant.io) | Vector database | Free 1GB cluster |
| [Firebase](https://console.firebase.google.com) | Frontend hosting | Free Spark plan |
| [LangSmith](https://smith.langchain.com) | Tracing (optional) | Free tier available |
