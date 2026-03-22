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

Create a `.env` file at the project root:

```env
OPENAI_API_KEY=sk-...
NCBI_API_KEY=your_ncbi_key
NCBI_EMAIL=your@email.com
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_key
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=weightloss-rag-pubmed
```

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

## Required API Accounts

| Service | Purpose | Free tier |
|---|---|---|
| [OpenAI](https://platform.openai.com) | LLM (`gpt-4o-mini`) + embeddings | Pay-per-use |
| [NCBI Entrez](https://www.ncbi.nlm.nih.gov/account/) | PubMed paper download | Free (key increases rate limit) |
| [Qdrant Cloud](https://cloud.qdrant.io) | Vector database | Free 1GB cluster |
| [Firebase](https://console.firebase.google.com) | Frontend hosting | Free Spark plan |
| [LangSmith](https://smith.langchain.com) | Tracing (optional) | Free tier available |
