# Backend Package Guide

`project/` is the Python backend for Xinyu Medical Agent.  It contains the
FastAPI API layer, LangGraph orchestration, RAG ingestion/retrieval, memory,
MCP hospital integration, and the Gradio admin/debug console.

For the full repository map, see `docs/PROJECT_STRUCTURE_CN.md`.

## Main Entry Points

| Path | Purpose |
| --- | --- |
| `api_app.py` | Starts the FastAPI server with Uvicorn. |
| `app.py` | Starts the Gradio admin/debug console. |
| `config.py` | Central environment-driven configuration. |
| `kb_jobs.py` | Knowledge-base maintenance jobs, such as bootstrap and official-source sync. |
| `import_medical_sources.py` | CLI-oriented medical source import helpers. |

## Package Map

| Directory | Purpose |
| --- | --- |
| `api/` | HTTP/SSE adapter layer for React. It handles auth, DTOs, route registration, and delegates business logic to core services. |
| `core/` | System bootstrap, document management, chunking, source sync, QA evaluation, and observability. |
| `rag_agent/` | LangGraph state, routing, edges, prompts, retrieval nodes, appointment nodes, and graph compilation. |
| `skills/` | Skill plugin framework for greeting, triage, booking, cancellation, medical RAG, and future intents. |
| `services/` | Business services used by skills, especially appointment workflows and mock hospital backends. |
| `mcp_integration/` | Per-user MCP hospital credentials, tool pools, registry, token encryption, and MCP skill routing. |
| `db/` | PostgreSQL schema, connection pooling, pgvector/tsvector retrieval, audit logs, sessions, import tasks, and stores. |
| `memory/` | Redis session memory, summary persistence, user memory extraction, and long-term semantic memory. |
| `ui/` | Gradio admin/debug console. React is in the top-level `frontend/` directory. |
| `benchmarks/` | Offline and regression evaluations for routing, retrieval, memory, and answer quality. |
| `assets/` | Backend-local static assets, such as the chatbot avatar. |

## Current Storage Model

The current retrieval layer is PostgreSQL-based:

- `pgvector` stores dense embeddings.
- PostgreSQL `tsvector` supports lexical search.
- Retrieval can use hybrid fusion and optional reranking.
- Parent chunks are still represented through `ParentStoreManager`, but the active document and child-chunk inventory is managed through PostgreSQL tables.

Some names such as `QDRANT_DB_PATH` and `SPARSE_MODEL = "Qdrant/bm25"` remain as backward-compatible configuration aliases. They are not the current primary storage architecture.

## Request Flow

```text
React frontend
  -> FastAPI route in api/routes/
  -> ChatInterface
  -> RAGSystem service container
  -> LangGraph graph in rag_agent/graph.py
  -> skill / RAG / appointment / MCP / memory services
  -> streamed SSE response
```

## Extension Points

- Add a new user intent: create a `BaseSkill` implementation in `skills/` and register it during `RAGSystem.initialize()`.
- Add or change graph behavior: update `rag_agent/graph.py`, `rag_agent/edges.py`, and the relevant node module.
- Add a document source: extend `core/medical_source_ingest.py` or `core/knowledge_base_sync.py`.
- Add an API surface: create a route under `api/routes/`, expose DTOs in `api/schemas.py`, and keep core logic outside the API layer.

## Runtime Data

Do not commit generated runtime data. The root `.gitignore` excludes:

- `runtime/`
- `markdown_docs/`
- `parent_store/`
- `qdrant_db/`
- `.env` and `project/.env`
- frontend build and dependency folders
