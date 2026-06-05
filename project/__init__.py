"""Agentic RAG for medical consultation — a LangGraph-powered assistant with:

- Intent routing (rule + LLM two-stage)
- Hybrid retrieval (pgvector + tsvector + rerank)
- Controlled appointment booking skill
- Multi-turn memory (Redis + LLM summary + PostgreSQL)
- Tiered LLM routing with circuit-breaker fallback
- Pluggable skill registration framework
"""
