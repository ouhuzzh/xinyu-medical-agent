"""Central configuration — reads project/.env and exposes all settings as module-level constants.

All env vars have sensible defaults for local development. See .env.example for the
full list of overridable settings.

Groups:
    - Directory paths (MARKDOWN_DIR, RUNTIME_DIR, etc.)
    - Multi-provider model config (LLM, embedding, rerank)
    - PostgreSQL + pgvector connection
    - Redis session memory
    - API auth, CORS, rate limiting, upload limits
    - RAG relevance/confidence thresholds
    - Knowledge base sync (official sources, scheduling)
    - Text splitter (chunk size, overlap)
    - Agent graph limits (iterations, tool calls, recursion)
    - Observability (Langfuse)
"""

from __future__ import annotations

import json
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# --- Directory Configuration ---
_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_RUNTIME_DIR = os.path.join(_BASE_DIR, "runtime")


def _load_json_mapping(env_name: str, default: dict | None = None) -> dict:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return dict(default or {})
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Environment variable `{env_name}` must be valid JSON.") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Environment variable `{env_name}` must decode to a JSON object.")
    return data

MARKDOWN_DIR = os.path.join(_BASE_DIR, "markdown_docs")
PARENT_STORE_PATH = os.path.join(_BASE_DIR, "parent_store")
QDRANT_DB_PATH = os.path.join(_BASE_DIR, "qdrant_db")
LANGGRAPH_CHECKPOINT_PATH = os.environ.get(
    "LANGGRAPH_CHECKPOINT_PATH",
    os.path.join(_RUNTIME_DIR, "langgraph_checkpoints.pkl"),
)
VECTOR_DIMENSION = int(os.environ.get("VECTOR_DIMENSION", "1024"))

# --- Qdrant Configuration ---
CHILD_COLLECTION = "document_child_chunks"
SPARSE_VECTOR_NAME = "sparse"

# --- Multi-Provider Model Configuration ---
ACTIVE_LLM_PROVIDER = os.environ.get("ACTIVE_LLM_PROVIDER", "deepseek").lower()
ACTIVE_EMBEDDING_PROVIDER = os.environ.get("ACTIVE_EMBEDDING_PROVIDER", "openai_compatible").lower()

LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-32B")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "45"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2048"))
LLM_STRUCTURED_MAX_TOKENS = int(os.environ.get("LLM_STRUCTURED_MAX_TOKENS", "384"))
ENABLE_RERANK = os.environ.get("ENABLE_RERANK", "true").lower() == "true"
RERANK_FETCH_K = int(os.environ.get("RERANK_FETCH_K", "12"))
ENABLE_HYBRID_RETRIEVAL = os.environ.get("ENABLE_HYBRID_RETRIEVAL", "true").lower() == "true"
KEYWORD_FETCH_K = int(os.environ.get("KEYWORD_FETCH_K", "8"))
RAG_HIGH_CONFIDENCE_SCORE = float(os.environ.get("RAG_HIGH_CONFIDENCE_SCORE", "0.85"))
RAG_MEDIUM_CONFIDENCE_SCORE = float(os.environ.get("RAG_MEDIUM_CONFIDENCE_SCORE", "0.72"))
RAG_HIGH_RELEVANCE_SCORE = float(os.environ.get("RAG_HIGH_RELEVANCE_SCORE", "0.86"))
RAG_MEDIUM_RELEVANCE_SCORE = float(os.environ.get("RAG_MEDIUM_RELEVANCE_SCORE", "0.74"))
RAG_LOW_RELEVANCE_SCORE = float(os.environ.get("RAG_LOW_RELEVANCE_SCORE", "0.70"))
RAG_HIGH_LEXICAL_OVERLAP = float(os.environ.get("RAG_HIGH_LEXICAL_OVERLAP", "0.55"))
RAG_MEDIUM_LEXICAL_OVERLAP = float(os.environ.get("RAG_MEDIUM_LEXICAL_OVERLAP", "0.30"))
RAG_LOW_LEXICAL_OVERLAP = float(os.environ.get("RAG_LOW_LEXICAL_OVERLAP", "0.12"))
RAG_DIRECT_EVIDENCE_SCORE = float(os.environ.get("RAG_DIRECT_EVIDENCE_SCORE", "0.84"))
RAG_LIMITED_EVIDENCE_SCORE = float(os.environ.get("RAG_LIMITED_EVIDENCE_SCORE", "0.76"))
RAG_RETRY_LIMIT = int(os.environ.get("RAG_RETRY_LIMIT", "1"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
OPENAI_ENABLE_THINKING = os.environ.get("OPENAI_ENABLE_THINKING", "false").lower() == "true"
OPENAI_THINKING_BUDGET = int(os.environ.get("OPENAI_THINKING_BUDGET", "1024"))
RERANK_API_KEY = os.environ.get("RERANK_API_KEY", OPENAI_API_KEY)
RERANK_BASE_URL = os.environ.get("RERANK_BASE_URL", "https://api.siliconflow.cn/v1/rerank")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Tiered LLM Routing ---
LLM_TIERS_JSON = os.environ.get("LLM_TIERS_JSON", "")
LLM_FALLBACK_PROVIDER = os.environ.get("LLM_FALLBACK_PROVIDER", "")

# --- Skill Plugin Framework ---
SKILLS_ENABLED = os.environ.get("SKILLS_ENABLED", "false").lower() == "true"

# Backward-compatible aliases used by the current vector layer.
DENSE_MODEL = EMBEDDING_MODEL
SPARSE_MODEL = os.environ.get("SPARSE_MODEL", "Qdrant/bm25")

# --- PostgreSQL Configuration ---
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "ai_companion")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")
VECTOR_INDEX_LISTS = int(os.environ.get("VECTOR_INDEX_LISTS", "100"))
AUTO_BOOTSTRAP_KNOWLEDGE_BASE = os.environ.get("AUTO_BOOTSTRAP_KNOWLEDGE_BASE", "false").lower() == "true"
STATUS_REFRESH_SECONDS = float(os.environ.get("STATUS_REFRESH_SECONDS", "2"))
RECENT_IMPORT_TASK_LIMIT = int(os.environ.get("RECENT_IMPORT_TASK_LIMIT", "8"))
ENABLE_KB_SYNC_SCHEDULER = os.environ.get("ENABLE_KB_SYNC_SCHEDULER", "false").lower() == "true"
KB_SYNC_INTERVAL_HOURS = int(os.environ.get("KB_SYNC_INTERVAL_HOURS", "24"))
KB_SYNC_OFFICIAL_SOURCES = [
    item.strip().lower()
    for item in os.environ.get("KB_SYNC_OFFICIAL_SOURCES", "medlineplus,nhc,who").split(",")
    if item.strip()
]
KB_SOFT_DELETE_MISSING = os.environ.get("KB_SOFT_DELETE_MISSING", "true").lower() == "true"
KB_REPLACE_LOCAL_DUPLICATES = os.environ.get("KB_REPLACE_LOCAL_DUPLICATES", "true").lower() == "true"

# --- Runtime / App Mode ---
APP_ENV = os.environ.get("APP_ENV", "development").strip().lower() or "development"

# --- API / Frontend Configuration ---
API_CORS_ORIGINS = [
    item.strip()
    for item in os.environ.get(
        "API_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if item.strip()
]
API_UPLOAD_MAX_FILES = int(os.environ.get("API_UPLOAD_MAX_FILES", "5"))
API_UPLOAD_MAX_FILE_SIZE_MB = int(os.environ.get("API_UPLOAD_MAX_FILE_SIZE_MB", "20"))
API_RATE_LIMIT_CHAT_PER_MINUTE = int(os.environ.get("API_RATE_LIMIT_CHAT_PER_MINUTE", "20"))
API_RATE_LIMIT_UPLOADS_PER_MINUTE = int(os.environ.get("API_RATE_LIMIT_UPLOADS_PER_MINUTE", "6"))
API_RATE_LIMIT_SYNCS_PER_MINUTE = int(os.environ.get("API_RATE_LIMIT_SYNCS_PER_MINUTE", "3"))

# --- JWT Authentication ---
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change-me-in-production-please")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24h
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.environ.get("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30"))
PASSWORD_MIN_LENGTH = int(os.environ.get("PASSWORD_MIN_LENGTH", "6"))

_DEFAULT_AUTH_TOKENS = (
    {
        "demo-user-token": {"user_id": "demo-user", "role": "user"},
        "other-user-token": {"user_id": "other-user", "role": "user"},
        "demo-admin-token": {"user_id": "demo-admin", "role": "admin"},
    }
    if APP_ENV == "development"
    else {}
)
API_AUTH_TOKENS = _load_json_mapping("API_AUTH_TOKENS_JSON", default=_DEFAULT_AUTH_TOKENS)

# --- Redis Configuration ---
REDIS_ENABLED = os.environ.get("REDIS_ENABLED", "true").lower() == "true"
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
REDIS_TTL_SECONDS = int(os.environ.get("REDIS_TTL_SECONDS", "86400"))
SHORT_TERM_WINDOW_SIZE = int(os.environ.get("SHORT_TERM_WINDOW_SIZE", "12"))
RECENT_CONTEXT_TURNS = int(os.environ.get("RECENT_CONTEXT_TURNS", "3"))
SUMMARY_REFRESH_THRESHOLD = int(os.environ.get("SUMMARY_REFRESH_THRESHOLD", "4"))

# --- User Memory Configuration ---
USER_MEMORY_ENABLED = os.environ.get("USER_MEMORY_ENABLED", "true").lower() == "true"
USER_MEMORY_EXTRACTION_ENABLED = os.environ.get("USER_MEMORY_EXTRACTION_ENABLED", "true").lower() == "true"
USER_MEMORY_INJECTION_ENABLED = os.environ.get("USER_MEMORY_INJECTION_ENABLED", "true").lower() == "true"
USER_MEMORY_MAX_RETRIEVED = int(os.environ.get("USER_MEMORY_MAX_RETRIEVED", "5"))
USER_MEMORY_IMPORTANCE_THRESHOLD = int(os.environ.get("USER_MEMORY_IMPORTANCE_THRESHOLD", "4"))
USER_MEMORY_DEDUP_SIMILARITY = float(os.environ.get("USER_MEMORY_DEDUP_SIMILARITY", "0.9"))
USER_MEMORY_RECENCY_WEIGHT = float(os.environ.get("USER_MEMORY_RECENCY_WEIGHT", "0.3"))
USER_MEMORY_IMPORTANCE_WEIGHT = float(os.environ.get("USER_MEMORY_IMPORTANCE_WEIGHT", "0.4"))
USER_MEMORY_RELEVANCE_WEIGHT = float(os.environ.get("USER_MEMORY_RELEVANCE_WEIGHT", "0.3"))

# --- Episodic Memory Configuration (L4) ---
# 默认关闭，需要时在 .env 中设 EPISODIC_MEMORY_ENABLED=true
EPISODIC_MEMORY_ENABLED = os.environ.get("EPISODIC_MEMORY_ENABLED", "false").lower() == "true"
EPISODIC_MEMORY_MAX_RETRIEVED = int(os.environ.get("EPISODIC_MEMORY_MAX_RETRIEVED", "3"))

# --- Core Memory Configuration (L5) ---
# 默认关闭，需要时在 .env 中设 CORE_MEMORY_ENABLED=true
CORE_MEMORY_ENABLED = os.environ.get("CORE_MEMORY_ENABLED", "false").lower() == "true"
CORE_MEMORY_MAX_LENGTH = int(os.environ.get("CORE_MEMORY_MAX_LENGTH", "2000"))

# --- Reflection Memory Configuration (L6) ---
# 默认关闭，需要时在 .env 中设 REFLECTION_MEMORY_ENABLED=true
REFLECTION_MEMORY_ENABLED = os.environ.get("REFLECTION_MEMORY_ENABLED", "false").lower() == "true"
REFLECTION_MEMORY_MIN_SOURCE_COUNT = int(os.environ.get("REFLECTION_MEMORY_MIN_SOURCE_COUNT", "3"))
REFLECTION_MEMORY_INTERVAL_TURNS = int(os.environ.get("REFLECTION_MEMORY_INTERVAL_TURNS", "10"))

HIGH_RISK_KEYWORDS = [
    "胸痛",
    "胸闷",
    "呼吸困难",
    "呼吸急促",
    "意识模糊",
    "抽搐",
    "大出血",
    "持续高热",
    "晕厥",
    "severe chest pain",
    "shortness of breath",
    "confusion",
    "convulsion",
    "heavy bleeding",
]

# --- Agent Configuration ---
MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10
GRAPH_RECURSION_LIMIT = 50
BASE_TOKEN_THRESHOLD = 4000
TOKEN_GROWTH_FACTOR = 0.9
ENABLE_PERSISTENT_GRAPH_CHECKPOINT = os.environ.get("ENABLE_PERSISTENT_GRAPH_CHECKPOINT", "true").lower() == "true"

# --- Text Splitter Configuration ---
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 100
MIN_PARENT_SIZE = 2000
MAX_PARENT_SIZE = 4000
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# --- Langfuse Observability ---
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")
