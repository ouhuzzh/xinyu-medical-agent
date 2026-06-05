import config
import psycopg
import threading
from pathlib import Path

_SQL_DIR = Path(__file__).with_name("sql")
_DEMO_APPOINTMENT_SEED_SQL = (_SQL_DIR / "seed_appointment_demo.sql").read_text(encoding="utf-8")


class SchemaManager:
    """Apply lightweight, repeatable PostgreSQL schema migrations."""

    _MIGRATIONS = [
        (
            "001_summary_dedup_and_indexes",
            "Clean duplicate summaries and create stable relational indexes.",
            [
                """
                DELETE FROM chat_session_summaries t
                USING (
                    SELECT ctid
                    FROM (
                        SELECT
                            ctid,
                            ROW_NUMBER() OVER (
                                PARTITION BY thread_id, summary_type
                                ORDER BY updated_at DESC, created_at DESC, id DESC
                            ) AS row_num
                        FROM chat_session_summaries
                    ) ranked
                    WHERE ranked.row_num > 1
                ) dupes
                WHERE t.ctid = dupes.ctid
                """,
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_chat_session_summaries_thread_type
                ON chat_session_summaries(thread_id, summary_type)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_patient_id
                ON chat_sessions(patient_id)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_appointments_patient_status_date
                ON appointments(patient_id, status, appointment_date)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_documents_source_name
                ON documents(source_name)
                """,
            ],
        ),
        (
            "002_child_chunks_vector_index",
            "Install a pgvector ANN index for child chunk recall.",
            [
                f"""
                CREATE INDEX IF NOT EXISTS idx_child_chunks_embedding_cosine
                ON child_chunks
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {config.VECTOR_INDEX_LISTS})
                """,
                "ANALYZE child_chunks",
            ],
        ),
        (
            "003_import_task_logs",
            "Persist recent import task history for the UI.",
            [
                """
                CREATE TABLE IF NOT EXISTS import_task_logs (
                    id                  BIGSERIAL PRIMARY KEY,
                    source              VARCHAR(64) NOT NULL,
                    label               VARCHAR(128),
                    status              VARCHAR(64) NOT NULL DEFAULT 'completed',
                    downloaded          INTEGER NOT NULL DEFAULT 0,
                    written             INTEGER NOT NULL DEFAULT 0,
                    skipped             INTEGER NOT NULL DEFAULT 0,
                    failed              INTEGER NOT NULL DEFAULT 0,
                    index_added         INTEGER NOT NULL DEFAULT 0,
                    index_skipped       INTEGER NOT NULL DEFAULT 0,
                    duration_ms         DOUBLE PRECISION NOT NULL DEFAULT 0,
                    note                TEXT,
                    conversion_details  JSONB NOT NULL DEFAULT '[]'::jsonb,
                    failure_details     JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_import_task_logs_created_at
                ON import_task_logs(created_at DESC)
                """,
            ],
        ),
        (
            "004_route_logs",
            "Persist routing decisions for transcript and route-quality evaluation.",
            [
                """
                CREATE TABLE IF NOT EXISTS route_logs (
                    id                  BIGSERIAL PRIMARY KEY,
                    request_id          VARCHAR(64),
                    thread_id           VARCHAR(128),
                    user_query          TEXT NOT NULL,
                    primary_intent      VARCHAR(64) NOT NULL,
                    secondary_intent    VARCHAR(64) NOT NULL DEFAULT '',
                    decision_source     VARCHAR(32) NOT NULL DEFAULT '',
                    route_reason        TEXT NOT NULL DEFAULT '',
                    had_pending_state   BOOLEAN NOT NULL DEFAULT FALSE,
                    extra_metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_route_logs_created_at
                ON route_logs(created_at DESC)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_route_logs_thread_id
                ON route_logs(thread_id)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_route_logs_request_id
                ON route_logs(request_id)
                """,
            ],
        ),
        (
            "005_appointment_skill_and_retrieval_quality",
            "Persist appointment skill logs and richer retrieval quality metadata.",
            [
                """
                ALTER TABLE retrieval_logs
                ADD COLUMN IF NOT EXISTS query_plan JSONB NOT NULL DEFAULT '[]'::jsonb
                """,
                """
                ALTER TABLE retrieval_logs
                ADD COLUMN IF NOT EXISTS graded_doc_count INTEGER NOT NULL DEFAULT 0
                """,
                """
                ALTER TABLE retrieval_logs
                ADD COLUMN IF NOT EXISTS sufficiency_result VARCHAR(64)
                """,
                """
                ALTER TABLE retrieval_logs
                ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0
                """,
                """
                ALTER TABLE retrieval_logs
                ADD COLUMN IF NOT EXISTS final_confidence_bucket VARCHAR(32)
                """,
                """
                CREATE TABLE IF NOT EXISTS appointment_skill_logs (
                    id                      BIGSERIAL PRIMARY KEY,
                    thread_id               VARCHAR(128),
                    skill_mode              VARCHAR(64) NOT NULL DEFAULT '',
                    request_type            VARCHAR(64) NOT NULL DEFAULT '',
                    selected_candidate_count INTEGER NOT NULL DEFAULT 0,
                    required_confirmation   BOOLEAN NOT NULL DEFAULT FALSE,
                    final_action            VARCHAR(64) NOT NULL DEFAULT '',
                    extra_metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at              TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_appointment_skill_logs_created_at
                ON appointment_skill_logs(created_at DESC)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_appointment_skill_logs_thread_id
                ON appointment_skill_logs(thread_id)
                """,
            ],
        ),
        (
            "006_knowledge_base_sync",
            "Add updatable knowledge-base sync fields and richer import task metrics.",
            [
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS source_key VARCHAR(256)
                """,
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS content_hash VARCHAR(128)
                """,
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS sync_status VARCHAR(32) NOT NULL DEFAULT 'active'
                """,
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE
                """,
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS last_synced_at TIMESTAMP
                """,
                """
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP
                """,
                """
                UPDATE documents
                SET source_key = CASE
                    WHEN lower(coalesce(metadata->>'source', source_name, '')) LIKE 'medlineplus-%'
                        THEN 'official:medlineplus:' || split_part(coalesce(metadata->>'source', source_name, document_no), '.', 1)
                    WHEN lower(coalesce(metadata->>'source', source_name, '')) LIKE 'who-%'
                        OR lower(coalesce(source_url, metadata->>'original_url', '')) LIKE '%who.int%'
                        THEN 'official:who:' || split_part(coalesce(metadata->>'source', source_name, document_no), '.', 1)
                    WHEN lower(coalesce(metadata->>'source', source_name, '')) LIKE 'nhc-%'
                        OR lower(coalesce(source_url, metadata->>'original_url', '')) LIKE '%gov.cn%'
                        THEN 'official:nhc:' || split_part(coalesce(metadata->>'source', source_name, document_no), '.', 1)
                    ELSE 'local:' || CASE
                        WHEN coalesce(metadata->>'source', '') <> '' THEN metadata->>'source'
                        WHEN coalesce(source_name, '') ~ '\\.[A-Za-z0-9]+$' THEN source_name
                        ELSE document_no || '.md'
                    END
                END
                WHERE coalesce(source_key, '') = ''
                """,
                """
                UPDATE documents
                SET sync_status = coalesce(nullif(sync_status, ''), 'active'),
                    is_active = coalesce(is_active, TRUE),
                    last_synced_at = coalesce(last_synced_at, updated_at, created_at, NOW())
                """,
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_source_key
                ON documents(source_key)
                WHERE source_key IS NOT NULL
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_documents_is_active
                ON documents(is_active)
                """,
                """
                ALTER TABLE import_task_logs
                ADD COLUMN IF NOT EXISTS updated INTEGER NOT NULL DEFAULT 0
                """,
                """
                ALTER TABLE import_task_logs
                ADD COLUMN IF NOT EXISTS deactivated INTEGER NOT NULL DEFAULT 0
                """,
                """
                ALTER TABLE import_task_logs
                ADD COLUMN IF NOT EXISTS unchanged INTEGER NOT NULL DEFAULT 0
                """,
                """
                ALTER TABLE import_task_logs
                ADD COLUMN IF NOT EXISTS trigger_type VARCHAR(32) NOT NULL DEFAULT 'manual'
                """,
                """
                ALTER TABLE import_task_logs
                ADD COLUMN IF NOT EXISTS scope VARCHAR(128) NOT NULL DEFAULT ''
                """,
            ],
        ),
        (
            "007_request_trace_ids",
            "Add request-level trace ids to route and retrieval logs.",
            [
                """
                ALTER TABLE route_logs
                ADD COLUMN IF NOT EXISTS request_id VARCHAR(64)
                """,
                """
                ALTER TABLE retrieval_logs
                ADD COLUMN IF NOT EXISTS request_id VARCHAR(64)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_route_logs_request_id
                ON route_logs(request_id)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_retrieval_logs_request_id
                ON retrieval_logs(request_id)
                """,
            ],
        ),
        (
            "008_appointment_demo_seed",
            "Seed demo departments, doctors, and future schedules for local booking flows.",
            [
                _DEMO_APPOINTMENT_SEED_SQL,
            ],
        ),
        (
            "009_chat_session_ownership",
            "Track chat session owners for API authorization.",
            [
                """
                ALTER TABLE chat_sessions
                ADD COLUMN IF NOT EXISTS owner_user_id VARCHAR(128)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_owner_user_id
                ON chat_sessions(owner_user_id)
                """,
            ],
        ),
        (
            "010_user_memories",
            "User-level long-term memories with pgvector similarity and importance scoring.",
            [
                """
                CREATE TABLE IF NOT EXISTS user_memories (
                    id              BIGSERIAL PRIMARY KEY,
                    user_id         VARCHAR(128) NOT NULL,
                    memory_type     VARCHAR(32) NOT NULL,
                    content         TEXT NOT NULL,
                    importance      SMALLINT NOT NULL DEFAULT 5,
                    embedding       VECTOR(1024),
                    source_thread_id VARCHAR(128),
                    merged_from     JSONB NOT NULL DEFAULT '[]'::jsonb,
                    access_count    INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at TIMESTAMP,
                    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_user_memories_user_id
                ON user_memories(user_id)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_user_memories_user_type
                ON user_memories(user_id, memory_type)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_user_memories_user_importance
                ON user_memories(user_id, importance DESC)
                """,
                f"""
                CREATE INDEX IF NOT EXISTS idx_user_memories_embedding_cosine
                ON user_memories
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {config.VECTOR_INDEX_LISTS})
                """,
            ],
        ),
        (
            "011_episodic_memory",
            "Episodic memory layer — full conversation timeline with vector search.",
            [
                """
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id              BIGSERIAL PRIMARY KEY,
                    user_id         VARCHAR(128) NOT NULL,
                    thread_id       VARCHAR(128) NOT NULL,
                    turn_index      INTEGER NOT NULL,
                    user_message    TEXT NOT NULL DEFAULT '',
                    assistant_message TEXT NOT NULL DEFAULT '',
                    embedding       VECTOR(1024),
                    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_episodic_memories_user_id
                ON episodic_memories(user_id)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_episodic_memories_user_time
                ON episodic_memories(user_id, created_at DESC)
                """,
                f"""
                CREATE INDEX IF NOT EXISTS idx_episodic_memories_embedding_cosine
                ON episodic_memories
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {config.VECTOR_INDEX_LISTS})
                """,
            ],
        ),
        (
            "012_reflection_memory",
            "Reflection memory layer — LLM-synthesized higher-order abstractions.",
            [
                """
                CREATE TABLE IF NOT EXISTS reflection_memories (
                    id              BIGSERIAL PRIMARY KEY,
                    user_id         VARCHAR(128) NOT NULL,
                    content         TEXT NOT NULL,
                    source_memory_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                    embedding       VECTOR(1024),
                    importance      SMALLINT NOT NULL DEFAULT 7,
                    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_reflection_memories_user_id
                ON reflection_memories(user_id)
                """,
                f"""
                CREATE INDEX IF NOT EXISTS idx_reflection_memories_embedding_cosine
                ON reflection_memories
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {config.VECTOR_INDEX_LISTS})
                """,
            ],
        ),
        (
            "013_users_table",
            "User accounts table for registration/login authentication.",
            [
                """
                CREATE TABLE IF NOT EXISTS users (
                    id              BIGSERIAL PRIMARY KEY,
                    username        VARCHAR(64) NOT NULL UNIQUE,
                    display_name    VARCHAR(128) NOT NULL DEFAULT '',
                    password_hash   VARCHAR(256) NOT NULL,
                    role            VARCHAR(32) NOT NULL DEFAULT 'user',
                    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
                    updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_users_username
                ON users(username)
                """,
            ],
        ),
    ]

    def __init__(self, conninfo: str):
        self._conninfo = conninfo
        self._base_schema_path = _SQL_DIR / "init_schema.sql"
        self._lock = threading.Lock()
        self._applied = False

    def _connect(self):
        return psycopg.connect(self._conninfo)

    def apply_migrations(self):
        with self._lock:
            if self._applied:
                return
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
                    cur.execute(self._base_schema_path.read_text(encoding="utf-8"))
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS schema_migrations (
                            version         VARCHAR(64) PRIMARY KEY,
                            description     TEXT NOT NULL,
                            applied_at      TIMESTAMP NOT NULL DEFAULT NOW()
                        )
                        """,
                    )
                    for version, description, statements in self._MIGRATIONS:
                        cur.execute("SELECT 1 FROM schema_migrations WHERE version = %s", (version,))
                        if cur.fetchone():
                            continue
                        for statement in statements:
                            cur.execute(statement)
                        cur.execute(
                            """
                            INSERT INTO schema_migrations (version, description)
                            VALUES (%s, %s)
                            ON CONFLICT (version) DO NOTHING
                            """,
                            (version, description),
                        )
                conn.commit()
            self._applied = True

    def inspect_schema(self):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT extname
                    FROM pg_extension
                    WHERE extname IN ('vector', 'pg_trgm')
                    ORDER BY extname
                    """
                )
                extensions = {row[0] for row in cur.fetchall()}

                cur.execute(
                    """
                    SELECT indexname
                    FROM pg_indexes
                    WHERE schemaname = current_schema()
                      AND indexname IN (
                          'uq_chat_session_summaries_thread_type',
                          'idx_child_chunks_embedding_cosine',
                          'idx_appointments_patient_status_date',
                          'idx_chat_sessions_patient_id',
                          'idx_chat_sessions_owner_user_id',
                          'idx_documents_source_name',
                          'uq_documents_source_key',
                          'idx_documents_is_active',
                          'idx_import_task_logs_created_at',
                          'idx_route_logs_created_at',
                          'idx_route_logs_thread_id',
                          'idx_route_logs_request_id',
                          'idx_retrieval_logs_request_id',
                          'idx_appointment_skill_logs_created_at',
                          'idx_appointment_skill_logs_thread_id',
                          'idx_user_memories_user_id',
                          'idx_user_memories_user_type',
                          'idx_user_memories_user_importance',
                          'idx_user_memories_embedding_cosine',
                          'idx_episodic_memories_user_id',
                          'idx_episodic_memories_user_time',
                          'idx_episodic_memories_embedding_cosine',
                          'idx_reflection_memories_user_id',
                          'idx_reflection_memories_embedding_cosine'
                      )
                    """
                )
                indexes = {row[0] for row in cur.fetchall()}

                cur.execute("SELECT version FROM schema_migrations ORDER BY version")
                versions = [row[0] for row in cur.fetchall()]

        return {
            "extensions": sorted(extensions),
            "indexes": sorted(indexes),
            "versions": versions,
        }
