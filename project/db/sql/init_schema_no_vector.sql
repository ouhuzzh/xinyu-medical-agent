CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS patients (
    id              BIGSERIAL PRIMARY KEY,
    patient_no      VARCHAR(64) UNIQUE,
    name            VARCHAR(64),
    gender          VARCHAR(16),
    birth_date      DATE,
    phone           VARCHAR(32),
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id              BIGSERIAL PRIMARY KEY,
    thread_id       VARCHAR(128) NOT NULL UNIQUE,
    owner_user_id   VARCHAR(128),
    patient_id      BIGINT REFERENCES patients(id),
    status          VARCHAR(32) NOT NULL DEFAULT 'active',
    current_intent  VARCHAR(64),
    risk_level      VARCHAR(32),
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chat_session_summaries (
    id                  BIGSERIAL PRIMARY KEY,
    thread_id           VARCHAR(128) NOT NULL REFERENCES chat_sessions(thread_id) ON DELETE CASCADE,
    summary_type        VARCHAR(32) NOT NULL DEFAULT 'long_term',
    summary_content     TEXT NOT NULL,
    last_message_index  INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_session_summaries_thread_id
ON chat_session_summaries(thread_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_chat_session_summaries_thread_type
ON chat_session_summaries(thread_id, summary_type);

CREATE TABLE IF NOT EXISTS departments (
    id              BIGSERIAL PRIMARY KEY,
    code            VARCHAR(64) NOT NULL UNIQUE,
    name            VARCHAR(128) NOT NULL,
    description     TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS doctors (
    id              BIGSERIAL PRIMARY KEY,
    department_id   BIGINT NOT NULL REFERENCES departments(id),
    name            VARCHAR(128) NOT NULL,
    title           VARCHAR(128),
    profile         TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS doctor_schedules (
    id              BIGSERIAL PRIMARY KEY,
    doctor_id       BIGINT NOT NULL REFERENCES doctors(id),
    department_id   BIGINT NOT NULL REFERENCES departments(id),
    schedule_date   DATE NOT NULL,
    time_slot       VARCHAR(32) NOT NULL,
    quota_total     INTEGER NOT NULL DEFAULT 0,
    quota_available INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_doctor_schedules_dept_date
ON doctor_schedules(department_id, schedule_date);

CREATE TABLE IF NOT EXISTS appointments (
    id                  BIGSERIAL PRIMARY KEY,
    appointment_no      VARCHAR(64) NOT NULL UNIQUE,
    patient_id          BIGINT NOT NULL REFERENCES patients(id),
    doctor_id           BIGINT REFERENCES doctors(id),
    department_id       BIGINT NOT NULL REFERENCES departments(id),
    schedule_id         BIGINT REFERENCES doctor_schedules(id),
    appointment_date    DATE NOT NULL,
    time_slot           VARCHAR(32) NOT NULL,
    status              VARCHAR(32) NOT NULL DEFAULT 'booked',
    created_by          VARCHAR(32) NOT NULL DEFAULT 'ai_agent',
    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_appointments_patient_id
ON appointments(patient_id);

CREATE INDEX IF NOT EXISTS idx_appointments_status
ON appointments(status);

CREATE TABLE IF NOT EXISTS appointment_logs (
    id                  BIGSERIAL PRIMARY KEY,
    appointment_id      BIGINT REFERENCES appointments(id) ON DELETE CASCADE,
    thread_id           VARCHAR(128),
    action              VARCHAR(32) NOT NULL,
    operator_type       VARCHAR(32) NOT NULL DEFAULT 'ai_agent',
    request_payload     JSONB,
    response_payload    JSONB,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS documents (
    id                  BIGSERIAL PRIMARY KEY,
    document_no         VARCHAR(64) NOT NULL UNIQUE,
    title               VARCHAR(512) NOT NULL,
    source_name         VARCHAR(512),
    file_type           VARCHAR(32),
    doc_type            VARCHAR(64),
    department          VARCHAR(128),
    authority_level     VARCHAR(32),
    source_url          TEXT,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_doc_type
ON documents(doc_type);

CREATE INDEX IF NOT EXISTS idx_documents_department
ON documents(department);

CREATE TABLE IF NOT EXISTS parent_chunks (
    id                  BIGSERIAL PRIMARY KEY,
    parent_id           VARCHAR(128) NOT NULL UNIQUE,
    document_id         BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    title               VARCHAR(512),
    department          VARCHAR(128),
    content             TEXT NOT NULL,
    token_count         INTEGER,
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_parent_chunks_document_id
ON parent_chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_parent_chunks_department
ON parent_chunks(department);

CREATE TABLE IF NOT EXISTS child_chunks (
    id                  BIGSERIAL PRIMARY KEY,
    chunk_id            VARCHAR(128) NOT NULL UNIQUE,
    parent_id           VARCHAR(128) NOT NULL REFERENCES parent_chunks(parent_id) ON DELETE CASCADE,
    document_id         BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index         INTEGER NOT NULL,
    content             TEXT NOT NULL,
    token_count         INTEGER,
    department          VARCHAR(128),
    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding_json      JSONB,
    tsv                 tsvector,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_child_chunks_parent_id
ON child_chunks(parent_id);

CREATE INDEX IF NOT EXISTS idx_child_chunks_document_id
ON child_chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_child_chunks_department
ON child_chunks(department);

CREATE INDEX IF NOT EXISTS idx_child_chunks_tsv
ON child_chunks
USING GIN (tsv);

CREATE TABLE IF NOT EXISTS retrieval_logs (
    id                  BIGSERIAL PRIMARY KEY,
    thread_id           VARCHAR(128),
    query_text          TEXT NOT NULL,
    rewritten_query     TEXT,
    retrieval_mode      VARCHAR(32),
    top_k               INTEGER,
    result_count        INTEGER,
    selected_parent_ids JSONB,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_retrieval_logs_thread_id
ON retrieval_logs(thread_id);

CREATE OR REPLACE FUNCTION child_chunks_tsv_trigger_fn()
RETURNS trigger AS $$
BEGIN
  NEW.tsv := to_tsvector('simple', COALESCE(NEW.content, ''));
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_child_chunks_tsv_update ON child_chunks;

CREATE TRIGGER trg_child_chunks_tsv_update
BEFORE INSERT OR UPDATE OF content
ON child_chunks
FOR EACH ROW
EXECUTE FUNCTION child_chunks_tsv_trigger_fn();
