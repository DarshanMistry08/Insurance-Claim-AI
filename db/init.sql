-- ============================================================
-- Insurance Claims AI Pipeline — Database Initialisation
-- Runs once on first `docker-compose up`
-- ============================================================

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- -----------------------------------------------------------
-- 1. Documents — one row per source document (image / PDF)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    document_id   SERIAL PRIMARY KEY,
    source_dataset TEXT        NOT NULL,   -- 'funsd', 'docvqa', 'synthetic', etc.
    file_path      TEXT        NOT NULL,   -- relative path inside data/
    doc_type       TEXT        NOT NULL DEFAULT 'claim_form',  -- claim_form, invoice, photo
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata       JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX idx_documents_source ON documents (source_dataset);
CREATE INDEX idx_documents_type   ON documents (doc_type);

-- -----------------------------------------------------------
-- 2. Chunks — embedded text segments from documents
--    Using 384-dim to match all-MiniLM-L6-v2
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    SERIAL PRIMARY KEY,
    document_id INTEGER     NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index INTEGER     NOT NULL DEFAULT 0,       -- ordering within the doc
    chunk_text  TEXT        NOT NULL,
    embedding   vector(384),                           -- nullable until embeddings computed
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX idx_chunks_document ON chunks (document_id);

-- IVFFlat index for approximate nearest-neighbor search
-- We start with lists=10 (suitable for <1000 rows); increase later.
-- This index is created as a placeholder — pgvector requires data before
-- building IVFFlat, so we use a partial workaround: create with HNSW instead.
CREATE INDEX idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- -----------------------------------------------------------
-- 3. Golden Labels — ground-truth annotations for evaluation
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS golden_labels (
    label_id          SERIAL PRIMARY KEY,
    document_id       INTEGER NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    field_name        TEXT    NOT NULL,   -- must match SPEC.md field names
    ground_truth_value TEXT,              -- null means "field not present in doc"
    confidence_note   TEXT,               -- annotator notes (e.g. "partially illegible")
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_golden_labels_document ON golden_labels (document_id);
CREATE INDEX idx_golden_labels_field    ON golden_labels (field_name);

-- Prevent duplicate labels for the same doc+field
CREATE UNIQUE INDEX idx_golden_labels_unique ON golden_labels (document_id, field_name);

-- -----------------------------------------------------------
-- 4. Processed Claims — output of the extraction pipeline
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS processed_claims (
    claim_id        SERIAL PRIMARY KEY,
    document_id     INTEGER NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    extracted_fields JSONB   NOT NULL DEFAULT '{}'::jsonb,  -- key-value of extracted fields
    flags           JSONB   NOT NULL DEFAULT '[]'::jsonb,   -- array of triggered flag objects
    llm_reasoning   TEXT,                                    -- raw LLM output for audit
    processing_time_ms INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_processed_claims_document ON processed_claims (document_id);

-- -----------------------------------------------------------
-- 5. Claims History — past claims embeddings for duplicate detection
--    Embedding represents key claim fields (claimant+incident+amount)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS claims_history (
    history_id      SERIAL PRIMARY KEY,
    claim_ref       TEXT        NOT NULL,       -- doc_id or external claim ID
    claimant_name   TEXT,
    incident_type   TEXT,
    amount_claimed  TEXT,
    embedding       vector(384),                -- all-MiniLM-L6-v2 embedding of concatenated fields
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_claims_history_ref ON claims_history (claim_ref);
CREATE INDEX idx_claims_history_embedding ON claims_history
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- -----------------------------------------------------------
-- 6. Policy Clauses — structured metadata per embedded chunk
--    Links a chunk to a human-readable clause ID for citations
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS policy_clauses (
    clause_id       TEXT        PRIMARY KEY,    -- e.g. "WATER_DAMAGE_B"
    chunk_id        INTEGER     REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    policy_doc      TEXT        NOT NULL,       -- e.g. "homeowners_standard"
    coverage_type   TEXT        NOT NULL,       -- e.g. "water" | "fire" | "theft"
    coverage_limit  NUMERIC,                    -- dollar limit (NULL if no fixed limit)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -----------------------------------------------------------
-- Quick sanity check
-- -----------------------------------------------------------
DO $$
BEGIN
    RAISE NOTICE '✅ Claims database initialised successfully.';
    RAISE NOTICE '   Tables: documents, chunks, golden_labels, processed_claims,';
    RAISE NOTICE '           claims_history, policy_clauses';
    RAISE NOTICE '   Extension: vector (pgvector)';
END
$$;
