-- =============================================================================
-- re3data metadata curation database — hybrid schema
-- Prof. Witt project (R1: normalization / R2: classification / R3: HIL workflow)
--
-- Design: normalize the fields R1/R2 actually need to work with, keep the rest
-- of each record in raw_json for provenance and later re-parsing. This lets
-- Phase 1 quantify "sorted vs unsorted" per repo without modeling all ~25
-- repeatable elements of the re3data v4.0 schema up front.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- fuzzy/fast text search on names & descriptions

-- ---------------------------------------------------------------------------
-- Core table: one row per repository record
-- ---------------------------------------------------------------------------
CREATE TABLE repositories (
    id                  SERIAL PRIMARY KEY,
    re3data_id          TEXT NOT NULL UNIQUE,          -- e.g. r3d100000001
    doi                 TEXT,

    repository_name     TEXT NOT NULL,
    repository_name_lang TEXT DEFAULT 'eng',

    repository_url      TEXT,

    -- primary (usually English) description; other-language descriptions live in raw_json
    description         TEXT,
    description_lang    TEXT DEFAULT 'eng',

    provider_type       TEXT[],                        -- e.g. {dataProvider,serviceProvider}

    entry_date          DATE,
    last_update         DATE,

    -- full parsed record (from XML->JSON or the API's JSON response), source of truth
    -- for anything not modeled above and for later re-parsing without re-fetching
    raw_json            JSONB NOT NULL,
    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_repositories_name_trgm ON repositories USING gin (repository_name gin_trgm_ops);
CREATE INDEX idx_repositories_raw_json ON repositories USING gin (raw_json jsonb_path_ops);

-- ---------------------------------------------------------------------------
-- repository_types — <r3d:type> is multiple (disciplinary, institutional, ...)
-- ---------------------------------------------------------------------------
CREATE TABLE repository_types (
    id              SERIAL PRIMARY KEY,
    repository_id   INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    type            TEXT NOT NULL,
    UNIQUE (repository_id, type)
);

-- ---------------------------------------------------------------------------
-- repository_subjects — R2 classification target; DFG subject classification
-- ---------------------------------------------------------------------------
CREATE TABLE repository_subjects (
    id              SERIAL PRIMARY KEY,
    repository_id   INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    subject_scheme  TEXT NOT NULL DEFAULT 'DFG',
    subject_id      TEXT,
    subject_name    TEXT NOT NULL
);

CREATE INDEX idx_repository_subjects_repo ON repository_subjects(repository_id);
CREATE INDEX idx_repository_subjects_name ON repository_subjects(subject_name);

-- ---------------------------------------------------------------------------
-- repository_content_types — COAR resource-type vocabulary
-- ---------------------------------------------------------------------------
CREATE TABLE repository_content_types (
    id                  SERIAL PRIMARY KEY,
    repository_id       INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    content_type_scheme TEXT NOT NULL DEFAULT 'COAR',
    content_type_id     TEXT,
    content_type_name   TEXT NOT NULL
);

CREATE INDEX idx_repository_content_types_repo ON repository_content_types(repository_id);

-- ---------------------------------------------------------------------------
-- repository_keywords — free-text, the messiest field; useful for R1 normalization
-- ---------------------------------------------------------------------------
CREATE TABLE repository_keywords (
    id              SERIAL PRIMARY KEY,
    repository_id   INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    keyword         TEXT NOT NULL
);

CREATE INDEX idx_repository_keywords_repo ON repository_keywords(repository_id);
CREATE INDEX idx_repository_keywords_trgm ON repository_keywords USING gin (keyword gin_trgm_ops);

-- ---------------------------------------------------------------------------
-- field_curation_status — tracks per-field data quality / curation state.
-- This is what actually answers the Phase 1 question ("what fraction of
-- repos have sorted vs. unsorted fields") and doubles as the changelog the
-- skill's Process section asks for (source + note per change).
-- ---------------------------------------------------------------------------
CREATE TABLE field_curation_status (
    id              SERIAL PRIMARY KEY,
    repository_id   INTEGER NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    field_name      TEXT NOT NULL,                     -- 'description', 'subject', 'type', 'keyword', ...
    status          TEXT NOT NULL CHECK (
                        status IN ('missing', 'messy', 'normalized', 'ai_suggested', 'human_verified')
                    ),
    note            TEXT,                              -- what changed / why flagged
    source          TEXT,                               -- e.g. 'ai_normalizer_v1', 'manual', 'import'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_field_curation_repo ON field_curation_status(repository_id);
CREATE INDEX idx_field_curation_status ON field_curation_status(status);

-- ---------------------------------------------------------------------------
-- keep updated_at fresh
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_repositories_updated_at
BEFORE UPDATE ON repositories
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------------
-- convenience view: quick "sorted vs unsorted" quantification for the poster
-- ---------------------------------------------------------------------------
CREATE VIEW repository_field_completeness AS
SELECT
    r.id,
    r.re3data_id,
    r.repository_name,
    (r.description IS NOT NULL AND length(trim(r.description)) > 0)   AS has_description,
    EXISTS (SELECT 1 FROM repository_subjects s WHERE s.repository_id = r.id)       AS has_subject,
    EXISTS (SELECT 1 FROM repository_content_types c WHERE c.repository_id = r.id)  AS has_content_type,
    EXISTS (SELECT 1 FROM repository_keywords k WHERE k.repository_id = r.id)       AS has_keywords,
    array_length(r.provider_type, 1) > 0                                            AS has_provider_type
FROM repositories r;
