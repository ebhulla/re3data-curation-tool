-- =============================================================================
-- re3data_toolkit — personal curation tool schema
-- Four tables: repositories (core) + subjects, contacts, institutions
-- (one-to-many children, matching re3data's repeating XML elements)
-- =============================================================================

CREATE TABLE repositories (
    id            TEXT PRIMARY KEY,   -- re3data's own id, e.g. r3d100010299 — natural key, not SERIAL
    name          TEXT NOT NULL,
    url           TEXT,
    doi           TEXT,
    description   TEXT,
    typology      TEXT
);

CREATE TABLE subjects (
    id            SERIAL PRIMARY KEY,
    repo_id       TEXT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    subject_name  TEXT NOT NULL
);

CREATE TABLE contacts (
    id            SERIAL PRIMARY KEY,
    repo_id       TEXT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    contact_info  TEXT,
    contact_type  TEXT
);

CREATE TABLE institutions (
    id                   SERIAL PRIMARY KEY,
    repo_id              TEXT NOT NULL REFERENCES repositories(id) ON DELETE CASCADE,
    institution_name     TEXT,
    institution_country  TEXT,
    institution_type     TEXT,
    institution_url      TEXT
);

-- helpful for looking up all children of a given repo quickly
CREATE INDEX idx_subjects_repo_id ON subjects(repo_id);
CREATE INDEX idx_contacts_repo_id ON contacts(repo_id);
CREATE INDEX idx_institutions_repo_id ON institutions(repo_id);