-- ============================================================
-- re3data Metadata Messiness Quantification
-- Date: 2026-09-23
-- Goal: Quantify how "messy" the re3data metadata actually is,
-- to determine which AI approach (rule-based, fine-tuning, RAG)
-- is actually needed for R1 normalization.
-- ============================================================


-- ------------------------------------------------------------
-- Query 1: Description field completeness
-- Aim: Check whether missing data is a real problem, and get a
-- baseline sense of description length across the corpus.
-- ------------------------------------------------------------
SELECT
    COUNT(*) AS total_repos,
    COUNT(description) AS has_description,
    COUNT(*) - COUNT(description) AS missing_description,
    AVG(LENGTH(description)) AS avg_desc_length
FROM repositories;


-- ------------------------------------------------------------
-- Query 2: Institution name variance (case/whitespace only)
-- Aim: See how much of the institution-name messiness is
-- explained by trivial casing/whitespace differences.
-- ------------------------------------------------------------
SELECT
    COUNT(DISTINCT institution_name) AS raw_variants,
    COUNT(DISTINCT LOWER(TRIM(institution_name))) AS normalized_variants
FROM institutions;


-- ------------------------------------------------------------
-- Query 3: Singleton institution names (sample)
-- Aim: Manually eyeball institution names that only appear once,
-- to look for near-duplicate naming patterns (e.g. "Univ." vs
-- "University").
-- ------------------------------------------------------------
SELECT institution_name, COUNT(*)
FROM institutions
GROUP BY institution_name
HAVING COUNT(*) = 1
ORDER BY institution_name
LIMIT 30;


-- ------------------------------------------------------------
-- Query 4: Targeted check on a known-ambiguous institution
-- Aim: Search for a specific institution likely to have naming
-- variants (MIT), to directly test the duplication hypothesis.
-- ------------------------------------------------------------
SELECT institution_name, COUNT(*)
FROM institutions
WHERE institution_name ILIKE '%massachusetts institute%'
   OR institution_name ILIKE '%mit%'
GROUP BY institution_name;


-- ------------------------------------------------------------
-- Query 5: Substring-containment pairs (sample)
-- Aim: Find institution name pairs where one string is fully
-- contained in another -- a proxy for duplicate/hierarchy
-- relationships that simple case/whitespace rules can't catch.
-- NOTE: run against raw `institutions` table -- this produces
-- inflated counts because of repo-institution row duplication
-- (see Query 7 for the corrected version).
-- ------------------------------------------------------------
SELECT a.institution_name, b.institution_name
FROM institutions a
JOIN institutions b
  ON a.institution_name != b.institution_name
  AND a.institution_name ILIKE '%' || b.institution_name || '%'
  AND LENGTH(b.institution_name) > 15
LIMIT 20;


-- ------------------------------------------------------------
-- Query 6: Substring-containment pair count (RAW -- inflated)
-- Aim: Get a scale estimate of how common the substring pattern
-- is. NOTE: this double(multi)-counts because `institutions` has
-- one row per repo-institution link, not one row per unique name.
-- Kept here for the record; superseded by Query 7.
-- ------------------------------------------------------------
SELECT COUNT(*)
FROM institutions a
JOIN institutions b
  ON a.institution_name != b.institution_name
  AND a.institution_name ILIKE '%' || b.institution_name || '%'
  AND LENGTH(b.institution_name) > 15;


-- ------------------------------------------------------------
-- Query 7: Substring-containment pair count (CORRECTED)
-- Aim: Same as Query 6, but deduplicated to distinct institution
-- names first, so each unique-name pair is counted once. This is
-- the real scale estimate.
-- ------------------------------------------------------------
WITH distinct_names AS (
  SELECT DISTINCT institution_name FROM institutions
)
SELECT COUNT(*)
FROM distinct_names a
JOIN distinct_names b
  ON a.institution_name != b.institution_name
  AND a.institution_name ILIKE '%' || b.institution_name || '%'
  AND LENGTH(b.institution_name) > 15;