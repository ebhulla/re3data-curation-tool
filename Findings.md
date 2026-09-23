# re3data Metadata Quality Investigation — Findings Report

**Author:** Ekam
**Date:** September 23, 2026
**Project:** AI-Enhanced Metadata Curation for re3data.org (under Prof. Witt)
**Research Question:** R1 — Normalization

---

## 1. Objective

Before choosing an AI approach for metadata normalization (rule-based,
fine-tuned classification, or retrieval-augmented generation), we needed to
answer a prior question: **how messy is the metadata, actually, and in what
way?** This report documents a first quantitative pass at that question,
using the `institution_name` and `description` fields as test cases, and
draws out what it implies for the R1 approach decision.

This is a first pass, not a final analysis — see Section 5 for what's still
open.

---

## 2. Methodology

- **Corpus:** All 3,524 repositories ingested from the re3data API into the
  `re3data_toolkit` PostgreSQL database (zero ingest failures).
- **Scope of this pass:** `repositories.description` (completeness check)
  and `institutions.institution_name` (consistency/duplication check). Other
  fields (subjects, contacts) have not yet been analyzed.
- **Method:** Direct SQL queries — completeness counts, string normalization
  comparisons (case/whitespace), and substring-containment joins as a cheap
  proxy for "these two names might refer to the same entity." No fuzzy
  matching (edit distance), embeddings, or manual review were used at this
  stage — this is meant to be a baseline before applying anything more
  sophisticated.
- **Full query set:** `quantify.sql` in this repo. Every query below is
  numbered to match that file.

---

## 3. Findings

### 3.1 Completeness (Query 1)

| total_repos | has_description | missing_description | avg_desc_length |
|---|---|---|---|
| 3,524 | 3,524 | 0 | 512.7 chars |

**Result:** 100% of repositories have a non-null description.

**Interpretation:** Completeness is not a problem in this dataset. This
matters because it rules out an entire class of "messiness" — we are not
dealing with missing data, so any normalization effort should focus on
consistency and correctness of existing text, not gap-filling.

### 3.2 Institution name variance — trivial normalization (Query 2)

| raw distinct strings | distinct after lowercase+trim |
|---|---|
| 6,142 | 6,094 |

**Result:** Only 48 of 6,142 distinct strings (0.8%) are resolved by basic
case/whitespace normalization.

**Interpretation:** If institution-name duplication exists at meaningful
scale, it is not explained by trivial formatting differences. A rule-based
normalizer limited to case/whitespace handling would have almost no effect.
This is evidence *against* relying on simple string rules alone, but it does
not yet tell us how much real duplication exists — that required further
digging (3.3–3.5).

### 3.3 Manual sample review (Query 3)

**Method:** Reviewed 30 alphabetically-sorted singleton institution names by
eye, looking for near-duplicate naming patterns (e.g. "Univ." vs
"University").

**Result:** No obvious variants found in this sample.

**Interpretation — important limitation:** This was a negative result, but
likely a false negative due to method, not because duplicates don't exist.
Alphabetical sorting separates name variants that don't share a common
prefix (e.g. "MIT" sorts nowhere near "Massachusetts Institute of
Technology"). This taught us that **untargeted eyeballing is not a reliable
detection method**, and motivated the targeted search in 3.4. Flagging this
explicitly because it's a methodology lesson worth stating out loud, not
hiding.

### 3.4 Targeted validation case (Query 4)

**Method:** Searched directly for a specific institution likely to have
multiple naming conventions ("MIT" / "Massachusetts Institute of
Technology") to test whether duplication exists when *not* relying on
alphabetical adjacency.

**Result:** Found `Canada's Michael Smith Genome Sciences Centre` (a
different institution, surfaced via a related search) represented as 4
distinct strings across 11 repository records:

| String | Count |
|---|---|
| `BC Cancer Agency, Canada's Michael Smith Genome Sciences Centre` | 1 |
| `BC Cancer Agency, Research Centre, Canada's Michael Smith Genome Sciences Centre` | 2 |
| `BC Cancer Research Centre, Canada's Michael Smith Genome Sciences Centre` | 1 |
| `Canada's Michael Smith Genome Sciences Centre` | 7 |

**Interpretation:** This is the first confirmed, concrete instance of real
duplication in the dataset. None of these four strings are case/whitespace
variants of each other — confirming that the 0.8% figure in 3.2 understates
the real problem, and that duplication in this dataset takes the form of
substantively different phrasings, not formatting noise.

### 3.5 Substring-containment pattern (Query 5)

**Method:** Searched for institution-name pairs where one string is fully
contained within another, as a proxy for duplicate or hierarchical
relationships.

**Result:** Surfaced two distinct, easily-confused patterns:

**Pattern A — genuine duplicates (same entity, should be merged):**
- `National Science Foundation` / `National Science foundation`
- `Japan Agency for Marine-Earth Science and Technology` /
  `Japan Agency for Marine-earth Science and TEChnology`

**Pattern B — parent/child hierarchy (different entities, must NOT be merged):**
- `German Aerospace Center` / `German Aerospace Center, Institute of
  Atmospheric Physics`
- `Universität Bonn` / `Rheinische Friedrich-Wilhelms-Universität Bonn`

**Interpretation — this is the central finding of this report.** A
similarity-based approach (fuzzy string matching or embedding similarity
alone) cannot reliably tell these two patterns apart — both involve one
string containing or closely resembling another. Pattern A should collapse
to one canonical name; Pattern B must stay separate, because a research
institute is not the same entity as its parent university. Merging Pattern
B incorrectly would **destroy real information**, not just fail to improve
it. Distinguishing the two requires contextual/semantic judgment, which is
the core argument for why this problem needs a model with real language
understanding rather than string-similarity heuristics alone.

### 3.6 Scale estimate (Queries 6–7)

**Method note — an error was caught and corrected here, documented for
transparency:** The first version of this query (Query 6) joined directly
against the `institutions` table, which stores one row per repository-
institution link rather than one row per unique institution name. This
inflated the result to 38,817, because institutions appearing on many repos
were counted multiple times per pair. Query 7 corrects this by deduplicating
to distinct institution names before joining.

| Metric | Value |
|---|---|
| Distinct institution names | 6,142 |
| Name pairs showing substring-containment (corrected) | 2,432 |

**Interpretation:** Roughly 2,432 pairs out of 6,142 distinct names show
this pattern — a substantial fraction of the institution namespace. This
number is an **upper bound**, not a clean duplicate count, because it mixes
Pattern A (true duplicates) and Pattern B (legitimate hierarchies) from
3.5. We do not yet know the split between the two. Establishing that split
is the immediate next step (Section 5).

---

## 4. Summary of Evidence

| Question | Answer | Confidence |
|---|---|---|
| Is missing data a problem? | No — 100% completeness | High (full corpus checked) |
| Does simple rule-based normalization solve institution-name messiness? | No — only 0.8% of variants affected | High (full corpus checked) |
| Does real duplication exist? | Yes — confirmed concrete case (3.4) | High (confirmed instance, not yet scaled) |
| Can duplication be reliably auto-detected by string similarity alone? | No — conflates true duplicates with legitimate hierarchies (3.5) | High (concrete counterexamples found) |
| How much duplication exists at full-corpus scale? | Upper bound: 2,432 candidate pairs; true rate unknown | **Low — not yet validated** |
| What fraction of those 2,432 pairs are real duplicates vs. hierarchies? | Unknown | **Not yet measured** |

---

## 5. What's Missing / Open Questions

Being explicit about this is the point of this section — this is what
should be presented as "not yet done" rather than glossed over:

1. **No ground-truth labels yet.** All findings above are pattern-level
   observations, not validated against a labeled sample. We do not have a
   real duplicate rate — only an upper-bound candidate count.
2. **No sampling beyond illustrative examples.** Sections 3.4 and 3.5 show
   real, confirmed cases, but they were found by targeted/manual search, not
   random sampling — so we can't yet claim they're representative of the
   whole 2,432-pair set.
3. **Only one field analyzed in depth.** `subjects` and `description`
   consistency (beyond the completeness check) haven't been examined yet.
4. **No baseline comparison run.** We haven't yet tested how a fuzzy-
   matching method (e.g. Levenshtein/Jaro-Winkler) performs on this same
   data, which would be a fairer comparison point than pure substring
   matching before concluding "needs a model."

## 6. Recommended Next Steps

1. **Manually label a random sample** of ~30 pairs from the 2,432
   candidates as `duplicate` / `hierarchy` / `false positive`, to get a real
   estimated duplicate rate (not just an upper bound) and produce the first
   labeled dataset for later classifier work.
2. **Run a fuzzy-matching baseline** (edit distance) on the same sample, to
   have an honest comparison point against a "smarter" model-based approach
   before claiming the latter is necessary.
3. **Repeat completeness/consistency checks on `subjects`**, since only
   `description` and `institution_name` have been examined so far.
4. **Bring this report to Monday's team meeting** as the evidence base for
   the R1 approach decision, with Section 4's table as the one-slide summary
   if needed.

---

## Appendix: Query Reference

All queries are in `quantify.sql`, numbered to match the section headers
above. Each query has an inline comment explaining its purpose.
