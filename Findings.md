## Metadata Messiness Findings (2026-09-23)

Investigation into how "messy" re3data's metadata actually is, run against the
full ingested corpus (3,524 repositories) in `re3data_toolkit`. Queries in
`quantify.sql`. Goal was to get quantified evidence for which AI approach R1
normalization actually needs, instead of guessing.

### 1. Description field completeness

| total_repos | has_description | missing_description | avg_desc_length |
|---|---|---|---|
| 3524 | 3524 | 0 | 512.7 |

**Inference:** Zero missing descriptions. Completeness is not the problem here
-- every repo has a description averaging ~513 characters. This rules out a
whole category of concern (nulls/gaps). Whatever "messiness" exists is about
*consistency*, not *completeness*.

### 2. Institution name variance (case/whitespace)

| raw_variants | normalized_variants |
|---|---|
| 6142 | 6094 |

**Inference:** Only 48 of 6,142 variants (~0.8%) collapse under basic
lowercase/trim normalization. Simple case/whitespace rules barely move the
needle -- meaning if real duplication exists, it isn't the trivial kind a
rule-based normalizer catches.

### 3. Singleton institution sample (manual eyeball)

Scanned ~30 singleton institution names alphabetically. No obvious
"University" vs "Univ." style variants surfaced in this particular sample --
these looked like genuinely distinct small organizations (foundations,
consortiums, projects).

**Inference:** Alphabetical sampling isn't a reliable way to spot duplicates,
since name variants of the same institution can sort far apart (e.g. "MIT"
vs "Massachusetts Institute of Technology"). Needed a targeted search instead.

### 4. Targeted case: Canada's Michael Smith Genome Sciences Centre

Found 4 distinct strings referring to the same real-world institution:
- `BC Cancer Agency, Canada's Michael Smith Genome Sciences Centre` (1)
- `BC Cancer Agency, Research Centre, Canada's Michael Smith Genome Sciences Centre` (2)
- `BC Cancer Research Centre, Canada's Michael Smith Genome Sciences Centre` (1)
- `Canada's Michael Smith Genome Sciences Centre` (7)

**Inference:** 11 records, same institution, 4 different strings -- none of
which are case/whitespace variants. This is the first concrete proof that
real duplication exists and that it requires semantic understanding, not
string rules, to catch.

### 5. Substring-containment sample

Surfaced two distinct patterns tangled together:

- **Genuine duplicates** (safe to merge): `National Science Foundation` vs
  `National Science foundation`; `Japan Agency for Marine-Earth Science and
  Technology` vs `...Marine-earth Science and TEChnology`
- **Parent/child hierarchy** (NOT duplicates -- must NOT be merged):
  `German Aerospace Center` vs `German Aerospace Center, Institute of
  Atmospheric Physics`; `Universität Bonn` vs `Rheinische
  Friedrich-Wilhelms-Universität Bonn`

**Inference:** This is the most important finding. A naive similarity-based
approach (fuzzy matching, embeddings alone) would risk incorrectly merging
parent/child pairs -- collapsing a specific research institute into its
parent university and destroying real information. Distinguishing "same
thing, different string" from "different thing, overlapping words" is a
semantic judgment call, not a string-matching problem. This is the strongest
evidence yet for needing a real model (fine-tuning/deeper understanding)
rather than rule-based or pure fuzzy-matching normalization.

### 6. Substring-containment pair count -- raw (inflated, superseded)

Initial count: 38,817 pairs. **This number is wrong** -- the `institutions`
table has one row per repo-institution link, not one row per unique name, so
repeated institutions inflated the join. Kept in the SQL file for the record,
not for reporting.

### 7. Substring-containment pair count -- corrected

| count |
|---|
| 2432 |

**Inference:** Deduplicated to distinct institution names first. Out of
6,142 distinct institution name strings, 2,432 pairs show a substring-
containment relationship. This is a substantial fraction -- a meaningful
share of institution records are entangled in either duplicate-variant or
parent/child relationships. This number mixes both patterns from finding #5,
so it's an upper bound on the "messiness" scale, not a clean duplicate count.

---

### Overall takeaway

- Missing data is not the issue (100% completeness on descriptions).
- Trivial rule-based normalization (case/whitespace) barely helps (~0.8% of
  variants).
- Real duplication exists and is substantial in scale (2,432 candidate
  pairs), but it's tangled with legitimate parent/child hierarchies that a
  naive similarity approach would incorrectly merge.
- **This is direct evidence that R1 normalization needs semantic judgment
  (fine-tuned model or LLM-assisted reasoning), not just string rules or
  basic fuzzy matching** -- the risk of silently destroying real institutional
  distinctions is real and demonstrated with concrete examples.

### Next step

Manually label a sample (~20-30) of the 2,432 substring-containment pairs as
duplicate / hierarchy / false-positive, to get a rough duplicate rate and
produce the first labeled dataset for later classifier work.
