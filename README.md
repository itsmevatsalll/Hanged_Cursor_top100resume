# Team Hanged Cursor — Top 100 Resume Ranking

This system ranks a pool of candidate resumes against a job description and
produces a `team_Hanged_Cursor.csv` submission file containing the top 100
candidates, their scores, and per-candidate reasoning.

The pipeline is split into **two stages**, per the challenge rules:

1. **Pre-computation** (`precompute.py`) — honeypot filtering, feature
   extraction, resume-text building, and embedding generation. This stage is
   *not* time-boxed and may take several minutes depending on pool size and
   hardware (it loads a sentence-transformer model and embeds every
   candidate).
2. **Ranking** (`run_pipeline.py`) — reads only the pre-computed artifacts
   and produces the submission CSV in well under 5 minutes, since it's pure
   numpy/pandas scoring over already-embedded vectors with no model
   inference in the loop.

---

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` pins `en_core_web_sm` (spaCy model) as a direct wheel URL,
so no separate `python -m spacy download` step is needed.

The first run of `precompute.py` will also download the
`BAAI/bge-small-en-v1.5` sentence-transformer model from Hugging Face
(used to embed both the job description and every candidate resume). Make
sure you have internet access the first time, or pre-warm the Hugging Face
cache — after that it's read from local cache.

### Required input files

| Path | Description |
|---|---|
| `data/candidates.jsonl` | Full candidate pool, one JSON object per line (see `sample_candidate.json` for the schema). Gitignored — too large for git, so it will **not** exist after a fresh clone. Place it here yourself, or pass `--candidates <path>` to `precompute.py`. |
| `data/job_description.docx` | The job description to rank candidates against. Already included in this repo. |

---

## 2. Stage 1 — Pre-computation (not time-boxed)

```bash
python precompute.py --candidates ./data/candidates.jsonl
```

This runs, in order:

1. `preprocessing/run_honeypot.py` — flags and separates honeypot/trap
   entries out of the raw candidate pool.
   → `artifacts/clean_candidates.jsonl`, `artifacts/flagged_honeypots.jsonl`,
   `artifacts/honeypot_summary.csv`
2. `preprocessing/feature_extractor.py` — extracts structured features
   (experience, skills, capabilities, recruiter/behavioral signals, etc.)
   from the cleaned candidates.
   → `artifacts/candidate_features.pkl`
3. `preprocessing/build_resume_text.py` — builds a flattened text
   representation of each resume for embedding.
   → `artifacts/candidate_texts.jsonl`
4. `embeddings/embed_job.py` — extracts text from the job description
   `.docx` and embeds it with `BAAI/bge-small-en-v1.5`.
   → `artifacts/jd_text.txt`, `artifacts/jd_embedding.npy`
5. `embeddings/embed_candidates.py` — embeds every candidate's resume text
   with the same model.
   → `artifacts/candidate_embeddings.npy`, `artifacts/candidate_ids.json`
6. `ranking/jd_analyzer.py` — builds a capability-weight/notice-period
   profile from the job description text (reads `artifacts/jd_text.txt`
   produced in step 4, never the raw `.docx` directly).
   → `artifacts/jd_profile.json`

**Optional flag:** for a fast sandbox/demo run instead of the full pool, use
`--sample-size`:

```bash
python precompute.py --candidates ./data/candidates.jsonl --sample-size 500
```

All artifacts are written to `artifacts/` (gitignored — regenerate them by
re-running this script; don't commit them).

---

## 3. Stage 2 — Ranking → submission CSV (must finish in < 5 minutes)

Once `artifacts/` is populated by the step above, produce the submission
CSV with a **single command**:

```bash
python run_pipeline.py
```

This only reads the artifacts already computed in Stage 1 — no embedding
model inference happens here, so it comfortably fits inside the 5-minute
window. It runs:

1. `ranking/rank.py` — combines semantic similarity, capability match,
   structured features, and behavioral signals into a `final_score` per
   candidate, and writes the top 500 to `artifacts/top500.json`.
2. `reranker/rerank.py` — re-scores those 500 with a deep, evidence-weighted
   capability score plus a corporate-pedigree/role-domain filter, takes the
   top 100, and writes:
   - `artifacts/top100.json` (full detail, JSON)
   - **`team_Hanged_Cursor.csv`** (the submission file, written to the
     project root)

### Validate the output

```bash
python validate_submission.py team_Hanged_Cursor.csv
```

Checks the header, row count (exactly 100 data rows), `candidate_id`
format, unique ranks 1–100, and score ordering/tie-breaking rules.

---

## 4. End-to-end (both stages back to back)

```bash
python precompute.py --candidates ./data/candidates.jsonl
python run_pipeline.py
python validate_submission.py team_Hanged_Cursor.csv
```

Only the second command (`run_pipeline.py`) needs to complete within the
5-minute window — `precompute.py` may run as long as it needs.

---

## Project layout

```
precompute.py            # Stage 1 orchestrator (honeypot -> features -> text -> embeddings -> JD profile)
run_pipeline.py           # Stage 2 orchestrator (rank -> rerank -> submission CSV)
validate_submission.py    # Sanity-checks the submission CSV against the rules

preprocessing/            # honeypot detection, feature extraction, resume-text building
embeddings/                # JD + candidate embedding (BAAI/bge-small-en-v1.5)
ranking/                    # rank.py (semantic + capability + structured + behavioral scoring), jd_analyzer.py
reranker/                    # rerank.py (deep evidence scoring, pedigree filter, reasoning generation, CSV writer)
utils/                         # shared schemas, inference rules, capability detection, text aggregation

data/job_description.docx     # job description (included)
data/candidates.jsonl         # candidate pool (NOT included — add your own; gitignored)

artifacts/                    # all intermediate/generated files (gitignored, regenerate via precompute.py)
team_Hanged_Cursor.csv        # final submission output (generated by run_pipeline.py)
```

## Tests

Unit tests for the honeypot detector, capability/text aggregator, and
inference rules:

```bash
pytest test_detector.py test_aggregator.py test_rules.py
```
