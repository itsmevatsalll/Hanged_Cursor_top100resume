"""
precompute.py — Pre-computation orchestrator for Hanged Cursor's
Redrob AI / India Runs hackathon submission.

Per the challenge spec, only ranking + reranking + CSV generation must
run inside the 5-minute window. Everything upstream of that is allowed
to be pre-computed ahead of time. This script runs that upstream half.

Stage order and I/O, verified directly against the actual source of
each module (not guessed):

    1. preprocessing.run_honeypot       data/candidates.jsonl            -> artifacts/clean_candidates.jsonl
                                         (via preprocessing.honeypot.is_honeypot)
                                                                          (+ artifacts/flagged_honeypots.jsonl,
                                                                             artifacts/honeypot_summary.csv)
    2. preprocessing.feature_extractor  artifacts/clean_candidates.jsonl -> artifacts/candidate_features.pkl
    3. preprocessing.build_resume_text  artifacts/clean_candidates.jsonl -> artifacts/candidate_texts.jsonl
    4. embeddings.embed_job             data/job_description.docx       -> artifacts/jd_text.txt,
                                                                             artifacts/jd_embedding.npy
    5. embeddings.embed_candidates      artifacts/candidate_texts.jsonl -> artifacts/candidate_embeddings.npy,
                                                                             artifacts/candidate_ids.json
    6. ranking.jd_analyzer              artifacts/jd_text.txt           -> artifacts/jd_profile.json
                                         (produced by stage 4, not the raw .docx --
                                          this is a HARD dependency: jd_analyzer.py
                                          only reads jd_text.txt, never the .docx itself)

Stages 2 and 3 both depend only on stage 1 and are independent of each
other. Stage 4 is independent of 1-3 but must finish before stage 6.
rank.py (used by run_pipeline.py) reads exactly the five artifacts
produced above: candidate_embeddings.npy, candidate_ids.json,
candidate_features.pkl, jd_embedding.npy, jd_profile.json -- and reads
jd_profile.json at IMPORT time, so run_pipeline.py cannot even be
imported until this script has completed.

Note: preprocessing/run_honeypot.py creates artifacts/ itself at import
time (module-level ARTIFACTS_DIR.mkdir(exist_ok=True)), before any stage
here runs -- the explicit mkdir below is just a redundant safety net.

Usage:
    python precompute.py
    python precompute.py --candidates data/my_pool.jsonl
    python precompute.py --sample-size 500      # fast sandbox/demo run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_CANDIDATES = ROOT / "data" / "candidates.jsonl"
JD_DOCX = ROOT / "data" / "job_description.docx"


def _sample_candidates(src: Path, n: int) -> Path:
    """Write the first n lines of src to artifacts/_sample_<n>.jsonl, return its path."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    sampled = ARTIFACTS_DIR / f"_sample_{n}.jsonl"
    with open(src, "r", encoding="utf-8") as fin, open(sampled, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= n:
                break
            fout.write(line)
    print(f"  -> wrote {sampled} ({n} candidates)")
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Pre-compute all artifacts run_pipeline.py needs.")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help=f"Path to the candidate pool JSONL (default: {DEFAULT_CANDIDATES})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Only process the first N candidates (useful for a fast sandbox/demo run).",
    )
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    candidates_path = args.candidates or DEFAULT_CANDIDATES
    if not candidates_path.exists():
        sys.exit(
            f"Candidate file not found: {candidates_path}\n"
            f"It's gitignored (too large for git), so it won't show up from a fresh "
            f"clone/pull -- place your full candidate pool there before running this, "
            f"or pass --candidates <path>."
        )

    if not JD_DOCX.exists():
        sys.exit(
            f"Job description not found: {JD_DOCX}\n"
            f"embed_job.py reads it directly -- make sure it's present at that path."
        )

    if args.sample_size:
        print(f"Sampling first {args.sample_size} candidates from {candidates_path} ...")
        candidates_path = _sample_candidates(candidates_path, args.sample_size)

    # Imported here (not at module load time) so --help works even before
    # heavy deps (sentence-transformers, python-docx, joblib) are installed.
    from preprocessing import run_honeypot
    from preprocessing import feature_extractor
    from preprocessing import build_resume_text
    from embeddings import embed_job
    from embeddings import embed_candidates
    from ranking import jd_analyzer

    # run_honeypot.py's INPUT_FILE constant (confirmed name, DATA_DIR /
    # "candidates.jsonl") is read by name inside main() on every call, so
    # patching it on the module object before calling main() works.
    run_honeypot.INPUT_FILE = candidates_path

    start = time.time()

    print("\n=== Stage 1/6: honeypot detection ===")
    run_honeypot.main()

    print("\n=== Stage 2/6: feature extraction ===")
    feature_extractor.main()

    print("\n=== Stage 3/6: resume text generation ===")
    build_resume_text.main()

    print("\n=== Stage 4/6: job description embedding ===")
    embed_job.main()

    print("\n=== Stage 5/6: candidate embedding ===")
    embed_candidates.main()

    print("\n=== Stage 6/6: JD capability profile ===")
    jd_analyzer.main()

    elapsed = time.time() - start
    print(f"\nPre-computation complete in {elapsed:.1f}s")
    print(f"Artifacts ready in: {ARTIFACTS_DIR}")
    print("You can now run: python run_pipeline.py")


if __name__ == "__main__":
    main()
