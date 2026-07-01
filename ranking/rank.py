from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

EMBEDDINGS = ROOT / "artifacts" / "candidate_embeddings.npy"
IDS = ROOT / "artifacts" / "candidate_ids.json"
FEATURES = ROOT / "artifacts" / "candidate_features.pkl"
JD = ROOT / "artifacts" / "jd_embedding.npy"
JD_PROFILE = ROOT / "artifacts" / "jd_profile.json"

OUTPUT = ROOT / "artifacts" / "ranked_candidates.pkl"
TOP500 = ROOT / "artifacts" / "top500.json"


# ------------------------------------------------------
# Load JD Capability Profile
# ------------------------------------------------------
# NOTE: jd_profile.json is now expected to carry both the capability
# importance weights AND the notice-period thresholds used by
# notice_score() below. Two shapes are supported so this doesn't break
# on the old flat-dict format:
#   {"capabilities": {...weights...}, "notice": {"preferred_days": N, "buyout_days": N}}
#   {...weights...}   <- old flat format, notice thresholds fall back to defaults

with open(JD_PROFILE, "r", encoding="utf-8") as f:
    _JD_RAW = json.load(f)

if "capabilities" in _JD_RAW:
    JD_WEIGHTS = _JD_RAW["capabilities"]
else:
    JD_WEIGHTS = _JD_RAW

# Fallback defaults if jd_profile.json hasn't been updated with a "notice"
# block yet. Update jd_profile.json directly to override these.
JD_NOTICE = _JD_RAW.get(
    "notice",
    {"preferred_days": 30, "buyout_days": 60},
)


# ------------------------------------------------------
# Capability Score
# ------------------------------------------------------

def capability_score(capabilities):
    score = 0.0
    matched = {}

    total_possible = sum(JD_WEIGHTS.values())

    for capability, importance in JD_WEIGHTS.items():

        if capability not in capabilities:
            continue

        confidence = capabilities[capability]["score"]

        score += confidence * importance

        matched[capability] = round(confidence, 3)

    if total_possible > 0:
        score /= total_possible

    return score, matched


# ------------------------------------------------------
# Structured Score
# ------------------------------------------------------
# FIX: previously raw values (years_of_experience, num_jobs, num_skills)
# were multiplied directly by their weights without normalization.
# A candidate with num_skills=40 could swing the score more than the
# 0.08 weight intended, because the raw scale (5-40) dwarfs a 0-1 scale.
# Each unbounded feature is now min-max normalized against the dataset
# BEFORE its weight is applied, so weights actually reflect intended
# contribution.

UNBOUNDED_FEATURES = ["years_of_experience", "num_jobs", "num_skills"]


def compute_feature_ranges(features: list[dict]) -> dict[str, tuple[float, float]]:
    """Compute (min, max) for each unbounded feature across the full dataset."""
    ranges = {}
    for key in UNBOUNDED_FEATURES:
        vals = [f[key] for f in features if f.get(key) is not None]
        if not vals:
            ranges[key] = (0.0, 1.0)
        else:
            ranges[key] = (min(vals), max(vals))
    return ranges


def _norm(val: float, lo: float, hi: float) -> float:
    if hi - lo < 1e-8:
        return 0.0
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def structured_score(feature: dict, ranges: dict[str, tuple[float, float]]) -> float:

    score = 0.0

    # ----------------------------
    # Experience (normalized)
    # ----------------------------
    lo, hi = ranges["years_of_experience"]
    score += 0.25 * _norm(feature["years_of_experience"], lo, hi)

    # ----------------------------
    # Career progression (normalized)
    # ----------------------------
    lo, hi = ranges["num_jobs"]
    score += 0.15 * _norm(feature["num_jobs"], lo, hi)

    # ----------------------------
    # Technical breadth (normalized)
    # ----------------------------
    lo, hi = ranges["num_skills"]
    score += 0.08 * _norm(feature["num_skills"], lo, hi)

    # ----------------------------
    # Engineering activity (already 0-1)
    # ----------------------------
    score += 0.15 * feature["github_activity"]

    # ----------------------------
    # Recruiter behaviour (already 0-1)
    # ----------------------------
    # NOTE: recruiter_response / interview_completion / offer_acceptance
    # remain here too (small weights, career-quality signal) in addition
    # to feeding the new behavioral_score() below (availability/urgency
    # signal). Same underlying fields, two different questions being
    # asked of them, so double-use is intentional, not a duplicate bug.
    score += 0.10 * feature["recruiter_response"]
    score += 0.10 * feature["interview_completion"]
    score += 0.10 * feature["offer_acceptance"]

    # ----------------------------
    # Profile quality (already 0-1)
    # ----------------------------
    score += 0.07 * feature["profile_completeness"]

    return score


# ------------------------------------------------------
# Behavioral score
# ------------------------------------------------------
# REPLACES the old behavioral_multiplier(). That function was a
# multiplicative term clamped to [0.5, 1.25] applied to raw_score before
# the final minmax — which is how a raw_final_score above 1.0 could sneak
# through (e.g. top100_again.json final_score max = 1.0096) if minmax
# wasn't applied consistently downstream of it.
#
# behavioral_score() below is bounded to [0, 1] by construction (four
# components, each already 0-1, weighted to sum to 1.0), so it's used as
# an ADDITIVE component alongside semantic/capability/structured rather
# than a multiplier — see the reweighted 0.45/0.25/0.10/0.20 split in
# main(). This can't produce a score outside [0, 1] on its own, and the
# final minmax below still guarantees the population as a whole spans
# [0, 1] exactly.

def months_since(date_string):
    if not date_string:
        return 24
    try:
        last = datetime.strptime(date_string[:10], "%Y-%m-%d")
        today = datetime.today()
        return (today.year - last.year) * 12 + (today.month - last.month)
    except Exception:
        return 24


def recency_score(last_active):
    months = months_since(last_active)
    if months <= 1:
        return 1.0
    elif months <= 3:
        return 0.8
    elif months <= 6:
        return 0.5
    elif months <= 12:
        return 0.2
    return 0.0


def responsiveness_score(feature):
    # NOTE: adapted to read from the `feature` dict (as produced by
    # feature_extractor.py) rather than candidate["redrob_signals"] —
    # feature_extractor.py already flattened these under different key
    # names: recruiter_response_rate -> recruiter_response,
    # interview_completion_rate -> interview_completion.
    rr = feature.get("recruiter_response", 0.5)
    ic = feature.get("interview_completion", 0.5)
    return rr * 0.5 + ic * 0.5


def availability_score(feature):
    # "Have they marked themselves available" -- the most direct signal
    # in the schema for the JD's "not actually available" concern.
    return 1.0 if feature.get("open_to_work_flag") else 0.4


def notice_score(days):
    # Thresholds keyed off jd_profile.json's "notice" block rather than
    # hardcoded, so this tracks the JD if the preferred/buyout window
    # ever changes.
    preferred = JD_NOTICE["preferred_days"]
    buyout = JD_NOTICE["buyout_days"]
    if days is None:
        return 0.5
    if days <= preferred:
        return 1.0
    if days <= buyout + 30:
        return 0.7
    if days <= buyout + 60:
        return 0.4
    return 0.1


def behavioral_score(feature: dict) -> float:
    recent = recency_score(feature.get("last_active_date"))
    response = responsiveness_score(feature)
    available = availability_score(feature)
    notice = notice_score(feature.get("notice_period_days"))
    return (
        0.30 * recent
        + 0.25 * response
        + 0.20 * available
        + 0.25 * notice
    )


# ------------------------------------------------------
# Min-Max Normalization (for final aggregate arrays)
# ------------------------------------------------------

def minmax(values):

    values = np.asarray(values, dtype=np.float32)

    mn = values.min()
    mx = values.max()

    if mx - mn < 1e-8:
        return np.ones_like(values)

    return (values - mn) / (mx - mn)


# ------------------------------------------------------
# Ranking
# ------------------------------------------------------

def main():

    print("Loading artifacts...")

    candidate_embeddings = np.load(EMBEDDINGS)

    jd_embedding = np.load(JD)

    semantic_scores = candidate_embeddings @ jd_embedding

    with open(IDS, "r") as f:
        ids = json.load(f)

    features = joblib.load(FEATURES)

    # Compute normalization ranges once, across the whole dataset,
    # before scoring any individual candidate.
    feature_ranges = compute_feature_ranges(features)
    print("Feature ranges used for normalization:", feature_ranges)
    print("Notice-period thresholds:", JD_NOTICE)

    capability_scores = []
    structured_scores = []
    matched_caps = []
    behavioral_scores = []

    for feature in features:

        cap_score, matched = capability_score(
            feature["capabilities"]
        )
        capability_scores.append(cap_score)

        structured_scores.append(
            structured_score(feature, feature_ranges)
        )

        matched_caps.append(matched)

        behavioral_scores.append(
            behavioral_score(feature)
        )

    semantic_scores = minmax(semantic_scores)
    structured_scores = minmax(structured_scores)

    # FIX (carried over from rank _re.py): capability_scores needs the
    # same minmax stretch as semantic/structured, or its real weight is
    # muted relative to its intended 0.25 contribution below.
    capability_scores = minmax(capability_scores)

    # behavioral_score() is already bounded [0, 1] by construction, and
    # unlike the other three signals we deliberately do NOT minmax-stretch
    # it here: stretching would inflate small real differences in
    # recency/responsiveness/notice into a full 0-1 spread, which
    # overstates how much behavioral signal should matter relative to
    # capability/semantic fit. It's used as computed.

    # ------------------------------------------------------
    # Compute raw scores
    # ------------------------------------------------------
    # Reweighted from the old 0.55 / 0.30 / 0.15 (semantic / capability /
    # structured) split to make room for behavioral_score as a fourth
    # additive component. Ratios between the original three are preserved
    # (0.55:0.30:0.15 == 0.4889:0.2667:0.1333, scaled to leave 0.20 for
    # behavioral): 0.45 / 0.25 / 0.10 / 0.20.

    raw_final_scores = []

    for i in range(len(ids)):
        # FIX: this had been reverted to a multiplicative
        # (base_ml_score * behavioral_multiplier, floor 0.1) combination,
        # which directly contradicts the design rationale documented above
        # behavioral_score() — that a bounded-[0,1] behavioral signal
        # should be an ADDITIVE component so it nudges the ranking rather
        # than dominating it. Under the multiplicative version, a
        # candidate with a near-perfect semantic/capability/structured fit
        # but a low behavioral_score (e.g. not marked open-to-work, long
        # notice period) could lose up to ~90% of their score instead of
        # the intended ~20% max swing. Restored to the documented additive
        # split.
        raw_score = (
            0.45 * semantic_scores[i]
            + 0.25 * capability_scores[i]
            + 0.10 * structured_scores[i]
            + 0.20 * behavioral_scores[i]
        )

        raw_final_scores.append(raw_score)

    # ------------------------------------------------------
    # Normalize final scores
    # ------------------------------------------------------

    final_scores = minmax(raw_final_scores)

    ranking = []

    for i in range(len(ids)):

        ranking.append({

            "candidate_id": ids[i],

            "semantic_score": round(
                float(semantic_scores[i]),
                4,
            ),

            "capability_score": round(
                float(capability_scores[i]),
                4,
            ),

            "structured_score": round(
                float(structured_scores[i]),
                4,
            ),

            "matched_capabilities": matched_caps[i],

            "behavioral_score": round(
                float(behavioral_scores[i]),
                4,
            ),

            "final_score": round(
                float(final_scores[i]),
                4,
            ),

        })
    # REPLACE THIS:
    # ranking.sort(
    #     key=lambda x: x["final_score"],
    #     reverse=True,
    # )

    # WITH THIS:
    ranking.sort(
        key=lambda x: (x["final_score"], x["candidate_id"]),
        reverse=True,
    )
    # ranking.sort(
    #     key=lambda x: x["final_score"],
    #     reverse=True,
    # )

    joblib.dump(
        ranking,
        OUTPUT,
    )

    with open(
        TOP500,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            ranking[:500],
            f,
            indent=2,
        )

    print(f"\nRanked {len(ranking)} candidates.")
    print(f"Saved -> {OUTPUT}")
    print(f"Saved -> {TOP500}")


if __name__ == "__main__":
    main()
