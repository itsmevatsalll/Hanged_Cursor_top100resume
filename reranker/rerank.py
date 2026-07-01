from __future__ import annotations

import json
import csv
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parent.parent


TOP500 = ROOT / "artifacts" / "top500.json"
FEATURES = ROOT / "artifacts" / "candidate_features.pkl"

OUTPUT = ROOT / "artifacts" / "top100.json"
# Replace xxx with your actual participant/team ID
CSV_OUTPUT = ROOT / "team_Hanged_Cursor.csv"

SECTION_WEIGHT = {
    "career": 1.0,      # Boosted: Prioritize actual work history
    "projects": 0.8,    # Reduced
    "summary": 0.6,
    "headline": 0.5,
    "education": 0.2,
    "inference": 0.2,
    "skills": 0.05,     # Nuked: Punish keyword stuffers
}

# SECTION_WEIGHT = {
#     "projects": 1.0,
#     "career": 0.9,
#     "summary": 0.6,
#     "headline": 0.5,
#     "skills": 0.3,
#     "education": 0.2,
#     "inference": 0.2,
# }

# ------------------------------------------------------
# FIX: deep_score normalization bug
# ------------------------------------------------------
# Previously: max_possible = sum(SECTION_WEIGHT.values()) = 3.7
# This assumes a single capability could have evidence drawn from ALL
# SEVEN sections simultaneously (projects + career + summary + headline +
# skills + education + inference). No realistic candidate profile has
# evidence spread that broadly for one capability — most have evidence
# from 1-3 sections. Dividing by 3.7 silently capped every real score
# near 0.06-0.25, regardless of how strong the evidence actually was.
#
# Fix: normalize each capability's evidence_quality against a REALISTIC
# ceiling — the sum of the 3 highest-weighted sections (the strongest
# plausible case: projects + career + summary = 2.5) — then clamp to 1.0.
# This lets genuinely strong candidates actually reach scores near 1.0.

_TOP_N_SECTIONS = 3
REALISTIC_MAX = sum(
    sorted(SECTION_WEIGHT.values(), reverse=True)[:_TOP_N_SECTIONS]
)  # 1.0 + 0.9 + 0.6 = 2.5

def corporate_pedigree_multiplier(feature):
    """Combines job title domain penalties with corporate pedigree rules."""
    # 1. Enforce Job Title Filter (Fixes the silent bug)
    title = feature.get("current_title", "").lower()
    trap_domains = [
        "marketing", "operations", "hr", "sales", 
        "accountant", "customer support", "graphic designer"
    ]
    if any(domain in title for domain in trap_domains):
        return 0.05  # 95% penalty for non-engineering domains
        
    # 2. Consulting Firm Filter (Enforces the explicit JD constraint)
    banned_consulting_firms = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]
    # FIX: feature["all_companies"] always includes current_company even
    # when it's "" (feature_extractor.py doesn't filter it the way it
    # filters career-history companies). An empty string never matches any
    # banned-firm substring, so it silently made only_consulting False for
    # anyone with a blank current_company field — letting a candidate whose
    # entire real career was at banned consulting firms dodge the penalty
    # just because their current-company field happened to be empty.
    # Filter out falsy entries so only real company names are checked.
    candidate_companies = [
        str(comp).lower() for comp in feature.get("all_companies", []) if comp
    ]
    
    if candidate_companies:
        # Check if EVERY single company they've worked at is a banned consulting firm
        only_consulting = all(
            any(firm in comp for firm in banned_consulting_firms) 
            for comp in candidate_companies
        )
        if only_consulting:
            return 0.01  # Hard penalty for lifelong consulting profiles with no product exposure
            
    return 1.0


def deep_score(feature):

    capabilities = feature["capabilities"]

    if not capabilities:
        return 0.0

    # -------------------------------------------------------------------
    # FIX: confidence-weighted averaging
    # -------------------------------------------------------------------
    # Previously every matched capability counted equally toward the
    # average (total_score / matched_caps), regardless of how strong its
    # signal was. A capability that only fired from a weak, generic
    # inference rule (e.g. production_ml getting +1 just from mentioning
    # AWS, per inference_rules.py's cloud rule) counted exactly as much
    # as a capability with strong, direct evidence (confidence ~0.9+).
    # A candidate with one genuinely strong capability and several weak
    # incidental ones got dragged down toward the weak end.
    #
    # Fix: weight each capability's contribution by its own confidence,
    # so strong signals dominate the average instead of being diluted
    # by a long tail of barely-triggered capabilities.

    total_weighted = 0.0
    total_weight = 0.0

    for capability in capabilities.values():

        confidence = capability["score"]

        unique_sections = {
            ev["section"] for ev in capability["evidence"]
        }

        evidence_quality = sum(
            SECTION_WEIGHT.get(section, 0.2) for section in unique_sections
        )

        # Normalize against the realistic ceiling, not the impossible
        # all-sections ceiling. Clamp at 1.0 in case evidence_quality
        # exceeds REALISTIC_MAX for an unusually well-evidenced capability.
        normalized_evidence = min(evidence_quality / REALISTIC_MAX, 1.0)

        weight = confidence

        total_weighted += confidence * normalized_evidence * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return min(total_weighted / total_weight, 1.0)

def main():

    with open(TOP500, "r", encoding="utf-8") as f:
        ranking = json.load(f)

    features = joblib.load(FEATURES)

    feature_map = {
        f["candidate_id"]: f
        for f in features
    }

    reranked = []

    for candidate in ranking:
        cid = candidate["candidate_id"]
        feature = feature_map[cid]

        # Deep semantic score
        ds = deep_score(feature)

        # Unified corporate and role filter multiplier
        pedigree_multiplier = corporate_pedigree_multiplier(feature)

        # Final rerank score
        final = (
            (0.70 * candidate["final_score"] + 0.30 * ds) * pedigree_multiplier
        )

        candidate["deep_score"] = round(ds, 4)
        candidate["penalty_multiplier"] = round(pedigree_multiplier, 3)
        candidate["rerank_score"] = round(final, 4)

        reranked.append(candidate)
    
    # REPLACE THIS:
    # reranked.sort(
    #     key=lambda x: x["rerank_score"],
    #     reverse=True,
    # )

    # WITH THIS:
    reranked.sort(
        key=lambda x: (x["rerank_score"], x["candidate_id"]),
        reverse=True,
    )

    # reranked.sort(
    #     key=lambda x: x["rerank_score"],
    #     reverse=True,
    # )

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            reranked[:100],
            f,
            indent=2,
        )

    print(f"Saved Top 100 -> {OUTPUT}")
    print(f"Deep score normalization ceiling: {REALISTIC_MAX}")

    # FIX: this used to redeclare CSV_OUTPUT as ROOT / "artifacts" /
    # "team_xxx.csv", a local variable that silently shadowed the
    # module-level CSV_OUTPUT = ROOT / "team_xxx.csv" declared above (near
    # OUTPUT). That meant the submission CSV was actually being written
    # one directory deeper than intended, not at the project root next to
    # run_pipeline.py where "replace xxx with your actual participant/team
    # ID" implies it should live. Removed the redundant re-declaration —
    # the module-level CSV_OUTPUT is used consistently now.

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 1. Write the exact required header
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        # 2. Write the Top 100 rows
        for idx, candidate in enumerate(reranked[:100]):
            cid = candidate["candidate_id"]
            rank_val = idx + 1
            score_val = candidate["rerank_score"]
            
            # Fetch features to generate fact-based, non-hallucinated reasoning
            feat = feature_map[cid]
            yoe = feat.get("years_of_experience", 0)
            matched_caps = len(candidate.get("matched_capabilities", {}))
            rr_rate = feat.get("recruiter_response", 0.0) * 100
            
            # Generate safe, factual reasoning string
            # FIX: penalty_multiplier < 1.0 covers two distinct causes from
            # corporate_pedigree_multiplier() -- 0.05 (non-engineering job
            # title) and 0.01 (entire career at banned consulting firms) --
            # but both used to get the same "non-technical domain
            # experience" reasoning, which is factually wrong for a
            # candidate with a real engineering title whose only issue is
            # an all-consulting career history. Branch on which penalty
            # actually fired so the CSV reasoning stays fact-based.
            if candidate["penalty_multiplier"] < 0.03:
                reasoning = f"Ranked lower: entire career history is at outsourcing/consulting firms with no product-company exposure, despite matching {matched_caps} technical capabilities."
            elif candidate["penalty_multiplier"] < 1.0:
                reasoning = f"Ranked lower due to non-technical domain experience, despite matching {matched_caps} technical capabilities."
            elif rr_rate < 30.0:
                reasoning = f"Strong technical match ({matched_caps} capabilities, {yoe} YOE) but penalized for low platform availability ({rr_rate:.0f}% response rate)."
            else:
                reasoning = f"Excellent fit displaying {matched_caps} required capabilities with {yoe} years of experience and strong recruiter responsiveness ({rr_rate:.0f}%)."
            
            writer.writerow([cid, rank_val, score_val, reasoning])

    print(f"Saved Top 100 CSV -> {CSV_OUTPUT}")

if __name__ == "__main__":
    main()
