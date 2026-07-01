from __future__ import annotations

import json
from pathlib import Path

import joblib
from tqdm import tqdm

from utils.text_aggregator import aggregate_candidate
from utils.capability_detector import detect_capabilities

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "artifacts" / "clean_candidates.jsonl"
OUTPUT = ROOT / "artifacts" / "candidate_features.pkl"


def extract_features(candidate):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    signals = candidate.get("redrob_signals", {})

    resume = aggregate_candidate(candidate)
    capabilities = detect_capabilities(resume)

    features = {}
    features["candidate_id"] = candidate["candidate_id"]

    # --- FIX: Extract Title & Corporate Industry Pedigree ---
    features["current_title"] = profile.get("current_title", "")
    
    # Collect all companies the candidate has ever worked for (Current + History)
    features["all_companies"] = [profile.get("current_company", "")] + [
        job.get("company", "") for job in career if job.get("company")
    ]
    
    # Collect all industries the candidate has operated within
    features["all_industries"] = [profile.get("current_industry", "")] + [
        job.get("industry", "") for job in career if job.get("industry")
    ]
    # ---------------------------------------------------------

    features["years_of_experience"] = profile.get("years_of_experience", 0)
    features["num_jobs"] = len(career)
    features["num_skills"] = len(skills)
    features["num_degrees"] = len(education)
    features["github_activity"] = signals.get("github_activity_score", 0)
    features["profile_completeness"] = signals.get("profile_completeness_score", 0)
    features["recruiter_response"] = signals.get("recruiter_response_rate", 0)
    features["interview_completion"] = signals.get("interview_completion_rate", 0)
    features["offer_acceptance"] = signals.get("offer_acceptance_rate", 0)
    
    features["last_active_date"] = signals.get("last_active_date", None)
    features["open_to_work_flag"] = signals.get("open_to_work_flag", False)
    features["notice_period_days"] = signals.get("notice_period_days", None)
    features["capabilities"] = capabilities

    return features




def main():

    feature_list = []

    with open(INPUT, "r", encoding="utf-8") as f:

        for line in tqdm(
            f,
            desc="Extracting Features",
        ):

            candidate = json.loads(line)

            feature_list.append(
                extract_features(candidate)
            )

    joblib.dump(
        feature_list,
        OUTPUT,
    )

    print()

    print(f"Saved {len(feature_list)} feature vectors")

    print(OUTPUT)


if __name__ == "__main__":
    main()