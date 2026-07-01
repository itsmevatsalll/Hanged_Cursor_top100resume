from __future__ import annotations

import json
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "artifacts" / "clean_candidates.jsonl"
OUTPUT_FILE = ROOT / "artifacts" / "candidate_texts.jsonl"


def build_resume(candidate: dict) -> str:

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    text = []

    # -------------------------------------------------
    # Profile
    # -------------------------------------------------

    text.append("Candidate Profile")

    if profile.get("headline"):
        text.append(f"Headline: {profile['headline']}")

    if profile.get("current_title"):
        text.append(f"Current Role: {profile['current_title']}")

    if profile.get("current_company"):
        text.append(f"Current Company: {profile['current_company']}")

    text.append(
        f"Years of Experience: {profile.get('years_of_experience',0)}"
    )

    if profile.get("summary"):
        text.append("")
        text.append("Professional Summary")
        text.append(profile["summary"])

    # -------------------------------------------------
    # Career
    # -------------------------------------------------

    text.append("")
    text.append("Career History")

    for job in career:

        text.append(
            f"{job.get('title','')} at {job.get('company','')}"
        )

        start = job.get("start_date", "")

        end = job.get("end_date") or "Present"

        text.append(f"{start} to {end}")

        if job.get("description"):
            text.append(job["description"])

    # -------------------------------------------------
    # Skills
    # -------------------------------------------------

    text.append("")
    text.append("Technical Skills")

    for skill in skills:

        text.append(
            f"{skill.get('name')} "
            f"({skill.get('proficiency')})"
        )

    # -------------------------------------------------
    # Education
    # -------------------------------------------------

    text.append("")
    text.append("Education")

    for edu in education:

        degree = edu.get("degree", "")

        field = edu.get("field_of_study", "")

        institute = edu.get("institution", "")

        end = edu.get("end_year", "")

        text.append(
            f"{degree} in {field}"
        )

        text.append(institute)

        if end:
            text.append(
                f"Graduated {end}"
            )

    # -------------------------------------------------
    # Behavioral Signals
    # -------------------------------------------------

    text.append("")
    text.append("Behavioral Signals")

    text.append(
        f"Recruiter response rate {signals.get('recruiter_response_rate',0):.2f}"
    )

    text.append(
        f"Interview completion rate {signals.get('interview_completion_rate',0):.2f}"
    )

    text.append(
        f"Offer acceptance rate {signals.get('offer_acceptance_rate',0):.2f}"
    )

    text.append(
        f"GitHub activity score {signals.get('github_activity_score',0)}"
    )

    return "\n".join(text)


def main():

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in tqdm(
            infile,
            desc="Building Resume Text",
        ):

            candidate = json.loads(line)

            json.dump(
                {
                    "candidate_id": candidate["candidate_id"],
                    "text": build_resume(candidate),
                },
                outfile,
                ensure_ascii=False,
            )

            outfile.write("\n")

    print()
    print("Resume text generation complete.")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()