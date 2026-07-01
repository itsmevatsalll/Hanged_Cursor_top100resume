from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "artifacts" / "clean_candidates.jsonl"

OUTPUT_DIR = ROOT / "artifacts"

SKILLS_FILE = OUTPUT_DIR / "unique_skills.json"
COMPANIES_FILE = OUTPUT_DIR / "unique_companies.json"
TITLES_FILE = OUTPUT_DIR / "unique_titles.json"
FIELDS_FILE = OUTPUT_DIR / "unique_fields_of_study.json"
STATS_FILE = OUTPUT_DIR / "dataset_statistics.json"


def save_counter(counter: Counter, path: Path):

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            dict(counter.most_common()),
            f,
            indent=4,
            ensure_ascii=False,
        )


def main():

    skills = Counter()
    companies = Counter()
    titles = Counter()
    fields = Counter()

    total = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        for line in tqdm(f, desc="Analyzing Dataset"):

            candidate = json.loads(line)

            total += 1

            for skill in candidate.get("skills", []):

                if skill.get("name"):
                    skills[skill["name"].strip()] += 1

            for job in candidate.get("career_history", []):

                if job.get("company"):
                    companies[job["company"].strip()] += 1

                if job.get("title"):
                    titles[job["title"].strip()] += 1

            for edu in candidate.get("education", []):

                if edu.get("field_of_study"):
                    fields[
                        edu["field_of_study"].strip()
                    ] += 1

    save_counter(skills, SKILLS_FILE)
    save_counter(companies, COMPANIES_FILE)
    save_counter(titles, TITLES_FILE)
    save_counter(fields, FIELDS_FILE)

    stats = {
        "total_candidates": total,
        "unique_skills": len(skills),
        "unique_companies": len(companies),
        "unique_titles": len(titles),
        "unique_fields_of_study": len(fields),
    }

    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=4)

    print("\nAnalysis Complete.\n")

    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    main()