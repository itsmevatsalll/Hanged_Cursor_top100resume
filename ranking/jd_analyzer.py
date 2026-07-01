from __future__ import annotations

import json
import re
from pathlib import Path

from utils.capability_schema import CAPABILITIES

ROOT = Path(__file__).resolve().parent.parent

JD_FILE = ROOT / "artifacts" / "jd_text.txt"
OUTPUT = ROOT / "artifacts" / "jd_profile.json"


def normalize(text: str) -> str:
    return (
        text.lower()
        .replace("-", " ")
        .replace("_", " ")
        .replace("/", " ")
    )


SECTION_WEIGHTS = {
    "required": 3.0,
    "preferred": 2.0,
    "nice to have": 1.5,
}

# FIX: SECTION_WEIGHTS was defined but never actually used anywhere in
# main() — the JD text was scanned as one flat, undifferentiated blob, so
# a skill mentioned under "nice to have" scored identically to one under
# "required." That silently discarded exactly the signal the rest of the
# pipeline's comments call out as most important (the JD's "absolutely
# need" section). split_sections() below splits the JD on heading lines
# so SECTION_WEIGHTS can actually be applied per section.
#
# ASSUMPTION: headings are matched case-insensitively at the start of a
# line, optionally preceded by markdown/bullet markers (#, -, *, digits)
# and followed by a colon (e.g. "Required:", "## Preferred Skills",
# "- Nice to have"). Text before the first recognized heading is treated
# as a neutral "general" section (weight 1.0 — same behavior as before
# this fix). If your jd_text.txt uses a different heading style, adjust
# HEADING_PATTERN below.
HEADING_PATTERN = re.compile(
    r"^\s*[#>\-\*\d\.\)]*\s*(required|preferred|nice[\s-]?to[\s-]?have)\b",
    re.IGNORECASE,
)


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"general": []}
    current = "general"

    for line in text.splitlines():
        match = HEADING_PATTERN.match(line)
        if match:
            current = re.sub(r"[\s-]+", " ", match.group(1).lower()).strip()
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)

    return {name: "\n".join(lines) for name, lines in sections.items()}


def main():

    text = JD_FILE.read_text(encoding="utf-8")
    sections = split_sections(text)

    weights = {}

    for capability, config in CAPABILITIES.items():

        score = 0.0

        for section_name, section_text in sections.items():

            if not section_text:
                continue

            section_weight = SECTION_WEIGHTS.get(section_name, 1.0)
            norm_section = normalize(section_text)

            for skill in config["skills"]:
                if normalize(skill) in norm_section:
                    score += 2 * section_weight

            for phrase in config["phrases"]:
                if normalize(phrase) in norm_section:
                    score += 1 * section_weight

        if score > 0:
            weights[capability] = score

    if not weights:
        raise RuntimeError("No capabilities detected from JD.")

    # --------------------------------------------------
# Importance Calibration
# --------------------------------------------------

    IMPORTANCE = {
        "retrieval": 1.00,
        "ranking": 0.95,
        "recommendation": 0.90,
        "evaluation": 0.90,
        "production_ml": 0.85,
        "llm": 0.80,
        "backend": 0.75,
        "vector_db": 0.75,
        "embeddings": 0.70,
        "cloud": 0.60,
        "distributed": 0.60,
        "data_engineering": 0.55,
        "opensource": 0.40,
        "fine_tuning": 0.40,
        "streaming": 0.30,
        "devops": 0.30,
    }

    for capability in list(weights.keys()):
        weights[capability] *= IMPORTANCE.get(capability, 0.5)

    maximum = max(weights.values())

    for capability in weights:
        weights[capability] /= maximum

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    print("\nJD Capability Profile\n")

    for k, v in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:20} {v:.2f}")

    print(f"\nSaved -> {OUTPUT}")


if __name__ == "__main__":
    main()