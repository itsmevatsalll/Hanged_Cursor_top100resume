"""
Creates a structured text representation of a candidate.

Instead of returning one giant string, it returns:

{
    "full_text": "...",
    "sections": {
        "headline": "...",
        "summary": "...",
        "career": "...",
        "projects": "...",
        "skills": "...",
        "education": "...",
        "current_role": "..."
    }
}

This allows downstream feature detectors to know WHERE
evidence was found.
"""

from typing import Dict, List


def _append(lines: List[str], value):

    if value is None:
        return

    if not isinstance(value, str):
        value = str(value)

    value = value.strip()

    if value:
        lines.append(value)


def aggregate_candidate(candidate: Dict):

    profile = candidate.get("profile", {})

    sections = {}

    # ---------------------------------------------------------
    # Headline
    # ---------------------------------------------------------

    headline = []

    _append(headline, profile.get("headline"))

    sections["headline"] = "\n".join(headline)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = []

    _append(summary, profile.get("summary"))

    sections["summary"] = "\n".join(summary)

    # ---------------------------------------------------------
    # Current Role
    # ---------------------------------------------------------

    current = []

    _append(current, profile.get("current_title"))

    _append(current, profile.get("current_company"))

    sections["current_role"] = "\n".join(current)

    # ---------------------------------------------------------
    # Career History
    # ---------------------------------------------------------

    career = []

    for job in candidate.get("career_history", []):

        _append(career, job.get("title"))

        _append(career, job.get("company"))

        _append(career, job.get("description"))

    sections["career"] = "\n".join(career)

    # ---------------------------------------------------------
    # Projects
    # ---------------------------------------------------------

    projects = []

    for project in candidate.get("projects", []):

        _append(projects, project.get("title"))

        _append(projects, project.get("description"))

    sections["projects"] = "\n".join(projects)

    # ---------------------------------------------------------
    # Skills
    # ---------------------------------------------------------

    skills = []

    for skill in candidate.get("skills", []):

        _append(skills, skill.get("name"))

    sections["skills"] = "\n".join(skills)

    # ---------------------------------------------------------
    # Education
    # ---------------------------------------------------------

    education = []

    for edu in candidate.get("education", []):

        _append(education, edu.get("degree"))

        _append(education, edu.get("field_of_study"))

        _append(education, edu.get("institution"))

    sections["education"] = "\n".join(education)

    # ---------------------------------------------------------
    # Full Text
    # ---------------------------------------------------------

    full_text = "\n\n".join(
        section
        for section in sections.values()
        if section
    )

    return {
        "sections": sections,
        "full_text": full_text,
    }