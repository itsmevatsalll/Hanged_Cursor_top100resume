from collections import defaultdict

from utils.capability_schema import CAPABILITIES
from utils.inference_rules import INFERENCE_RULES


# SECTION_WEIGHTS = {
#     "current_role": 5.0,    # Maximize weight of what they actually do right now
#     "career": 4.0,          # High weight for past experience
#     "summary": 3.0,         # Moderate weight for professional summary
#     "projects": 1.5,        # Significantly reduced (easily fabricated)
#     "headline": 1.0, 
#     "education": 0.5,
#     "skills": 0.1           # Nuke the skills array weight to punish stuffers
# }
SECTION_WEIGHT = {
    "career": 1.0,      # Boosted: Prioritize actual work history
    "projects": 0.8,    # Reduced
    "summary": 0.6,
    "headline": 0.5,
    "education": 0.2,
    "inference": 0.2,
    "skills": 0.05,     # Nuked: Punish keyword stuffers
}
# SECTION_WEIGHTS = {
#     "projects": 5,
#     "career": 4,
#     "summary": 3,
#     "headline": 2,
#     "skills": 2,
#     "current_role": 2,
#     "education": 1,
# }


# ---------------------------------------------------------------------------
# FIX: per-capability normalization ceiling
# ---------------------------------------------------------------------------
# The original code used one flat divisor (raw / 25.0) for every capability.
# This unfairly disadvantaged capabilities with a narrower vocabulary in
# capability_schema.py (e.g. production_ml has only 2 skills + 8 phrases,
# versus retrieval's 11 skills + 15 phrases) — a narrow-vocabulary capability
# can never realistically accumulate enough raw score to approach the same
# ceiling as a broad-vocabulary one, even with strong genuine evidence.
#
# These ceilings were calibrated against realistic "strong evidence" raw
# scores for each capability (2-3 genuine matches across career/summary
# plus any inference-rule bonuses from inference_rules.py). Capabilities
# not listed fall back to the original 25.0 default.

CAPABILITY_NORMALIZER = {
    "retrieval":        25.0,
    "llm":              25.0,
    "embeddings":       22.0,
    "vector_db":        18.0,
    "backend":          20.0,
    "cloud":            18.0,
    "data_engineering": 22.0,
    "streaming":        15.0,
    "production_ml":    20.0,
    "evaluation":       15.0,
    "ranking":          18.0,
    "recommendation":   15.0,
    "distributed":      18.0,
    "opensource":       12.0,
}

DEFAULT_NORMALIZER = 25.0


def normalize(text: str) -> str:
    return (
        text.lower()
        .replace("-", " ")
        .replace("_", " ")
        .replace("/", " ")
    )


def detect_capabilities(aggregated_resume):

    sections = aggregated_resume["sections"]

    capability_scores = defaultdict(float)

    evidence = defaultdict(list)

    # ---------------------------------------------------------------------
    # FIX: detected_terms used to be a flat set(), so a term mentioned once
    # in a weak section (e.g. "Kubernetes" listed under education, weight 1)
    # triggered the exact same inference_rules.py bonus as a candidate who
    # shipped it under projects (weight 5). Shallow, throwaway mentions got
    # full inference credit. We now track the STRONGEST section weight each
    # term was matched in, so inference bonuses can be scaled by evidence
    # strength instead of applied as a flat all-or-nothing bonus.
    # ---------------------------------------------------------------------

    detected_terms = {}  # term -> strongest section weight it was found in

    # ---------------------------------------------------------
    # Explicit Detection
    # ---------------------------------------------------------

    for capability, config in CAPABILITIES.items():

        for section, text in sections.items():

            if not text:
                continue

            weight = SECTION_WEIGHT.get(section, 1)

            text = normalize(text)

            # Explicit skills

            for skill in config["skills"]:

                skill_norm = normalize(skill)

                if skill_norm in text:

                    capability_scores[capability] += 2 * weight

                    # FIX: key detected_terms by the normalized (lowercased)
                    # form of the term, not the raw casing from
                    # capability_schema.py. inference_rules.py references
                    # these terms with its own casing (sometimes Title
                    # Case, sometimes lowercase) and a case-sensitive dict
                    # key made several if_any/if_all rules silently never
                    # fire (e.g. "Docker"/"Kubernetes" never matched the
                    # stored lowercase phrase key "docker"/"kubernetes").
                    # Normalizing both sides removes that whole bug class.
                    detected_terms[skill_norm] = max(
                        detected_terms.get(skill_norm, 0),
                        weight,
                    )

                    evidence[capability].append(
                        {
                            "section": section,
                            "match": skill,
                            "type": "skill",
                        }
                    )

            # Semantic phrases

            for phrase in config["phrases"]:

                phrase_norm = normalize(phrase)

                if phrase_norm in text:

                    capability_scores[capability] += weight

                    detected_terms[phrase_norm] = max(
                        detected_terms.get(phrase_norm, 0),
                        weight,
                    )

                    evidence[capability].append(
                        {
                            "section": section,
                            "match": phrase,
                            "type": "phrase",
                        }
                    )

    # ---------------------------------------------------------
    # Rule Inference
    # ---------------------------------------------------------

    for rule in INFERENCE_RULES:

        triggering_weights = []

        if "if_any" in rule:

            triggering_weights = [
                detected_terms[normalize(term)]
                for term in rule["if_any"]
                if normalize(term) in detected_terms
            ]

        elif "if_all" in rule:

            if all(normalize(term) in detected_terms for term in rule["if_all"]):
                triggering_weights = [
                    detected_terms[normalize(term)] for term in rule["if_all"]
                ]

        if not triggering_weights:
            continue

        # FIX: previously every firing rule applied its full "infer" bonus
        # regardless of where the evidence came from. A single shallow
        # mention (e.g. education, weight 0.2) could trigger the same
        # production_ml/ranking/etc. bonus as strong evidence from
        # career (weight 1.0). Scale the bonus by the strongest
        # contributing section weight, normalized against the top
        # section weight (career = 1.0), so weak/incidental mentions
        # only partially trigger the inference bonus.
        scale = max(triggering_weights) / max(SECTION_WEIGHT.values())

        for capability, value in rule["infer"].items():

            capability_scores[capability] += value * scale

            evidence[capability].append(
                {
                    "section": "inference",
                    "match": str(rule),
                    "type": "rule",
                    "scale": round(scale, 3),
                }
            )

    # ---------------------------------------------------------
    # Normalize (per-capability ceiling, not one flat divisor)
    # ---------------------------------------------------------

    output = {}

    for capability in capability_scores:

        raw = capability_scores[capability]

        ceiling = CAPABILITY_NORMALIZER.get(capability, DEFAULT_NORMALIZER)

        output[capability] = {

            "score": min(raw / ceiling, 1.0),

            "raw_score": raw,

            "evidence": evidence[capability],

        }

    return output