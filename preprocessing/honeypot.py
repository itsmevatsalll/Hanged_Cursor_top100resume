from datetime import date, datetime
from typing import Any


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _months_between(start, end):
    return (end.year - start.year) * 12 + (end.month - start.month)


# --------------------------------------------------
# Honeypot Detection
# --------------------------------------------------

def is_honeypot(candidate: dict[str, Any]) -> tuple[bool, str]:
    """
    Hard honeypot detection — catches profiles with structurally impossible data.

    Checks:
      A. Expert proficiency with 0 duration on >=2 skills
      B. >=10 expert skills total
      C. Claimed YoE > total career months by more than 24 months
      D. A single job's duration_months > actual (end - start) by more than 12 months
    """

    today = date.today()
    profile = candidate.get("profile", {})
    years_exp = profile.get("years_of_experience", 0)
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # --------------------------------------------------
    # CHECK A
    # Expert proficiency claimed but 0 months of usage — on 2+ skills.
    # Knowing something at expert level without ever using it is impossible.
    # Threshold >=2 to avoid flagging a single data-entry slip.
    # --------------------------------------------------

    expert_zero = [
        s.get("name", "?")
        for s in skills
        if s.get("proficiency") == "expert"
        and (s.get("duration_months") or 0) == 0
    ]

    if len(expert_zero) >= 2:
        return True, (
            f"Expert proficiency with 0 months usage on "
            f"{len(expert_zero)} skills: {expert_zero[:4]}"
        )

    # --------------------------------------------------
    # CHECK B
    # Too many expert skills.
    # No realistic candidate holds genuine expert-level mastery across 10+ skills.
    # --------------------------------------------------

    expert_count = sum(
        1
        for s in skills
        if s.get("proficiency") == "expert"
    )

    if expert_count >= 10:
        return True, (
            f"Claims expert proficiency in "
            f"{expert_count} skills"
        )

    # --------------------------------------------------
    # CHECK C
    # Claimed years_of_experience far exceeds the sum of all job durations.
    # 24-month buffer allows for career breaks and rounding.
    # --------------------------------------------------

    total_months_worked = sum(
        job.get("duration_months", 0)
        for job in career
        if job.get("duration_months") is not None
    )

    if (years_exp * 12) > (total_months_worked + 24):
        return True, (
            f"Claimed {years_exp} yrs experience but "
            f"total career history only spans {total_months_worked} months"
        )

    # --------------------------------------------------
    # CHECK D  ← NEW
    # duration_months on a single job is impossibly large relative to its
    # actual start_date → end_date span.
    #
    # This catches the "8 years at a company founded 3 years ago" pattern
    # from the spec — e.g. start_date=2023-09 but duration_months=166.
    # Buffer of 12 months to avoid false positives from rounding / data lag.
    # --------------------------------------------------

    for job in career:
        start = _parse_date(job.get("start_date"))
        end = _parse_date(job.get("end_date")) or today
        claimed_dur = job.get("duration_months")

        if start is None or claimed_dur is None:
            continue
        if start > end:          # end-before-start is a separate concern
            continue

        actual_months = _months_between(start, end)
        overclaim = claimed_dur - actual_months

        if overclaim > 12:
            return True, (
                f"Job at '{job.get('company', '?')}' claims {claimed_dur} months "
                f"but start_date {job.get('start_date')} → "
                f"{'present' if not job.get('end_date') else job.get('end_date')} "
                f"is only {actual_months} months (overclaim: {overclaim} months)"
            )

    return False, ""


# --------------------------------------------------
# Batch Helper
# --------------------------------------------------

def filter_honeypots(candidates):
    clean = []
    flagged = {}

    for candidate in candidates:

        hp, reason = is_honeypot(candidate)

        if hp:
            flagged[
                candidate.get("candidate_id", "?")
            ] = reason
        else:
            clean.append(candidate)

    return clean, flagged