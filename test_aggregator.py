import json

from utils.text_aggregator import aggregate_candidate

with open(
    "artifacts/clean_candidates.jsonl",
    "r",
    encoding="utf-8",
) as f:

    candidate = json.loads(next(f))

result = aggregate_candidate(candidate)

print("=" * 80)
print(result["sections"]["career"])

print("=" * 80)

print(result["sections"]["projects"])

print("=" * 80)

print(result["full_text"])