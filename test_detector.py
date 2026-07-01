import json

from utils.text_aggregator import aggregate_candidate
from utils.capability_detector import detect_capabilities


with open(
    "artifacts/clean_candidates.jsonl",
    "r",
    encoding="utf-8",
) as f:

    candidate = json.loads(next(f))

resume = aggregate_candidate(candidate)

capabilities = detect_capabilities(resume)

for capability, info in sorted(capabilities.items()):

    print("=" * 60)

    print(capability)

    print("Confidence :", info["score"])

    print("Evidence")

    for e in info["evidence"]:

        print(e)