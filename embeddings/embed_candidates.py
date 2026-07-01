from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "artifacts" / "candidate_texts.jsonl"

OUTPUT_EMBEDDINGS = ROOT / "artifacts" / "candidate_embeddings.npy"
OUTPUT_IDS = ROOT / "artifacts" / "candidate_ids.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

BATCH_SIZE = 32


def batch_reader(file_path, batch_size):

    ids = []
    texts = []

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            obj = json.loads(line)

            ids.append(obj["candidate_id"])
            texts.append(obj["text"])

            if len(ids) == batch_size:
                yield ids, texts
                ids = []
                texts = []

    if ids:
        yield ids, texts


def main():

    print("\nLoading model...\n")

    model = SentenceTransformer(MODEL_NAME)

    all_embeddings = []
    all_ids = []

    total = 0

    for ids, texts in tqdm(
        batch_reader(INPUT_FILE, BATCH_SIZE),
        desc="Embedding Batches",
    ):

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        all_embeddings.append(embeddings)

        all_ids.extend(ids)

        total += len(ids)

    embeddings = np.vstack(all_embeddings)

    np.save(
        OUTPUT_EMBEDDINGS,
        embeddings,
    )

    with open(
        OUTPUT_IDS,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(all_ids, f)

    print()

    print(f"Embedded {total:,} resumes")

    print(embeddings.shape)

    print()

    print("Saved")

    print(OUTPUT_EMBEDDINGS)

    print(OUTPUT_IDS)


if __name__ == "__main__":
    main()