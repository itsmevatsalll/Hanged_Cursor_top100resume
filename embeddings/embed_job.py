from __future__ import annotations

from pathlib import Path
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

JD_FILE = ROOT / "data" / "job_description.docx"

TEXT_FILE = ROOT / "artifacts" / "jd_text.txt"

OUTPUT = ROOT / "artifacts" / "jd_embedding.npy"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def read_docx(path):

    doc = Document(path)

    text = []

    for para in doc.paragraphs:

        if para.text.strip():

            text.append(para.text.strip())

    return "\n".join(text)


def main():

    print("Reading JD...")

    jd = read_docx(JD_FILE)

    with open(TEXT_FILE, "w", encoding="utf-8") as f:

        f.write(jd)

    print("Loading BGE...")

    model = SentenceTransformer(MODEL_NAME)

    embedding = model.encode(
        jd,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    np.save(
        OUTPUT,
        embedding,
    )

    print()

    print("JD Embedded")

    print(OUTPUT)


if __name__ == "__main__":
    main()