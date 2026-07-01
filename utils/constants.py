"""
Project-wide constants.
"""

from pathlib import Path

# -----------------------------
# Project Paths
# -----------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
ARTIFACT_DIR = ROOT_DIR / "artifacts"

# -----------------------------
# Embedding
# -----------------------------

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_BATCH_SIZE = 64

# -----------------------------
# Ranking
# -----------------------------

TOP_K = 100
TOP_RERANK = 500

# -----------------------------
# Random Seed
# -----------------------------

RANDOM_STATE = 42