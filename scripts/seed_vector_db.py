"""Rebuilds the ChromaDB vector store from data/docs/*.md.

Normally you don't need to run this manually -- app/services/vector_store.py
builds the collection automatically on first use. This script is useful
after editing/adding documentation files, to force a rebuild.

Usage:
    PYTHONPATH=. python scripts/seed_vector_db.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.vector_store import build_vector_store  # noqa: E402


def main():
    collection = build_vector_store(persist=True)
    print(f"Rebuilt collection '{collection.name}' with {collection.count()} documents.")


if __name__ == "__main__":
    main()
