"""Documentation retrieval layer backed by ChromaDB.

Embeddings: this project ships with a lightweight TF-IDF embedding
function (scikit-learn) so the whole demo runs fully offline with no
external model downloads or API keys required. In production, swap
`TfidfEmbeddingFunction` for `chromadb.utils.embedding_functions
.OpenAIEmbeddingFunction` or an Anthropic/Voyage embedding function --
the rest of the pipeline (ChromaDB storage + LangGraph retrieval node)
does not need to change.
"""

import os
from pathlib import Path
from typing import List

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DOCS_DIR = DATA_DIR / "docs"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "product_docs"


class TfidfEmbeddingFunction:
    """A chromadb-compatible embedding function with no external
    dependencies beyond scikit-learn. Fits a TF-IDF vectorizer once over
    the corpus at index time and reuses it for queries.
    """

    def __init__(self, corpus: List[str]):
        self.vectorizer = TfidfVectorizer(max_features=512)
        self.vectorizer.fit(corpus)

    def __call__(self, input: List[str]) -> List[List[float]]:
        matrix = self.vectorizer.transform(input)
        return matrix.toarray().tolist()

    def embed_query(self, input: List[str]) -> List[List[float]]:
        # Same vector space for documents and queries -- TF-IDF doesn't
        # distinguish between the two the way some neural embedders do.
        return self.__call__(input)

    def name(self) -> str:
        return "tfidf-512"


def _load_docs() -> List[dict]:
    docs = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        docs.append({"id": path.stem, "text": path.read_text(encoding="utf-8")})
    return docs


def build_vector_store(persist: bool = True) -> chromadb.api.models.Collection.Collection:
    """(Re)builds the ChromaDB collection from the markdown docs in data/docs."""
    docs = _load_docs()
    if not docs:
        raise RuntimeError(f"No documentation files found in {DOCS_DIR}")

    embed_fn = TfidfEmbeddingFunction(corpus=[d["text"] for d in docs])

    client = (
        chromadb.PersistentClient(path=str(CHROMA_DIR))
        if persist
        else chromadb.EphemeralClient()
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
    )
    return collection


_collection = None


def get_collection():
    """Loads the persisted collection, building it on first use."""
    global _collection
    if _collection is not None:
        return _collection

    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        _collection = build_vector_store(persist=True)
        return _collection

    docs = _load_docs()
    embed_fn = TfidfEmbeddingFunction(corpus=[d["text"] for d in docs])
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        _collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    except Exception:
        _collection = build_vector_store(persist=True)
    return _collection


def search_docs(query: str, top_k: int = 2) -> List[dict]:
    """Returns the top_k most relevant documentation chunks for a query."""
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    docs = []
    ids = results.get("ids", [[]])[0]
    texts = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else [0.0] * len(ids)

    for doc_id, text, distance in zip(ids, texts, distances):
        docs.append({"source": doc_id, "text": text.strip(), "score": round(1 - distance, 4)})
    return docs
