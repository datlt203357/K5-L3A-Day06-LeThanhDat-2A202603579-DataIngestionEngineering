"""Choose a real embedding provider (API or local) and implement this contract."""
from typing import Protocol
from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Local Sentence Transformers adapter; never falls back to mock vectors."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        # This lab uses a model downloaded to the local cache. Offline loading
        # avoids Hub retries and guarantees document text is embedded locally.
        self.model = SentenceTransformer(model_name, local_files_only=True)
        self.dimensions = self.model.get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode_document(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, question: str) -> list[float]:
        vector = self.model.encode_query(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()
