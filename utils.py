import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class TextEncoder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the text encoder with a Sentence-BERT model."""
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for input text(s)."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )

        return embeddings


class VectorStore:
    def __init__(self, dimension: int):
        """Initialize a FAISS index for vector similarity search."""
        self.index = faiss.IndexFlatL2(dimension)
        self.text_lookup: dict[int, str] = {}
        self.current_id = 0

    def add_texts(self, texts: list[str], embeddings: np.ndarray) -> None:
        """Add texts and their embeddings to the index."""
        for text in texts:
            self.text_lookup[self.current_id] = text
            self.current_id += 1

        self.index.add(embeddings.astype(np.float32))

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Search for most similar texts."""
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, indices = self.index.search(query_vector.astype(np.float32), k)

        results = [
            (self.text_lookup[int(idx)], float(dist))
            for idx, dist in zip(indices[0], distances[0])
            if idx in self.text_lookup
        ]

        return results
