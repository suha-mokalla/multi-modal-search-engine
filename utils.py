from pathlib import Path

import faiss
import numpy as np
import pymupdf
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


def extract_text_from_pdf(pdf_path: str) -> list[str]:
    """Extract text from PDF file, returning a list of page contents."""
    doc = pymupdf.open(pdf_path)
    texts = []
    for page in doc:
        text = page.get_text()
        if text.strip():  # Only add non-empty pages
            texts.append(text)
    return texts


def process_pdf_folder(folder_path: str) -> list[str]:
    """Process all PDFs in a folder and return their texts."""
    all_texts = []
    pdf_files = Path(folder_path).glob("*.pdf")

    for pdf_path in pdf_files:
        try:
            texts = extract_text_from_pdf(str(pdf_path))
            # Add source information to each page
            texts = [
                f"Source: {pdf_path.name}, Page {i + 1}: {text}"
                for i, text in enumerate(texts)
            ]
            all_texts.extend(texts)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")

    return all_texts
