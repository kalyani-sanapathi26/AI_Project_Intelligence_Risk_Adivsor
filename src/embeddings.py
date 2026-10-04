from sentence_transformers import SentenceTransformer


class _LazyEmbeddingModel:
    """Create and cache the embedding model on its first use."""

    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def encode(self, *args, **kwargs):
        return self._get_model().encode(*args, **kwargs)


# Preserve the existing embedding_model.encode(...) interface without loading
# the model until the first embedding call.
embedding_model = _LazyEmbeddingModel()


def generate_embeddings(chunks):
    """
    Generate embeddings for document chunks.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False
    )

    return embeddings
