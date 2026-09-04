from sentence_transformers import SentenceTransformer


# Load the embedding model once
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


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