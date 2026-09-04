from src.embeddings import embedding_model
from src.vector_store import collection


def retrieve_documents(query, top_k=3):
    """
    Retrieve the most relevant project document chunks
    from ChromaDB for a given user query.
    """

    # Convert the user's question into an embedding
    query_embedding = embedding_model.encode(
        [query]
    )[0].tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_documents = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        retrieved_documents.append({
            "text": document,
            "filename": metadata.get("filename"),
            "file_type": metadata.get("file_type"),
            "distance": distance
        })

    return retrieved_documents