import chromadb


# Persistent vector database
client = chromadb.PersistentClient(
    path="data/chroma_db"
)

collection = client.get_or_create_collection(
    name="project_documents"
)


def store_embeddings(chunks, embeddings):
    """
    Store document chunks and their embeddings
    in the ChromaDB vector store.
    """

    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        {
            "filename": chunk["filename"],
            "file_type": chunk["file_type"]
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    return len(ids)