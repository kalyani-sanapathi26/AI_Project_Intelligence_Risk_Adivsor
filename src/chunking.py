from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Split extracted project documents into smaller chunks
    suitable for embedding and retrieval.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []

    for document in documents:

        text = document["text"]

        document_chunks = text_splitter.split_text(text)

        for chunk in document_chunks:

            chunks.append({
                "filename": document["filename"],
                "file_type": document["file_type"],
                "text": chunk
            })

    return chunks