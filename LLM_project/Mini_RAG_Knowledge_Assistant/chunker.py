# This function takes a document and splits it into smaller chunks of text based on the specified chunk size and overlap. 
# It returns a list of dictionaries, each containing the filename, chunk ID, and the content of the chunk.
def chunk_document(document, chunk_size=300, overlap=50):
    content = document["content"]
    filename = document["filename"]

    words = content.split()

    chunks = []

    start = 0
    chunk_id = 0

    # Create a chunk containing up to chunk_size words.
    # The final chunk may contain fewer words if there are not enough words remaining.
    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunks.append({
            "filename": filename,
            "chunk_id": chunk_id,
            "content": " ".join(chunk_words)
        })

        chunk_id += 1

        start += chunk_size - overlap

    return chunks