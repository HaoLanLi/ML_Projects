# This function takes a document and splits it into smaller chunks of text based on the specified chunk size and overlap. 
# It returns a list of dictionaries, each containing the filename, chunk ID, and the content of the chunk.
def chunk_document(document, chunk_size=300, overlap=50):
    content = document["content"]
    filename = document["filename"]

    lines = content.split("\n")

    chunks = []
    chunk_id = 0

    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue

        chunks.append({
            "filename": filename,
            "chunk_id": chunk_id,
            "content": cleaned_line
        })
        chunk_id += 1

    return chunks