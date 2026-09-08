# This is a data loader for the RAG system.
from pathlib import Path

def load_documents(folder_path):
    documents = []
    folder = Path(folder_path)

    for file_path in folder.glob("*.txt"):
        content = file_path.read_text(encoding="utf-8")
        documents.append({
            "filename": file_path.name,
            "content": content
        })
    if not documents:
        print("No .txt files found.")

    return documents