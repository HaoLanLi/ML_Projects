from loader import load_documents
from chunker import chunk_document
from engine import BM25Engine


# Load documents
documents = load_documents("Knowledge_base")

# Create chunks
all_chunks = []

for document in documents:
    chunks = chunk_document(document)
    all_chunks.extend(chunks)


# Create BM25 engine
engine = BM25Engine(all_chunks)


# Test query
query = "supervised learning"

results = engine.rank_documents(query, top_n=3)


# Display results
print("Query:", query)
print("=" * 60)

for result in results:
    print("Filename:", result["filename"])
    print("Chunk ID:", result["chunk_id"])
    print("Score:", result["score"])
    print("Content:", result["content"][:300])
    print("-" * 60)
