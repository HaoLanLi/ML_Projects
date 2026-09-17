from loader import load_documents
from chunked import chunk_document
from engine import BM25Engine
import ollama


# Load documents
documents = load_documents("Knowledge_base")

# Create chunks
all_chunks = []

for document in documents:
    chunks = chunk_document(document)
    all_chunks.extend(chunks)


# Create BM25 engine
engine = BM25Engine(all_chunks)

query = input("Enter your question: ")
results = engine.rank_documents(query, top_n=3)

score_threshold = 1.0

if not results or results[0]["score"] < score_threshold:
    print("No relevant information found in the knowledge base.")
    exit()

context_parts = []
for result in results:
    context_parts.append(result["content"])

context = "\n\n".join(context_parts)

# ========================================
# 7. Create prompt
# ========================================
prompt = f"""
Answer the question using only the context in Knowledge Base.

If the answer cannot be found in the knowledge base, then say:
"I don't know based on the provided knowledge."

Context:
{context}

Question:
{query}
"""

response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nAnswer:")
print(response.message.content)

print("\nSources:")
print("=" * 60)

for result in results:
    print("Filename:", result["filename"])
    print("Chunk ID:", result["chunk_id"])
    print("Score:", result["score"])
    print("-" * 60)
