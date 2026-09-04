import asyncio
from ollama import AsyncClient
from engine import CustomBM25Engine


# ==========================================
# Simulated Internal Company Knowledge Base
# ==========================================

KNOWLEDGE_BASE = [
    (
        "Python microservices should use FastAPI because it "
        "inherently supports async/await operations and leverages "
        "Pydantic for fast data validation."
    ),

    (
        "When designing system architecture, Redis is used as an "
        "in-memory database to cache database queries and lower "
        "read latency to sub-milliseconds."
    ),

    (
        "Large Language Models (LLMs) suffer from hallucinations. "
        "Retrieval-Augmented Generation (RAG) fixes this by providing "
        "external accurate context."
    ),

    (
        "LeetCode Top 150 questions cover fundamental patterns like "
        "sliding windows, two pointers, fast and slow pointers, "
        "and dynamic programming."
    )
]


# ==========================================
# Hybrid RAG
# ==========================================

async def run_hybrid_rag(user_query: str):

    print(f"\nUser Query: '{user_query}'")

    # --------------------------------------
    # Step 1: BM25 Retrieval
    # --------------------------------------

    print("\nStep 1: Running Custom BM25 Retrieval...")

    bm25 = CustomBM25Engine(
        KNOWLEDGE_BASE
    )

    top_matches = bm25.rank_documents(
        user_query,
        top_n=1
    )

    if not top_matches:

        print("No relevant documents found.")

        return

    # top_matches contains:
    # [(document_index, bm25_score)]

    best_doc_idx, best_score = top_matches[0]

    retrieved_context = KNOWLEDGE_BASE[
        best_doc_idx
    ]

    print(
        f"\nBest Document Index: {best_doc_idx}"
    )

    print(
        f"BM25 Score: {best_score:.4f}"
    )

    print(
        f'Retrieved Context: "{retrieved_context}"'
    )


    # --------------------------------------
    # Step 2: Create System Prompt
    # --------------------------------------

    print(
        "\nStep 2: Building RAG Prompt..."
    )

    system_prompt = (
        "You are a strict technical assistant. "
        "Answer the user's query using ONLY the provided context. "
        "Do not use outside knowledge. "
        "If the answer cannot be derived from the context, "
        "say exactly: 'Information not available.'"
    )

    full_prompt = (
        f"Context:\n"
        f"{retrieved_context}\n\n"
        f"Query:\n"
        f"{user_query}"
    )


    # --------------------------------------
    # Step 3: Query Ollama
    # --------------------------------------

    print(
        "\nStep 3: Querying Local Ollama "
        "(qwen2.5:3b)..."
    )

    try:

        client = AsyncClient()

        response = await client.generate(
            model="qwen2.5:3b",
            system=system_prompt,
            prompt=full_prompt
        )

        print("\n" + "=" * 50)
        print("OLLAMA RESPONSE")
        print("=" * 50)

        print(response["response"])

        print("=" * 50)

    except Exception as e:

        print(
            "\nError connecting to Ollama:"
        )

        print(e)

        print(
            "\nMake sure Ollama is running "
            "and qwen2.5:3b is installed."
        )


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    query = (
        "Why should we pick FastAPI for Python microservices?"
    )

    asyncio.run(
        run_hybrid_rag(query)
    )