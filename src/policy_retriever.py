from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_DIR = BASE_DIR / "data" / "policy_vector_db"

COLLECTION_NAME = "credit_underwriting_policies"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

print("Loading policy knowledge base...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(VECTOR_DIR),
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4
    }
)


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_policy(query: str, k: int = 4):

    results = vectorstore.similarity_search(
        query,
        k=k
    )

    return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("POLICY RETRIEVER TEST")
    print("=" * 70)

    query = (
        "What requirements apply to collection and use "
        "of borrower data in digital lending?"
    )

    print(f"\nQuery:\n{query}")

    results = search_policy(query, k=4)

    print(f"\nRetrieved documents: {len(results)}")

    for i, doc in enumerate(results, start=1):

        print("\n" + "-" * 70)
        print(f"RESULT {i}")

        print("\nSource:")
        print(doc.metadata.get("source_file"))

        print("Page:")
        print(doc.metadata.get("page"))

        print("\nContent:")
        print(doc.page_content[:1200])