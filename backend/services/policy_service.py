"""
Policy Service — RAG over RBI Guidelines on Digital Lending.

Loads the Chroma vector database ONCE at startup.
Returns evidence with proper metadata (Authority, Title, Page).
"""

import logging
from typing import Optional

from backend.config import settings
from backend.schemas.underwriting import PolicyEvidence as UWPolicyEvidence
from backend.schemas.policy import PolicyEvidence, PolicySearchResponse

logger = logging.getLogger(__name__)


class PolicyService:
    """Singleton service for RBI policy RAG retrieval."""

    _instance: Optional["PolicyService"] = None

    def __init__(self):
        self._vectorstore = None
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "PolicyService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        """Load the Chroma vector store. Called once at startup."""
        if self._loaded:
            return

        vector_db_path = settings.vector_db_path

        if not vector_db_path.exists():
            logger.warning(
                "Policy vector DB not found at %s. "
                "Policy search will return empty results. "
                "Run src/build_policy_rag.py to build it.",
                vector_db_path,
            )
            self._loaded = True
            return

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_chroma import Chroma

            logger.info("Loading policy vector DB from %s", vector_db_path)

            embeddings = HuggingFaceEmbeddings(
                model_name=settings.rag_embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            self._vectorstore = Chroma(
                collection_name=settings.rag_collection_name,
                persist_directory=str(vector_db_path),
                embedding_function=embeddings,
            )

            logger.info("Policy vector DB loaded.")

        except Exception as e:
            logger.error("Failed to load policy vector DB: %s", e)
            self._vectorstore = None

        self._loaded = True

    def search(self, query: str, k: int = 4) -> list[UWPolicyEvidence]:
        """
        Search the policy knowledge base and return PolicyEvidence objects.
        Returns empty list if vector DB is unavailable.
        """
        if not self._loaded:
            self.load()

        if self._vectorstore is None:
            return []

        try:
            docs = self._vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.warning("Policy search failed: %s", e)
            return []

        results = []
        for doc in docs:
            meta = doc.metadata
            results.append(UWPolicyEvidence(
                source=meta.get("source_file", "Guidelines_on_Digital_Lending.pdf"),
                authority=meta.get("authority", "Reserve Bank of India"),
                title=meta.get("title", "Guidelines on Digital Lending"),
                page=str(meta.get("page", "")) if meta.get("page") else None,
                version_date=meta.get("version_date", "2022-09-02"),
                jurisdiction=meta.get("jurisdiction", "India"),
                content=doc.page_content,
            ))

        return results

    def search_with_schema(self, query: str, k: int = 4) -> PolicySearchResponse:
        """Search and return the full PolicySearchResponse schema."""
        evidence = self.search(query, k=k)

        # Convert to policy schema
        policy_items = []
        for e in evidence:
            policy_items.append(PolicyEvidence(
                source=e.source,
                authority=e.authority,
                title=e.title,
                page=e.page,
                version_date=e.version_date,
                jurisdiction=e.jurisdiction,
                content=e.content,
            ))

        return PolicySearchResponse(
            query=query,
            results=policy_items,
            total_results=len(policy_items),
        )


# Module-level singleton
policy_service = PolicyService.get_instance()
