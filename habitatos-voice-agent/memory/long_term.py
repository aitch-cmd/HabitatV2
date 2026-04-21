"""
Long-term memory — MongoDB vector store with sentence-transformers embeddings.
Stores session summaries and retrieves relevant user history via $vectorSearch.
"""

import logging
from datetime import datetime, timezone

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

from config import MONGODB_URL, TOP_K_MEMORIES

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "agent_memories"
DB_NAME = "habitat_db"


class LongTermMemory:
    """Persist and retrieve user interaction summaries as vector embeddings."""

    def __init__(self) -> None:
        logger.info("Initialising LongTermMemory (model=%s)", EMBEDDING_MODEL)
        self._client = MongoClient(MONGODB_URL)
        self._db = self._client[DB_NAME]
        self._col = self._db[COLLECTION_NAME]
        self._model = SentenceTransformer(EMBEDDING_MODEL)

    # ── Retrieval ────────────────────────────────────────────────

    def retrieve(self, query: str, user_id: str) -> str:
        """
        Run MongoDB $vectorSearch for the closest memories belonging to
        *user_id*. Returns top TOP_K_MEMORIES summaries joined by newlines,
        or an empty string when nothing matches.
        """
        try:
            query_embedding = self._model.encode(query).tolist()

            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "memory_vector_index",
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": TOP_K_MEMORIES * 10,
                        "limit": TOP_K_MEMORIES,
                        "filter": {"user_id": user_id},
                    }
                },
                {"$project": {"summary": 1, "_id": 0}},
            ]

            results = list(self._col.aggregate(pipeline))
            if not results:
                return ""
            return "\n".join(r["summary"] for r in results)

        except Exception:
            logger.exception("Vector search failed for user_id=%s", user_id)
            return ""

    # ── Storage ──────────────────────────────────────────────────

    def store(self, user_id: str, session_turns: list[dict]) -> None:
        """Summarize *session_turns*, embed, and persist into MongoDB."""
        if not session_turns:
            return

        summary = self.summarize_turns(session_turns)
        if not summary.strip():
            return

        try:
            embedding = self._model.encode(summary).tolist()
            self._col.insert_one(
                {
                    "user_id": user_id,
                    "summary": summary,
                    "embedding": embedding,
                    "created_at": datetime.now(timezone.utc),
                }
            )
            logger.info("Stored memory for user_id=%s (%d chars)", user_id, len(summary))
        except Exception:
            logger.exception("Failed to store memory for user_id=%s", user_id)

    # ── Summarisation (heuristic, no LLM) ────────────────────────

    @staticmethod
    def summarize_turns(turns: list[dict]) -> str:
        """
        Concatenate the last 10 turns into a plain-text paragraph.
        No LLM call — fast and deterministic.
        """
        recent = turns[-10:]
        parts = [f"{t['role']}: {t['text']}" for t in recent if t.get("text")]
        return " ".join(parts)


# Module-level singleton
long_term_memory = LongTermMemory()
