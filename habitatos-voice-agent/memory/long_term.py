"""
Long-term memory — MongoDB vector store with sentence-transformers embeddings.
Stores session summaries and retrieves relevant user history via $vectorSearch.

Async entry points (`aretrieve`, `astore`) push blocking pymongo + encode work
onto a thread so LiveKit's event loop is never stalled.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

from config import MONGODB_URL, TOP_K_MEMORIES

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "agent_memories"
DB_NAME = "habitat_db"

SUMMARY_MODEL = "claude-haiku-4-5-20251001"
SUMMARY_PROMPT = (
    "Summarize the following rental-assistant conversation in 2-4 sentences. "
    "Preserve concrete user preferences (locations, budget, BHK, must-haves), "
    "any saved alerts, and decisions made. Omit pleasantries and filler.\n\n"
)


class LongTermMemory:
    """Persist and retrieve user interaction summaries as vector embeddings."""

    def __init__(self) -> None:
        logger.info("Initialising LongTermMemory (model=%s)", EMBEDDING_MODEL)
        self._client = MongoClient(MONGODB_URL)
        self._db = self._client[DB_NAME]
        self._col = self._db[COLLECTION_NAME]
        self._model = SentenceTransformer(EMBEDDING_MODEL)

        # Anthropic client is optional — if no key is available we fall back
        # to a deterministic heuristic summary instead of failing the store.
        self._anthropic = None
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                from anthropic import AsyncAnthropic
                self._anthropic = AsyncAnthropic()
            except Exception:
                logger.exception("Anthropic init failed; using heuristic summaries")

    # ── Sync core (blocking — wrap in to_thread from async code) ─────

    def retrieve(self, query: str, user_id: str) -> str:
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

    def _persist(self, user_id: str, summary: str) -> None:
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

    # ── Async entry points ──────────────────────────────────────────

    async def aretrieve(self, query: str, user_id: str) -> str:
        """Async retrieval — safe to await from LiveKit coroutines."""
        return await asyncio.to_thread(self.retrieve, query, user_id)

    async def astore(self, user_id: str, session_turns: list[dict]) -> None:
        """Summarize *session_turns*, embed, and persist — fully non-blocking."""
        if not session_turns:
            return
        summary = await self._summarize(session_turns)
        if not summary.strip():
            return
        await asyncio.to_thread(self._persist, user_id, summary)

    # ── Summarisation ───────────────────────────────────────────────

    async def _summarize(self, turns: list[dict]) -> str:
        recent = [t for t in turns[-20:] if t.get("text")]
        if not recent:
            return ""

        if self._anthropic is not None:
            try:
                transcript = "\n".join(f"{t['role']}: {t['text']}" for t in recent)
                resp = await self._anthropic.messages.create(
                    model=SUMMARY_MODEL,
                    max_tokens=300,
                    messages=[{"role": "user", "content": SUMMARY_PROMPT + transcript}],
                )
                text = "".join(
                    block.text for block in resp.content
                    if getattr(block, "type", None) == "text"
                ).strip()
                if text:
                    return text
            except Exception:
                logger.exception("LLM summary failed; falling back to heuristic")

        return self._heuristic_summary(recent)

    @staticmethod
    def _heuristic_summary(turns: list[dict]) -> str:
        parts = [f"{t['role']}: {t['text']}" for t in turns[-10:]]
        return " ".join(parts)


# Module-level singleton
long_term_memory = LongTermMemory()
