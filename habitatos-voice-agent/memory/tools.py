"""
LiveKit function_tool wrappers that expose memory operations to the LLM
mid-turn. Bound per-session via a closure over `user_id` so the tool
signature the model sees stays simple (just a query string).
"""

import logging

from livekit.agents import function_tool

from memory.long_term import long_term_memory

logger = logging.getLogger(__name__)


def build_memory_tools(user_id: str) -> list:
    """Return memory tools scoped to the given user_id."""

    @function_tool()
    async def recall_memory(query: str) -> str:
        """Recall past interactions, preferences, or prior searches for this user.

        Call this when the user references something from a previous session
        ("like last time", "my usual area") or when their past preferences
        would meaningfully improve the next response. Returns a few short
        summaries, or an empty string if nothing relevant is stored.

        Args:
            query: What to look up — be specific ("budget and preferred areas
                in Bangalore"), not generic ("memory").
        """
        try:
            return await long_term_memory.aretrieve(query, user_id)
        except Exception:
            logger.exception("recall_memory failed for user_id=%s", user_id)
            return ""

    return [recall_memory]
