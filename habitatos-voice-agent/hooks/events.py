"""
Room + session event handlers.

OOS gating and optimistic replies now live on the `HabitatAgent` class in
`agent.py` (via `on_user_turn_completed`) — this module is just for room
lifecycle and conversation-item memory tracking.
"""

import asyncio
import logging

from memory.long_term import long_term_memory

logger = logging.getLogger(__name__)


async def on_participant_connected(session, participant, short_term_memory) -> None:
    """Log join and deliver the greeting via the active AgentSession."""
    logger.info("Participant connected: %s", participant.identity)
    await session.say(
        "Hi, I'm HabitatOS! I can help you search rental properties, "
        "set up alerts, or manage your listings. "
        "What are you looking for today?"
    )


async def on_participant_disconnected(participant, short_term_memory, user_id: str) -> None:
    """Log disconnect and persist session memory (fire-and-forget)."""
    logger.info("Participant disconnected: %s", participant.identity)
    asyncio.create_task(_store_memory(user_id, short_term_memory.get_turns()))


async def on_conversation_item_added(item, short_term_memory) -> None:
    """Append both user and assistant conversation items to short-term memory."""
    role = getattr(item, "role", None)
    text = getattr(item, "text_content", None)
    if role in ("user", "assistant") and text:
        short_term_memory.append(role, text)


async def _store_memory(user_id: str, turns: list[dict]) -> None:
    """Background write — summarization + DB insert both run off-loop."""
    try:
        await long_term_memory.astore(user_id, turns)
    except Exception:
        logger.exception("Background memory store failed for user_id=%s", user_id)
