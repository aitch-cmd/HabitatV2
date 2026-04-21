"""
Short-term memory — per-session conversation buffer.

Stores plain {"role", "text"} dicts and converts to livekit-agents 1.x
`ChatMessage` objects on demand (where `content` is a list of string /
content parts).
"""

import logging

from livekit.agents.llm import ChatMessage

from config import MAX_CONTEXT_TURNS

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """Session-scoped sliding window of conversation turns."""

    def __init__(self) -> None:
        self._turns: list[dict] = []

    def append(self, role: str, text: str) -> None:
        """Add a turn and trim to the last MAX_CONTEXT_TURNS entries."""
        if role not in ("user", "assistant"):
            return
        self._turns.append({"role": role, "text": text})
        if len(self._turns) > MAX_CONTEXT_TURNS:
            self._turns = self._turns[-MAX_CONTEXT_TURNS:]

    def to_chat_messages(self) -> list[ChatMessage]:
        """Convert internal turns to livekit-agents 1.x ChatMessage objects."""
        return [
            ChatMessage(role=t["role"], content=[t["text"]])
            for t in self._turns
        ]

    def get_turns(self) -> list[dict]:
        """Return a copy of the raw turn list for long-term memory storage."""
        return list(self._turns)

    def clear(self) -> None:
        """Reset for a new session."""
        self._turns.clear()
        logger.info("Short-term memory cleared")
