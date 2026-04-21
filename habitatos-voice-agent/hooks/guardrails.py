"""
Voice-side guardrails — out-of-scope filtering and optimistic response hinting.

PII masking and prompt injection detection are handled server-side by the
FastMCP middleware stack (Presidio + injection patterns). This module only
has the two gates that save LLM tokens by running pre-LLM.
"""

import logging
import random

logger = logging.getLogger(__name__)

# ── Out-of-Scope Detection ──────────────────────────────────────
IN_SCOPE_KEYWORDS = [
    "property", "flat", "apartment", "bhk", "rent", "rental",
    "house", "room", "pg", "studio", "listing",
    "bangalore", "mumbai", "delhi", "noida", "hyderabad", "pune",
    "near", "area", "location", "city",
    "search", "find", "show", "list", "create alert", "delete alert",
    "set alert", "save listing", "add listing", "price", "budget",
    "bedroom", "bathroom", "furnished", "pet", "parking", "amenities",
]

OUT_OF_SCOPE_RESPONSES = [
    "I'm only able to help with rental properties right now.",
    "That's outside what I can help with — I focus on rental property search and alerts.",
    "I specialize in rental properties. Can I help you find a place or set up a listing alert?",
]


def is_in_scope(text: str) -> bool:
    """
    Fast keyword check — runs BEFORE LLM, costs nothing.
    Greetings and short utterances always pass through.
    """
    if len(text.split()) <= 4:
        return True
    lowered = text.lower()
    return any(kw in lowered for kw in IN_SCOPE_KEYWORDS)


def get_oos_response() -> str:
    return random.choice(OUT_OF_SCOPE_RESPONSES)


# ── Optimistic Response ─────────────────────────────────────────
_TOOL_KEYWORDS = [
    "find", "search", "show", "list", "create alert",
    "delete", "set up", "set alert", "remove alert",
]


def should_give_optimistic_response(text: str) -> bool:
    """Returns True if the utterance likely requires a tool call."""
    lowered = text.lower()
    return any(kw in lowered for kw in _TOOL_KEYWORDS)
