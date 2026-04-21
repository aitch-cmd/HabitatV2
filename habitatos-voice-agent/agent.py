"""
HabitatOS Voice Agent — Main Entrypoint

Wires a LiveKit `AgentSession` (livekit-agents 1.x) with:
  • MCP tools (via FastMCP streamable-HTTP client in `mcp_bridge`)
  • Short-term + long-term memory
  • Voice-side guardrails (out-of-scope deflection, optimistic replies)
"""

import asyncio
import logging

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    StopResponse,
    WorkerOptions,
    cli,
)
from livekit.plugins import anthropic, cartesia, deepgram, silero

from config import CARTESIA_VOICE_ID
from hooks.events import (
    on_conversation_item_added,
    on_participant_connected,
    on_participant_disconnected,
)
from hooks.guardrails import (
    get_oos_response,
    is_in_scope,
    should_give_optimistic_response,
)
from mcp_bridge import build_habitat_tools
from memory.long_term import long_term_memory
from memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)


# ── System Prompt Builder ────────────────────────────────────────

def build_system_prompt(user_id: str) -> str:
    """Base persona plus any long-term memories we have for this user."""
    base = (
        "You are HabitatOS, a friendly and efficient rental property assistant "
        "specializing in the Indian rental market. You help users search for "
        "properties, set up listing alerts, and manage their rental preferences.\n\n"
        "Voice-first rules:\n"
        "- Speak naturally, no markdown, no bullet points, no special formatting.\n"
        "- Keep responses under 3 sentences unless you are listing properties.\n"
        "- Always confirm before creating or deleting anything.\n"
        "- Never read out MongoDB ObjectIds — say 'done' or use a friendly reference.\n"
        "- Use Indian English conventions (lakh, BHK, etc.).\n"
    )

    memories = long_term_memory.retrieve(
        query="user preferences and history", user_id=user_id
    )
    if memories:
        base += f"\nWhat you already know about this user:\n{memories}\n"

    return base


# ── Agent ────────────────────────────────────────────────────────

class HabitatAgent(Agent):
    """
    Habitat agent with pre-LLM gating:
      • Drops out-of-scope turns before an LLM call is made (`StopResponse`).
      • Emits an optimistic "Let me check…" for tool-heavy phrasings.
    """

    def __init__(self, instructions: str) -> None:
        super().__init__(
            instructions=instructions,
            tools=build_habitat_tools(),
        )

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        text = (new_message.text_content or "").strip()
        if not text:
            return

        if not is_in_scope(text):
            logger.info("OOS utterance — deflecting without LLM call")
            await self.session.say(get_oos_response())
            raise StopResponse()

        if should_give_optimistic_response(text):
            # Fire-and-forget so the main LLM turn still runs in parallel.
            asyncio.create_task(self.session.say("Let me check that for you."))


# ── Entrypoint ───────────────────────────────────────────────────

async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    user_id = ctx.room.name or "anonymous"
    logger.info("Session started — user_id=%s, room=%s", user_id, ctx.room.name)

    short_term = ShortTermMemory()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3", language="en-IN"),
        llm=anthropic.LLM(model="claude-sonnet-4-5"),
        tts=cartesia.TTS(voice=CARTESIA_VOICE_ID),
        allow_interruptions=True,
    )

    agent = HabitatAgent(instructions=build_system_prompt(user_id))

    # ── Wire event hooks ─────────────────────────────────────────

    @session.on("conversation_item_added")
    def _on_item(event) -> None:
        asyncio.ensure_future(
            on_conversation_item_added(event.item, short_term)
        )

    @ctx.room.on("participant_connected")
    def _on_join(participant) -> None:
        asyncio.ensure_future(
            on_participant_connected(session, participant, short_term)
        )

    @ctx.room.on("participant_disconnected")
    def _on_leave(participant) -> None:
        asyncio.ensure_future(
            on_participant_disconnected(participant, short_term, user_id)
        )

    # ── Start ────────────────────────────────────────────────────

    await session.start(agent=agent, room=ctx.room)

    # Greet anyone already in the room at start time.
    for participant in ctx.room.remote_participants.values():
        await on_participant_connected(session, participant, short_term)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
