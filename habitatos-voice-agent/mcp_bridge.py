"""
MCP Bridge — exposes the FastMCP server's tools as LiveKit `function_tool`s.

Speaks the real MCP protocol via the FastMCP streamable-HTTP client. No
bespoke REST shim — tools are called by their MCP-registered names. A
short-lived client is opened per call; the LLM/TTS pipeline dominates
overall latency, so the handshake cost is negligible.

Tool naming follows FastMCP's mount convention: `{namespace}_{tool}` with
the default underscore separator (see `server.py` — namespaces are
`search`, `listings`, `alerts`).
"""

import json
import logging

from fastmcp import Client
from livekit.agents import function_tool

from config import MCP_SERVER_URL, MCP_TIMEOUT

logger = logging.getLogger(__name__)

FALLBACK = "Sorry, I couldn't reach the property database right now."


# ── Transport helper ─────────────────────────────────────────────

async def _call_mcp(tool_name: str, args: dict) -> str:
    """Open a short-lived MCP client, call `tool_name`, stringify the result."""
    try:
        async with Client(MCP_SERVER_URL, timeout=MCP_TIMEOUT) as client:
            result = await client.call_tool(tool_name, args)
    except Exception:
        logger.exception("MCP call failed: %s", tool_name)
        return FALLBACK

    texts: list[str] = []
    for block in result.content or []:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    if texts:
        return "\n".join(texts)

    structured = getattr(result, "structured_content", None)
    if structured:
        return json.dumps(structured)

    return FALLBACK


# ── Tools ────────────────────────────────────────────────────────

@function_tool()
async def search_properties(query: str, max_results: int = 5) -> str:
    """Search rental properties using a natural-language query.

    Args:
        query: Natural language query, e.g. "2BHK in Bangalore under 20k".
        max_results: Maximum number of results to return.
    """
    return await _call_mcp(
        "search_search_properties",
        {"user_query": query, "max_results": max_results},
    )


@function_tool()
async def create_alert(
    email: str,
    location: str = "",
    max_price: int = 0,
    min_bedrooms: int = 0,
) -> str:
    """Create an email alert for new listings matching the given criteria.

    Args:
        email: Destination email for the alert.
        location: City or area filter.
        max_price: Maximum monthly rent (0 = no limit).
        min_bedrooms: Minimum bedrooms (0 = no limit).
    """
    return await _call_mcp(
        "alerts_create_alert",
        {
            "email": email,
            "location": location,
            "max_price": max_price,
            "min_bedrooms": min_bedrooms,
        },
    )


@function_tool()
async def list_alerts(email: str) -> str:
    """List all active alerts for the given email.

    Args:
        email: Email address to look up alerts for.
    """
    return await _call_mcp("alerts_list_alerts", {"email": email})


@function_tool()
async def delete_alert(alert_id: str) -> str:
    """Deactivate an alert by its ID.

    Args:
        alert_id: The alert ID to deactivate.
    """
    return await _call_mcp("alerts_delete_alert", {"alert_id": alert_id})


# ── Public builder ───────────────────────────────────────────────

def build_habitat_tools() -> list:
    """Return the tool list to pass to `Agent(tools=...)`."""
    return [search_properties, create_alert, list_alerts, delete_alert]
