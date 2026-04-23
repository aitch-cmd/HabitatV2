"""
MCP Bridge — dynamic tool discovery.

Design A: at agent startup, open an MCP client, call `tools/list`, and
synthesize one LiveKit `function_tool` per discovered tool. The server's
schema is the single source of truth for name, arguments, and types.
"""

import inspect
import json
import logging
from typing import Any

from fastmcp import Client
from livekit.agents import function_tool

from config import MCP_JWT, MCP_SERVER_URL, MCP_TIMEOUT

logger = logging.getLogger(__name__)

FALLBACK = "Sorry, I couldn't reach the property database right now."


# ── Transport ────────────────────────────────────────────────────

_CLIENT_KWARGS: dict = {"timeout": MCP_TIMEOUT}
if MCP_JWT:
    _CLIENT_KWARGS["auth"] = MCP_JWT


async def _call_mcp(tool_name: str, args: dict) -> str:
    """Open a short-lived MCP client, call `tool_name`, stringify the result."""
    try:
        async with Client(MCP_SERVER_URL, **_CLIENT_KWARGS) as client:
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


# ── JSON Schema → Python types ───────────────────────────────────

_JSON_TO_PY: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

_DEFAULTS: dict[type, Any] = {
    str: "",
    int: 0,
    float: 0.0,
    bool: False,
    list: [],
    dict: {},
}

# Python keywords that can't be used as parameter names — skip tools that
# try to. In practice our server never ships these, but the check is cheap.
_RESERVED = frozenset(
    {"class", "def", "return", "if", "else", "for", "while", "import",
     "from", "as", "lambda", "global", "nonlocal", "with", "try", "except",
     "finally", "raise", "yield", "pass", "break", "continue", "and", "or",
     "not", "is", "in", "True", "False", "None"}
)


def _py_type(schema: dict) -> type:
    """Map a JSON-schema fragment to a Python type for signature building."""
    t = schema.get("type")
    if isinstance(t, list):
        # JSON Schema allows unions like ["string", "null"] — pick the
        # first non-null type.
        non_null = [x for x in t if x != "null"]
        t = non_null[0] if non_null else "string"
    return _JSON_TO_PY.get(t, str)


# ── Dynamic function synthesis ───────────────────────────────────

def _synthesize_tool(name: str, description: str, schema: dict):
    """
    Build an async function whose signature exactly matches `schema`, so
    LiveKit's `function_tool` decorator can introspect it and emit a
    JSON-schema tool definition for Claude.

    We use `exec` because LiveKit's schema generation reads `__code__` /
    `inspect.signature` — setting `__signature__` alone isn't enough across
    versions. The `exec` source is fully controlled (no user input reaches
    it), so there's no injection surface.
    """
    properties: dict = (schema or {}).get("properties", {}) or {}
    required: set[str] = set((schema or {}).get("required", []) or [])

    for pname in properties:
        if not pname.isidentifier() or pname in _RESERVED:
            raise ValueError(f"tool {name!r} has unusable param name {pname!r}")

    # Required params first (no default), optional params after (with default)
    ordered = sorted(properties.items(), key=lambda kv: kv[0] not in required)

    sig_parts: list[str] = []
    annotations: dict[str, Any] = {}
    defaults: list[Any] = []

    for pname, pschema in ordered:
        py_t = _py_type(pschema)
        annotations[pname] = py_t
        if pname in required:
            sig_parts.append(pname)
        else:
            sig_parts.append(pname)
            defaults.append(_DEFAULTS.get(py_t))

    # Docstring that the LLM will see, in Google-style so LiveKit picks up
    # per-argument descriptions from the server.
    doc_lines = [description.strip() or f"Invoke the MCP tool {name!r}.", ""]
    if properties:
        doc_lines.append("Args:")
        for pname, pschema in ordered:
            arg_desc = (pschema.get("description") or "").strip()
            doc_lines.append(f"    {pname}: {arg_desc}".rstrip())
    docstring = "\n".join(doc_lines)

    # Dict-literal body that forwards every named arg to `_call_mcp`.
    arg_pairs = ", ".join(f'"{p}": {p}' for p in properties)
    source = (
        f"async def _dyn({', '.join(sig_parts)}):\n"
        f"    return await _call_mcp(_MCP_NAME, {{{arg_pairs}}})\n"
    )

    ns: dict[str, Any] = {"_call_mcp": _call_mcp, "_MCP_NAME": name}
    exec(source, ns)  # noqa: S102 — source is built from a trusted schema
    fn = ns["_dyn"]

    fn.__name__ = name
    fn.__qualname__ = name
    fn.__doc__ = docstring
    fn.__annotations__ = {**annotations, "return": str}
    if defaults:
        fn.__defaults__ = tuple(defaults)

    return fn


# ── Public API ───────────────────────────────────────────────────

async def build_habitat_tools() -> list:
    """
    Discover MCP tools via `tools/list` and return them as LiveKit
    function_tool callables, ready to pass to `Agent(tools=...)`.

    Async because discovery is a real MCP roundtrip — must be awaited
    from the LiveKit entrypoint before the agent is constructed.
    """
    try:
        async with Client(MCP_SERVER_URL, **_CLIENT_KWARGS) as client:
            tools = await client.list_tools()
    except Exception:
        logger.exception(
            "MCP tool discovery failed — MCP tools will be unavailable this session"
        )
        return []

    wrapped: list = []
    for tool in tools:
        schema = (
            getattr(tool, "inputSchema", None)
            or getattr(tool, "input_schema", None)
            or {}
        )
        try:
            raw = _synthesize_tool(
                name=tool.name,
                description=tool.description or "",
                schema=schema,
            )
            wrapped.append(function_tool()(raw))
            logger.info("  registered MCP tool: %s(%s)",
                        tool.name,
                        ", ".join(inspect.signature(raw).parameters))
        except Exception:
            logger.exception("Skipping tool %s — failed to synthesize wrapper", tool.name)

    logger.info("MCP tool discovery complete: %d tool(s) available", len(wrapped))
    return wrapped
