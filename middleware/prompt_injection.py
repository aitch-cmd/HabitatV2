# guardrails/injection_middleware.py

import re
import logging
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.exceptions import ToolError

logger = logging.getLogger(__name__)

INJECTION_PATTERNS = {

    "instruction_override": [
        r"\bignore\b.{0,30}\b(previous|above|prior|all)\b.{0,30}\b(instructions?|prompt|rules?|context)\b",
        r"\bdisregard\b.{0,30}\b(instructions?|prompt|rules?|guidelines?)\b",
        r"\bforget\b.{0,30}\b(instructions?|everything|above|context)\b",
        r"\boverride\b.{0,30}\b(instructions?|prompt|rules?|settings?)\b",
        r"\byou (are|must|should|will) now\b",
        r"\bnew (instructions?|rules?|directives?|prompt)\b",
    ],

    "role_hijack": [
        r"\bact as\b.{0,20}\b(an?|the)\b",
        r"\bpretend (you are|to be|you're)\b",
        r"\byou are now\b",
        r"\bswitch (to|into) (mode|role|persona)\b",
        r"\bdo anything now\b",            # DAN
        r"\bjailbreak\b",
        r"\bdan mode\b",
        r"\bdevmode\b",
        r"\bgrandma (trick|exploit|mode)\b",
    ],

    "system_probe": [
        r"\b(print|reveal|show|output|repeat|display)\b.{0,30}\b(system prompt|instructions?|context|your prompt)\b",
        r"\bwhat (are|were) your instructions\b",
        r"\bwhat is (in |your )?(the )?system prompt\b",
        r"\binitial prompt\b",
        r"\bhidden (instructions?|rules?|prompt)\b",
    ],

    "delimiter_injection": [
        r"<\|?(system|user|assistant|im_start|im_end)\|?>",   # ChatML / special tokens
        r"\[INST\]",                                           # Llama instruct tokens
        r"<<SYS>>",
        r"###\s*(instruction|system|human|assistant)",
        r"---+\s*(system|instruction)",
    ],

    "indirect_payload": [
        r"\bexecute\b.{0,20}\b(this|the following|below)\b",
        r"\brun\b.{0,20}\b(this command|the following)\b",
        r"\benable\b.{0,20}\b(developer|admin|debug|god)\b.{0,10}mode\b",
        r"\bunrestricted\b.{0,20}\bmode\b",
        r"\bno (restrictions?|filters?|limits?|guardrails?)\b",
    ],
}

# Flatten with category tracking
_COMPILED: list[tuple[str, re.Pattern]] = [
    (category, re.compile(pattern, re.IGNORECASE | re.DOTALL))
    for category, patterns in INJECTION_PATTERNS.items()
    for pattern in patterns
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_strings(arguments: dict) -> list[tuple[str, str]]:
    """
    Recursively extract (field_path, value) for all string values in args dict.
    Handles nested dicts and lists.
    """
    results = []

    def _walk(obj, path: str):
        if isinstance(obj, str):
            results.append((path, obj))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, f"{path}[{i}]")

    _walk(arguments, "")
    return results


def _scan(text: str) -> tuple[bool, str | None, str | None]:
    """
    Returns (is_injection, category, matched_pattern_string)
    """
    for category, pattern in _COMPILED:
        if pattern.search(text):
            return True, category, pattern.pattern
    return False, None, None


# ── Middleware ────────────────────────────────────────────────────────────────

class PromptInjectionMiddleware(Middleware):
    """
    Scans all string arguments of every tool call for prompt injection patterns.
    Raises ToolError and logs the attempt if injection is detected.
    """

    def __init__(
        self,
        blocked_tools: set[str] | None = None,   # only scan these (None = all)
        skipped_tools: set[str] | None = None,    # never scan these
        raise_on_detection: bool = True,           # False = log only, let through
    ):
        self.blocked_tools = blocked_tools
        self.skipped_tools = skipped_tools or set()
        self.raise_on_detection = raise_on_detection

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        tool_name = context.message.name
        arguments  = context.message.arguments or {}

        # Scope check
        if tool_name in self.skipped_tools:
            return await call_next(context)
        if self.blocked_tools and tool_name not in self.blocked_tools:
            return await call_next(context)

        # Scan all string fields
        for field_path, value in _extract_strings(arguments):
            is_injection, category, pattern = _scan(value)

            if is_injection:
                logger.warning(
                    "Prompt injection detected",
                    extra={
                        "tool": tool_name,
                        "field": field_path,
                        "category": category,
                        "pattern": pattern,
                        # Do NOT log value — it may contain the malicious payload
                    }
                )

                if self.raise_on_detection:
                    raise ToolError(
                        f"Input rejected: potential prompt injection in field '{field_path}'."
                    )

        return await call_next(context)