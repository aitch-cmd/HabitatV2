from typing import Any
from collections import defaultdict
from collections.abc import Callable

from fastmcp.server.middleware.middleware import Middleware, MiddlewareContext, CallNext
from fastmcp.server.middleware.rate_limiting import (
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    RateLimitError
)
import mcp.types as mt

class GuardrailRateLimitingMiddleware(Middleware):
    """
    Middleware that implements multi-layered guardrails for tools calls:
    1. Global Session Limit: Sustained usage bucket tracking overall session calls.
    2. Per-Tool Session Limit: Token bucket ensuring single tools aren't spammed.
    3. Strict 'alert' Limit: Precise sliding window limiter that drops excessive 'alert' tool usage.
    """
    
    def __init__(self, 
                 session_req_per_sec: float = 0.5, session_burst: int = 30,
                 tool_req_per_sec: float = 2.0, tool_burst: int = 10,
                 alert_max_req: int = 2, alert_window_sec: int = 300):
        
        # 1. Per-session Token Bucket (general usage across all tools)
        self.session_limiters = defaultdict(lambda: TokenBucketRateLimiter(capacity=session_burst, refill_rate=session_req_per_sec))
        
        # 2. Per-tool standard Token Bucket (helps prevent a single tool from eating up the whole session)
        self.tool_limiters = defaultdict(lambda: TokenBucketRateLimiter(capacity=tool_burst, refill_rate=tool_req_per_sec))
        
        # 3. Strict sliding window specifically for the "alert" tool
        self.alert_limiters = defaultdict(lambda: SlidingWindowRateLimiter(max_requests=alert_max_req, window_seconds=alert_window_sec))

    async def on_call_tool(self, context: MiddlewareContext[mt.CallToolRequestParams], call_next: CallNext) -> Any:
        # Extract metadata
        session_id = getattr(context.fastmcp_context, 'session_id', 'default_session') if context.fastmcp_context else 'default_session'
        tool_name = context.message.name

        # Stage 1: Check strict Alert Limit first
        if tool_name == "alert":
            alert_limiter = self.alert_limiters[session_id]
            if not await alert_limiter.is_allowed():
                raise RateLimitError(f"Strict rate limit exceeded for 'alert' tool.")

        # Stage 2: Check general Limit specific to this tool
        tool_limiter_key = f"{session_id}_{tool_name}"
        if not await self.tool_limiters[tool_limiter_key].consume():
            raise RateLimitError(f"Rate limit exceeded for tool: {tool_name}")

        # Stage 3: Check global Session Limit
        if not await self.session_limiters[session_id].consume():
            raise RateLimitError(f"Global session rate limit exceeded.")

        # Proceed to next middleware or tool execution if all checks pass
        return await call_next(context)
