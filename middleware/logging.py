import logging
import time
from fastmcp.server.middleware import Middleware, MiddlewareContext

logger = logging.getLogger(__name__)

class GuardrailsLoggingMiddleware(Middleware):
    
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        tool_name = context.message.name
        start = time.perf_counter()
        
        logger.info(f"Tool called: {tool_name}")
        
        try:
            result = await call_next(context)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"Tool completed: {tool_name} | {elapsed:.1f}ms")
            return result
            
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(f"Tool failed: {tool_name} | {elapsed:.1f}ms | {type(e).__name__}: {e}")
            raise