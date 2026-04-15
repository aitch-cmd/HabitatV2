import hashlib
import logging
from typing import Optional, Any
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from fastmcp.server.middleware.middleware import Middleware, MiddlewareContext, CallNext
from fastmcp.tools.base import ToolResult
import mcp.types as mt

# Initialize engines at module load time for efficiency
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

US_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN",
    "US_DRIVER_LICENSE", "US_PASSPORT", "US_BANK_NUMBER",
    "US_ITIN", "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS",
    "LOCATION", "DATE_TIME", "URL", "MEDICAL_LICENSE", "NRP"
]

CONFIDENCE_THRESHOLD = 0.75
logger = logging.getLogger("pii_guardrail")

def _make_token(entity_type: str, value: str) -> str:
    """
    Deterministic token so same value always maps to same placeholder.
    Useful for correlation in audit logs without exposing original.
    """
    hash_val = hashlib.sha256(value.encode()).hexdigest()[:8].upper()
    return f"[{entity_type}_{hash_val}]"

def mask_pii(
    text: str,
    entities: Optional[list[str]] = None,
    score_threshold: float = CONFIDENCE_THRESHOLD
) -> tuple[str, list[dict]]:
    """
    Returns:
        masked_text  — safe to pass to agent
        audit_items  — list of {type, original_value, token, start, end, score}
                       store this in audit log, never pass to agent
    """
    entities_to_check = entities or US_ENTITIES
    
    # Analyze
    results: list[RecognizerResult] = analyzer.analyze(
        text=text,
        language="en",
        entities=entities_to_check,
        score_threshold=score_threshold
    )

    if not results:
        return text, []

    # Build audit items BEFORE anonymization
    audit_items = []
    for r in results:
        original_value = text[r.start:r.end]
        token = _make_token(r.entity_type, original_value)
        audit_items.append({
            "type": r.entity_type,
            "original_value": original_value,
            "token": token,
            "start": r.start,
            "end": r.end,
            "score": round(r.score, 3)
        })

    # Replace with deterministic tokens
    operators = {
        item["type"]: OperatorConfig(
            "replace",
            {"new_value": item["token"]}
        )
        for item in audit_items
    }

    masked = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators
    )

    return masked.text, audit_items

def get_pii_summary(audit_items: list[dict]) -> dict:
    """
    Safe summary — no original values, just stats.
    Fine to log anywhere / return in API responses for debugging.
    """
    from collections import Counter
    counts = Counter(item["type"] for item in audit_items)
    return {
        "pii_detected": len(audit_items) > 0,
        "entity_counts": dict(counts),
        "total": len(audit_items)
    }


class PIIGuardrailMiddleware(Middleware):
    """
    Middleware that intercepts tool responses and automatically masks PII
    before it is returned to the agent.
    """
    async def on_call_tool(self, context: MiddlewareContext[mt.CallToolRequestParams], call_next: CallNext) -> ToolResult:
        # 1. Execute the tool normally
        result: ToolResult = await call_next(context)
        
        # 2. Iterate over the content items returned by the tool
        if hasattr(result, 'content') and isinstance(result.content, list):
            for i, content_block in enumerate(result.content):
                # We only mask Text items
                if hasattr(content_block, 'type') and content_block.type == "text" and getattr(content_block, 'text', None):
                    masked_text, audit_items = mask_pii(content_block.text)
                    
                    if audit_items:
                        # Log Summary directly securely
                        summary = get_pii_summary(audit_items)
                        logger.warning(f"🔒 [PII Guardrail Triggered] Tool '{context.message.name}' response masked: {summary}")
                        
                        # Apply masking to the response sent to the agent
                        result.content[i] = mt.TextContent(
                            type="text",
                            text=masked_text
                        )
                        
        return result
