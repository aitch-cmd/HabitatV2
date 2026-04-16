import asyncio
from typing import Any
from mcp.server.fastmcp import FastMCP
from tools.search.utils.parser import UserMessageParser
from tools.search.utils.mgdb_filter import MongoDBFilter
from tools.search.utils.bi_encoder import BiEncoderReranker
from tools.search.utils.cross_encoder import CrossEncoderReranker
import json

search_server = FastMCP("property_search")

BI_ENCODER_THRESHOLD = 50  # Skip bi-encoder and go straight to cross-encoder below this


class PropertySearchPipeline:
    """
    Unified pipeline for property search:
    1. Parse user message  (extract location, price, preferences)
    2. MongoDB filter      (hard constraints: location, price)
    3a. > 50 candidates  → BiEncoder (BM25 + semantic) → top 30 → CrossEncoder → top K
    3b. ≤ 50 candidates  → CrossEncoder directly         → top K
    """

    def __init__(self):
        self.message_parser = UserMessageParser()
        self.mongo_filter   = MongoDBFilter(collection_name="tulire_listings")
        self.bi_encoder     = BiEncoderReranker()
        self.cross_encoder  = CrossEncoderReranker()

    def search(self, user_message: str, top_k: int = 5) -> list[dict]:
        """
        Execute the full search pipeline.

        Args:
            user_message: Natural language query from user
            top_k:        Number of top results to return (default: 5)

        Returns:
            List of ranked property listings
        """
        # Step 1: Parse user message into structured fields
        parsed_message = self.message_parser.extract(user_message)
        print(f" Parsed: {parsed_message}")

        # Step 2: MongoDB hard-filter (location, price)
        mongo_query = self.mongo_filter.build_query(parsed_message)
        print(f" MongoDB Query: {mongo_query}")

        candidates = list(self.mongo_filter.collection.find(mongo_query))
        print(f" Retrieved {len(candidates)} candidates from MongoDB")

        if not candidates:
            return []

        # Normalise rag_content once for both rerankers
        rag_content = (parsed_message.get("rag_content") or "").strip()
        has_preferences = bool(rag_content) and len(candidates) > 1

        # Step 3: Reranking — branch on candidate count
        if not has_preferences:
            # No semantic preferences at all — return mongo order, capped at top_k
            print("⚠️  No preferences in query — skipping reranking")
            return candidates[:top_k]

        if len(candidates) > BI_ENCODER_THRESHOLD:
            # Full two-stage pipeline
            print(f" Running BiEncoder stage ({len(candidates)} candidates > {BI_ENCODER_THRESHOLD})")
            stage1 = self.bi_encoder.rerank(
                user_query=parsed_message,
                candidates=candidates,
            )
            print(f"   BiEncoder → {len(stage1)} candidates passed to CrossEncoder")
        else:
            # Small result set — skip bi-encoder, go straight to cross-encoder
            print(f" Skipping BiEncoder ({len(candidates)} candidates ≤ {BI_ENCODER_THRESHOLD}), going straight to CrossEncoder")
            stage1 = candidates

        ranked = self.cross_encoder.rerank(
            user_query=parsed_message,
            candidates=stage1,
            top_k=top_k,
        )
        print(f" CrossEncoder → {len(ranked)} final results")
        return ranked


# Singleton — initialised once at import time
_pipeline = PropertySearchPipeline()


def _serialize_document(doc: dict) -> dict:
    """Convert MongoDB document to JSON-serialisable format (ObjectId, datetime)."""
    from bson import ObjectId
    from datetime import datetime

    serialized = {}
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            serialized[key] = str(value)
        elif isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, dict):
            serialized[key] = _serialize_document(value)
        elif isinstance(value, list):
            serialized[key] = [
                _serialize_document(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            serialized[key] = value
    return serialized


@search_server.tool()
def search_properties(user_query: str, max_results: int = 10) -> str:
    """
    Search for rental properties based on a natural language query.

    Pipeline:
    1. Message parsing  — extracts location, price, preferences
    2. MongoDB filter   — hard constraints (location, price range)
    3. Reranking        — BiEncoder + CrossEncoder (or CrossEncoder only if < 50 results)

    Args:
        user_query:  Natural language search query
                     e.g. "2BHK furnished near MIT under 25k"
        max_results: Maximum number of results to return (default: 10)

    Returns:
        JSON string with ranked property listings.

    Examples:
        - "Find 2BHK apartments in Bangalore under 20000"
        - "3BHK furnished flat with parking near MG Road"
        - "Studio apartment pet friendly under 15k"
    """
    try:
        results = _pipeline.search(user_query, top_k=max_results)

        if not results:
            return json.dumps({
                "status": "success",
                "message": "No properties found matching your criteria",
                "count": 0,
                "results": [],
            }, indent=2)

        cleaned_results = [_serialize_document(listing) for listing in results]

        return json.dumps({
            "status": "success",
            "message": f"Found {len(cleaned_results)} matching properties",
            "count": len(cleaned_results),
            "query": user_query,
            "results": cleaned_results,
        }, indent=2, ensure_ascii=False)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({
            "status": "error",
            "message": f"Search failed: {str(e)}",
            "count": 0,
            "results": [],
        }, indent=2)