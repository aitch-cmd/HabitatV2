import json
import logging
import re
from datetime import date, datetime
from decimal import Decimal
from fastmcp import FastMCP
from tools.search.utils.parser import UserMessageParser
from tools.search.utils.hybrid_search import HybridRAGSearcher
from tools.search.utils.ranker import LambdaMARTReranker
from tools.search.utils.valuation import QuantileRentValuator
from db.connection import postgres_client

logger = logging.getLogger("property_search_server")

search_server = FastMCP("property_search")


class ProductionPropertyPipeline:
    """
    4-Stage Production Rental Retrieval & Valuation Pipeline:
    Stage 1: Structured SQL Pre-filtering (BHK, price with 15% buffer, city).
    Stage 2: Hybrid In-Database RAG (pgvector Cosine Distance + tsvector BM25 RRF) -> Top 25.
    Stage 3: LambdaMART Custom Trained Ranker (Combines dense/lexical/RRF with tabular fit features).
    Stage 4: Quantile Rent Valuation & SHAP Explainer (P10, P50, P90 rent fairness on top 3).
    """

    def __init__(self):
        self.message_parser = UserMessageParser()
        self.hybrid_searcher = HybridRAGSearcher()
        self.ranker = LambdaMARTReranker()
        self.valuator = QuantileRentValuator()

    def parse_query_intent(self, user_message: str) -> dict:
        """Extract structured criteria and parse BHK, budget, and city cues."""
        parsed = self.message_parser.extract(user_message)
        lower_msg = user_message.lower()

        # 1. Budget extraction fallback with regex if LLM parser didn't capture it
        price = parsed.get("price")
        if not price:
            k_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:k|thousand)\b", lower_msg)
            lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:l|lac|lakh)\b", lower_msg)
            plain_match = re.search(r"(?:under|around|budget|below|upto|within|₹)\s*(\d{4,6})\b", lower_msg)

            if k_match:
                price = int(float(k_match.group(1)) * 1000)
            elif lakh_match:
                price = int(float(lakh_match.group(1)) * 100000)
            elif plain_match:
                price = int(plain_match.group(1))

        # 2. Detect BHK patterns from user text (e.g. 1BHK, 2 BHK, 3 bedroom)
        bhk_match = re.search(r"(\d+)\s*(?:bhk|bedroom|rk)", lower_msg)
        target_bhk = int(bhk_match.group(1)) if bhk_match else None

        # 3. Detect common Indian cities
        city = None
        for c in ["delhi", "mumbai", "bangalore", "kolkata", "hyderabad", "chennai"]:
            if c in lower_msg:
                city = c.title()
                break

        return {
            "location": parsed.get("location"),
            "price": price,
            "rag_content": parsed.get("rag_content") or user_message,
            "target_bhk": target_bhk,
            "city": city,
        }

    def search(self, user_message: str, top_k: int = 5) -> list[dict]:
        """Execute the 4-stage pipeline."""
        # 1. Parse query
        intent = self.parse_query_intent(user_message)
        logger.info(f"Query parsed: {intent}")

        # 2. Stage 1 & Stage 2: In-Database Hybrid RAG (pgvector + BM25 RRF)
        candidates = self.hybrid_searcher.search(
            query_text=user_message,
            budget=intent["price"],
            target_bhk=intent["target_bhk"],
            city=intent["city"],
            top_k=25,
        )
        logger.info(f"Stage 2 (Hybrid RAG) retrieved {len(candidates)} candidates.")

        if not candidates:
            return []

        # 3. Stage 3: LambdaMART Reranking
        ranked = self.ranker.rerank(
            candidates=candidates,
            budget=intent["price"],
            target_bhk=intent["target_bhk"],
            target_locality=intent.get("location"),
            top_k=top_k,
        )
        logger.info(f"Stage 3 (LambdaMART) ranked top {len(ranked)} listings.")

        # 4. Stage 4: Quantile Rent Valuation & SHAP Explanation (for top 3 results)
        for idx, listing in enumerate(ranked):
            if idx < 3:
                valuation = self.valuator.evaluate_listing(listing)
                listing["rent_valuation"] = valuation

        return ranked


# Pipeline singleton
_pipeline = ProductionPropertyPipeline()


def _serialize_document(doc: dict) -> dict:
    """Convert PostgreSQL & NumPy objects to JSON-serialisable format."""
    serialized = {}
    for key, value in doc.items():
        if isinstance(value, (datetime, date)):
            serialized[key] = value.isoformat()
        elif isinstance(value, (Decimal, float)):
            serialized[key] = float(value)
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
def search_properties(user_query: str, max_results: int = 5) -> str:
    """
    Search for rental properties using the 4-stage Production Pipeline:
    1. SQL Pre-filtering (with budget buffer)
    2. In-Database Hybrid RAG (pgvector + BM25 tsvector Reciprocal Rank Fusion)
    3. Custom-Trained LambdaMART Tabular + Semantic Reranker
    4. Quantile Rent Valuation (P10, P50, P90) + SHAP Explainability on Top 3 Deals

    Args:
        user_query:  Spoken/natural language query (e.g., 'Sunlit 2BHK in South Delhi Hauz Khas around 25k')
        max_results: Number of results to return (default: 5)

    Returns:
        JSON string with ranked listings and valuation explanations.
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
            "message": f"Found {len(cleaned_results)} properties matching query",
            "count": len(cleaned_results),
            "query": user_query,
            "pipeline_stages": [
                "Stage 1: Structured SQL Pre-filtering",
                "Stage 2: Hybrid In-Database RAG (pgvector + BM25 RRF)",
                "Stage 3: LambdaMART Ranking Model",
                "Stage 4: Quantile Rent Valuation & TreeSHAP Explainer",
            ],
            "results": cleaned_results,
        }, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": f"Search failed: {str(e)}",
            "count": 0,
            "results": [],
        }, indent=2)


@search_server.tool()
def evaluate_rent_deal(listing_id: int) -> str:
    """
    FastMCP Valuation & Deal Assessment Tool:
    Evaluates fair rent range [P10 bargain, P50 fair median, P90 premium] and returns
    actionable SHAP drivers explaining why the property is above, below, or at market price.

    Args:
        listing_id: Database ID of listing to evaluate

    Returns:
        JSON string with fair market range [P10, P50, P90], percentage delta, verdict, and SHAP drivers.
    """
    return evaluate_listing_valuation(listing_id)


@search_server.tool()
def evaluate_listing_valuation(listing_id: int) -> str:
    """
    Evaluate fair rent value (P10, P50, P90) and SHAP drivers for a specific listing by ID.

    Args:
        listing_id: Database ID of listing to evaluate

    Returns:
        JSON string with valuation breakdown and explanation drivers.
    """
    try:
        listing = postgres_client.fetch_one(
            "SELECT * FROM listings WHERE id = %s;",
            (listing_id,),
        )
        if not listing:
            return json.dumps({"error": f"Listing ID {listing_id} not found."})

        val_result = _pipeline.valuator.evaluate_listing(listing)
        return json.dumps({
            "listing_id": listing_id,
            "title": listing.get("title"),
            "address": listing.get("address"),
            "valuation_evaluation": val_result,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
