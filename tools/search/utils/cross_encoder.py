from typing import List, Dict
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.cross_encoder = CrossEncoder(model_name, device="cpu")

    def build_listing_text(self, c: Dict) -> str:
        title        = c.get("title", "")
        description  = c.get("description", "")
        bed_bath     = f"{c.get('bedroom', '')} bedroom {c.get('bathroom', '')} bathroom"
        appliances   = " ".join(c.get("amenities", {}).get("appliances", []))
        utilities    = " ".join(c.get("amenities", {}).get("utilities_included", []))
        rental_terms = " ".join([str(v) for v in c.get("rental_terms", {}).values()])
        return f"{title} {description} {bed_bath} {appliances} {utilities} {rental_terms}"

    def rerank(self, user_query: Dict, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Stage 2 — Cross-encoder precise reranking.
        Receives top 30 from BiEncoder, returns final top_k.
        """
        rag_content = (user_query.get("rag_content") or "").strip().lower()

        if not rag_content or len(candidates) <= 1:
            return candidates[:top_k]

        # Build query-document pairs
        pairs = [
            [rag_content, self.build_listing_text(doc)]
            for doc in candidates
        ]

        # Cross-encoder scores all pairs together
        cross_scores = self.cross_encoder.predict(pairs)

        reranked = sorted(
            zip(cross_scores, candidates),
            reverse=True,
            key=lambda x: x[0]
        )

        final = [doc for _, doc in reranked][:top_k]
        logger.info("CrossEncoder stage: %d → %d candidates", len(candidates), len(final))
        return final


if __name__ == "__main__":
    user_message = {
        "location": "Greater Noida",
        "price": "",
        "rag_content": "pet friendly, balcony, furnished"
    }
    candidates = [
        {"title": "2BHK Apartment", "description": "Spacious, furnished, with balcony",
         "amenities": {"appliances": [], "utilities_included": []},
         "bedroom": 2, "bathroom": 1, "rental_terms": {}},
        {"title": "Studio Flat", "description": "Affordable unit",
         "amenities": {"appliances": [], "utilities_included": []},
         "bedroom": 1, "bathroom": 1, "rental_terms": {}},
        {"title": "3BHK Luxury", "description": "Pet friendly furnished apartment",
         "amenities": {"appliances": [], "utilities_included": []},
         "bedroom": 3, "bathroom": 2, "rental_terms": {}},
    ]
    reranker = CrossEncoderReranker()
    results  = reranker.rerank(user_message, candidates, top_k=2)
    for r in results:
        print(r["title"])