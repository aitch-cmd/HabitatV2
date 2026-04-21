from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import logging
import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = {"alpha": 0.6, "beta": 0.4}


def _load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.warning('params.yaml not found at %s, using defaults', params_path)
        return {"primary_ranker": _DEFAULT_CONFIG}
    except yaml.YAMLError as e:
        logger.error('YAML error in %s: %s — using defaults', params_path, e)
        return {"primary_ranker": _DEFAULT_CONFIG}
    except Exception as e:
        logger.error('Unexpected error loading %s: %s — using defaults', params_path, e)
        return {"primary_ranker": _DEFAULT_CONFIG}


class BiEncoderReranker:

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        alpha: float = None,
        beta: float = None,
        stage1_top_k: int = 30,
        params_path: str = "params.yaml",
    ):
        config = _load_params(params_path).get("primary_ranker", _DEFAULT_CONFIG)
        self.embedder = SentenceTransformer(model_name, device="cpu")
        self.alpha = alpha if alpha is not None else config.get("alpha", 0.6)
        self.beta = beta if beta is not None else config.get("beta", 0.4)
        self.stage1_top_k = stage1_top_k

    def build_listing_text(self, c: Dict) -> str:
        title        = c.get("title", "")
        description  = c.get("description", "")
        # Support both `bedrooms`/`bathrooms` (PropertyListing schema) and
        # legacy `bedroom`/`bathroom` (older tulire_listings documents).
        bedrooms     = c.get("bedrooms", c.get("bedroom", ""))
        bathrooms    = c.get("bathrooms", c.get("bathroom", ""))
        bed_bath     = f"{bedrooms} bedroom {bathrooms} bathroom"
        amenities    = c.get("amenities") or {}
        appliances   = " ".join(amenities.get("appliances", []))
        utilities    = " ".join(amenities.get("utilities_included", []))
        rental_terms = " ".join(str(v) for v in (c.get("rental_terms") or {}).values())
        return f"{title} {description} {bed_bath} {appliances} {utilities} {rental_terms}"

    def rerank(self, user_query: Dict, candidates: List[Dict]) -> List[Dict]:
        """
        Stage 1 — BiEncoder + BM25 hybrid scoring.
        Narrows large candidate set down to stage1_top_k for cross-encoder.
        """
        rag_content = (user_query.get("rag_content") or "").strip().lower()

        if not rag_content or len(candidates) <= 1:
            return candidates[:self.stage1_top_k]

        # BM25
        corpus      = [self.build_listing_text(c).lower().split() for c in candidates]
        bm25        = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(rag_content.split())

        # Normalize BM25 to [0, 1] so alpha/beta weights make sense
        max_bm25        = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_scores_norm = [s / max_bm25 for s in bm25_scores]

        # BiEncoder — batch encode all docs at once (faster than loop)
        query_emb = self.embedder.encode(rag_content, convert_to_tensor=True)
        doc_texts  = [self.build_listing_text(c) for c in candidates]
        doc_embs   = self.embedder.encode(doc_texts, convert_to_tensor=True, batch_size=32)

        scores = []
        for idx, c in enumerate(candidates):
            emb_score  = util.cos_sim(query_emb, doc_embs[idx]).item()
            bm25_score = bm25_scores_norm[idx]
            final      = self.alpha * emb_score + self.beta * bm25_score
            scores.append((final, c))

        scores.sort(reverse=True, key=lambda x: x[0])
        top = [c for _, c in scores[:self.stage1_top_k]]

        logger.info("BiEncoder stage: %d → %d candidates", len(candidates), len(top))
        return top


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
    reranker = BiEncoderReranker()
    results  = reranker.rerank(user_message, candidates)
    for r in results:
        print(r["title"])