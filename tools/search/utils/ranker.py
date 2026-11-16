"""
Stage 3: LambdaMART Custom Tabular + Semantic Reranker.
Replaces generic text-only cross-encoders with a trained LightGBM ranker
combining RAG similarity metrics with structured user preferences.
"""

import pickle
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger("lambdamart_ranker")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "lambdamart_ranker.pkl"


class LambdaMARTReranker:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model = None
        self.features = []
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                    self.model = data["model"]
                    self.features = data.get("features", [])
                logger.info(f"Loaded LambdaMART ranker with features: {self.features}")
            except Exception as e:
                logger.warning(f"Could not load LambdaMART ranker from {model_path}: {e}")
        else:
            logger.warning(f"LambdaMART model not found at {model_path}")

    def rerank(
        self,
        candidates: list[dict],
        budget: float | None = None,
        target_bhk: int | None = None,
        target_locality: str | None = None,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Score and rank candidate listings using the trained LambdaMART model.
        """
        if not candidates:
            return []

        if len(candidates) == 1 or self.model is None:
            return candidates[:top_k]

        ref_budget = (
            budget
            if (budget and budget > 0)
            else float(np.median([c.get("price", 20000) for c in candidates]))
        )
        ref_bhk = (
            target_bhk
            if (target_bhk and target_bhk > 0)
            else int(np.median([c.get("bedrooms") or 2 for c in candidates]))
        )

        feat_matrix = []
        for c in candidates:
            price = float(c.get("price") or ref_budget)
            bedrooms = int(c.get("bedrooms") or ref_bhk)
            area = float(c.get("area_sqft") or (bedrooms * 500))
            contact = str(c.get("contact") or "").lower()
            locality = str(c.get("locality") or "").lower()

            dense_sim = float(c.get("dense_sim") or 0.5)
            lexical_bm25 = float(c.get("lexical_score") or 0.0)
            rrf_score = float(c.get("rrf_score") or 0.01)

            price_ratio = price / max(ref_budget, 1.0)
            bhk_diff = float(abs(bedrooms - ref_bhk))
            area_per_bhk = area / max(bedrooms, 1)
            is_owner = 1.0 if ("owner" in contact or "verified owner" in contact) else 0.0

            loc_match = 0.0
            if target_locality:
                loc_match = 1.0 if target_locality.lower() in locality else 0.0

            # Build feature dictionary
            feat_map = {
                "dense_sim": dense_sim,
                "lexical_bm25": lexical_bm25,
                "rrf_score": rrf_score,
                "price_to_budget_ratio": price_ratio,
                "bhk_diff": bhk_diff,
                "area_per_bhk": area_per_bhk,
                "is_direct_owner": is_owner,
                "locality_match": loc_match,
            }

            if self.features:
                feat = [feat_map.get(f, 0.0) for f in self.features]
            else:
                feat = [
                    dense_sim,
                    lexical_bm25,
                    rrf_score,
                    price_ratio,
                    bhk_diff,
                    area_per_bhk,
                    is_owner,
                ]

            feat_matrix.append(feat)

        X = np.array(feat_matrix)
        scores = self.model.predict(X)

        for c, score in zip(candidates, scores):
            c["lambdamart_score"] = float(score)

        ranked = sorted(candidates, key=lambda x: x["lambdamart_score"], reverse=True)
        return ranked[:top_k]
