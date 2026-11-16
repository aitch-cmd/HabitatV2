"""
Train custom LambdaMART Ranker (LightGBM LGBMRanker) for Stage 3 Reranking.
Combines RAG similarity signals (dense, lexical, RRF) with tabular fit features
(budget fit, BHK match, area/BHK, direct owner, furnishing match).
"""

import os
import sys
import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from db.connection import postgres_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_lambdamart")

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def generate_training_data(listings: list[dict], n_queries: int = 1200) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Generate synthetic search sessions simulating real-world voice queries
    and label candidate listings with graded relevance (0 to 4).
    """
    np.random.seed(42)
    df = pd.DataFrame(listings)
    cities = df["city"].dropna().unique().tolist()

    all_X = []
    all_y = []
    group_sizes = []

    for q_id in range(n_queries):
        # Sample synthetic user query
        city = np.random.choice(cities)
        target_bhk = int(np.random.choice([1, 2, 3, 4], p=[0.25, 0.45, 0.25, 0.05]))
        city_df = df[df["city"] == city]
        if len(city_df) < 15:
            continue

        median_price = city_df[city_df["bedrooms"] == target_bhk]["price"].median()
        if pd.isna(median_price):
            median_price = city_df["price"].median()
        budget = float(np.random.uniform(0.8, 1.25) * median_price)
        prefer_owner = bool(np.random.rand() > 0.5)

        # Sample candidate listings (top 20 candidates per query)
        cand_indices = city_df.sample(min(20, len(city_df)), replace=False).index

        q_X = []
        q_y = []

        for idx in cand_indices:
            row = df.loc[idx]
            price = float(row["price"])
            bedrooms = int(row["bedrooms"] or 1)
            area = float(row["area_sqft"] or 600)
            contact = str(row["contact"] or "").lower()
            is_owner = 1.0 if "owner" in contact else 0.0

            # Synthetic RAG similarity components
            bhk_diff = abs(bedrooms - target_bhk)
            price_ratio = price / max(budget, 1.0)

            # Simulated semantic & lexical scores
            dense_sim = float(np.clip(1.0 - (bhk_diff * 0.25) - abs(price_ratio - 1.0) * 0.3 + np.random.normal(0, 0.05), 0.0, 1.0))
            lexical_bm25 = float(np.clip(dense_sim * 0.8 + np.random.normal(0, 0.1), 0.0, 1.0))
            rrf_score = float(1.0 / (60 + (1.0 - dense_sim) * 30) + 1.0 / (60 + (1.0 - lexical_bm25) * 30))

            area_per_bhk = area / max(bedrooms, 1)

            # Feature vector: 7 features
            feat = [
                dense_sim,
                lexical_bm25,
                rrf_score,
                price_ratio,
                float(bhk_diff),
                area_per_bhk,
                is_owner,
            ]
            q_X.append(feat)

            # Ground truth relevance grade (0 to 4)
            rel = 0
            if bhk_diff == 0:
                rel += 2
            elif bhk_diff == 1:
                rel += 1

            if price <= budget:
                rel += 2
            elif price <= budget * 1.15:
                rel += 1

            if prefer_owner and is_owner:
                rel = min(4, rel + 1)

            q_y.append(min(4, rel))

        if len(q_X) > 0:
            all_X.extend(q_X)
            all_y.extend(q_y)
            group_sizes.append(len(q_X))

    return np.array(all_X), np.array(all_y), group_sizes


def train_lambdamart():
    logger.info("Loading listings from PostgreSQL for LambdaMART training...")
    rows = postgres_client.fetch_all("""
        SELECT id, price, bedrooms, bathrooms, area_sqft, locality, city, contact, property_type
        FROM listings;
    """)
    logger.info(f"Loaded {len(rows)} listings.")

    logger.info("Generating query sessions and relevance judgments...")
    X, y, groups = generate_training_data(rows, n_queries=1000)
    logger.info(f"Generated {len(X)} instances across {len(groups)} query sessions.")

    # Split train/eval by groups
    split_idx = int(len(groups) * 0.8)
    train_groups = groups[:split_idx]
    eval_groups = groups[split_idx:]
    n_train_samples = sum(train_groups)

    X_train, y_train = X[:n_train_samples], y[:n_train_samples]
    X_eval, y_eval = X[n_train_samples:], y[n_train_samples:]

    logger.info("Training LightGBM LambdaMART ranker...")
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[3, 5, 10],
        n_estimators=100,
        learning_rate=0.08,
        num_leaves=20,
        random_state=42,
        verbose=-1,
    )

    ranker.fit(
        X_train,
        y_train,
        group=train_groups,
        eval_set=[(X_eval, y_eval)],
        eval_group=[eval_groups],
    )

    feature_names = [
        "dense_sim",
        "lexical_bm25",
        "rrf_score",
        "price_to_budget_ratio",
        "bhk_diff",
        "area_per_bhk",
        "is_direct_owner",
    ]

    model_path = MODEL_DIR / "lambdamart_ranker.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": ranker, "features": feature_names}, f)

    logger.info(f"LambdaMART Ranker trained and saved to {model_path}!")


if __name__ == "__main__":
    train_lambdamart()
