"""
Train Quantile Rent Valuation Models (P10, P50, P90) with LightGBM and SHAP explainer.
Uses the 18,746 rental listings from PostgreSQL to learn pricing drivers.
"""

import os
import sys
import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from db.connection import postgres_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_valuation")

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def extract_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean and featurize listings for quantile regression."""
    df = df.copy()

    # Filter out corrupted / extreme outlier prices
    df = df[(df["price"] >= 2000) & (df["price"] <= 1000000) & (df["area_sqft"] >= 100)].copy()

    # Numeric features
    df["bedrooms"] = df["bedrooms"].fillna(1).astype(int)
    df["bathrooms"] = df["bathrooms"].fillna(df["bedrooms"]).astype(int)
    df["area_sqft"] = df["area_sqft"].fillna(df["bedrooms"] * 450).astype(int)

    # Categorical features
    df["city"] = df["city"].fillna("Other").astype(str).str.strip()
    df["furnishing_status"] = df["furnishing_status"].fillna("Unfurnished").astype(str).str.strip()
    df["property_type"] = df["property_type"].fillna("Apartment").astype(str).str.strip()

    # Locality frequency encoding
    locality_counts = df["locality"].value_counts()
    df["locality_freq"] = df["locality"].map(locality_counts).fillna(1)

    # Target
    y = df["price"].astype(float)

    feature_cols = [
        "bedrooms",
        "bathrooms",
        "area_sqft",
        "locality_freq",
        "city",
        "furnishing_status",
        "property_type",
    ]

    X = df[feature_cols].copy()
    for col in ["city", "furnishing_status", "property_type"]:
        X[col] = X[col].astype("category")

    return X, y


def train_and_save_models():
    logger.info("Loading listings data from PostgreSQL...")
    rows = postgres_client.fetch_all("""
        SELECT price, bedrooms, bathrooms, area_sqft, locality, city, furnishing_status, property_type
        FROM listings;
    """)

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} listings.")

    X, y = extract_features(df)
    logger.info(f"Cleaned training dataset: {len(X)} records across {len(X.columns)} features.")

    quantiles = [0.10, 0.50, 0.90]
    models = {}

    for q in quantiles:
        logger.info(f"Training LightGBM Quantile Regressor for alpha={q}...")
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=150,
            learning_rate=0.08,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        model.fit(X, y)
        models[f"p{int(q*100)}"] = model

    # Fit TreeSHAP on P50 model
    logger.info("Building TreeSHAP Explainer for P50 (Fair Market Value)...")
    p50_model = models["p50"]
    # Use sample background for fast SHAP evaluation
    explainer = shap.TreeExplainer(p50_model)

    artifacts = {
        "models": models,
        "explainer": explainer,
        "feature_names": list(X.columns),
        "categorical_features": ["city", "furnishing_status", "property_type"],
    }

    output_path = MODEL_DIR / "valuation_models.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(artifacts, f)

    logger.info(f"Successfully saved quantile models & explainer to {output_path}")

    # Test sample evaluation
    sample = X.iloc[[0]]
    p10 = models["p10"].predict(sample)[0]
    p50 = models["p50"].predict(sample)[0]
    p90 = models["p90"].predict(sample)[0]
    logger.info(f"Sample test prediction -> P10: ₹{p10:,.0f} | P50: ₹{p50:,.0f} | P90: ₹{p90:,.0f}")


if __name__ == "__main__":
    train_and_save_models()
