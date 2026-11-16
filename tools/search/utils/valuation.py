"""
Stage 4: Quantile Rent Valuation & SHAP Explainer.
Predicts P10, P50, and P90 rent benchmarks and generates explainable pricing insights.
"""

import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger("rent_valuation")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "valuation_models.pkl"


class QuantileRentValuator:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.models = None
        self.explainer = None
        self.feature_names = []
        self.categorical_features = []

        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    data = pickle.load(f)
                    self.models = data["models"]
                    self.explainer = data["explainer"]
                    self.feature_names = data["feature_names"]
                    self.categorical_features = data["categorical_features"]
                logger.info("Quantile Rent Valuation & SHAP explainer loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load valuation models: {e}")
        else:
            logger.warning(f"Valuation model file not found at {model_path}")

    def evaluate_listing(self, listing: dict) -> dict:
        """
        Evaluate listing rent against P10, P50, P90 market quantiles and compute SHAP drivers.
        """
        actual_price = float(listing.get("price") or 0.0)

        if not self.models:
            return {
                "actual_rent": actual_price,
                "fair_market_rent_p50": actual_price,
                "bargain_rent_p10": actual_price * 0.85,
                "premium_rent_p90": actual_price * 1.25,
                "valuation_verdict": "Fair Market Price",
                "savings_or_premium": 0.0,
                "explanation_drivers": ["Market valuation model unavailable."],
            }

        # Build single-row DataFrame matching training feature schema
        bedrooms = int(listing.get("bedrooms") or 1)
        bathrooms = int(listing.get("bathrooms") or bedrooms)
        area_sqft = int(listing.get("area_sqft") or (bedrooms * 500))
        locality = str(listing.get("locality") or "")
        city = str(listing.get("city") or "Delhi").strip()
        furnishing = str(listing.get("furnishing_status") or "Unfurnished").strip()
        ptype = str(listing.get("property_type") or "Apartment").strip()

        row = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "area_sqft": area_sqft,
            "locality_freq": 10,  # default median frequency
            "city": city,
            "furnishing_status": furnishing,
            "property_type": ptype,
        }

        df = pd.DataFrame([row])
        for col in self.categorical_features:
            df[col] = df[col].astype("category")

        # Predict quantiles
        p10 = float(self.models["p10"].predict(df)[0])
        p50 = float(self.models["p50"].predict(df)[0])
        p90 = float(self.models["p90"].predict(df)[0])

        diff = actual_price - p50
        pct_diff = (diff / p50) * 100 if p50 > 0 else 0.0

        if pct_diff <= -8.0:
            verdict = "Great Deal (Underpriced)"
            savings = abs(diff)
        elif pct_diff >= 10.0:
            verdict = "Above Market Value"
            savings = -diff
        else:
            verdict = "Fair Market Price"
            savings = 0.0

        # SHAP explanation for P50
        drivers = []
        try:
            shap_values = self.explainer(df)
            values = shap_values.values[0]
            # Pair feature names with their marginal impact in Rupees
            ranked_impacts = sorted(
                zip(self.feature_names, values, [row[f] for f in self.feature_names]),
                key=lambda x: abs(x[1]),
                reverse=True,
            )

            for feat_name, impact, val in ranked_impacts[:3]:
                sign = "+" if impact >= 0 else "-"
                formatted_feat = feat_name.replace("_", " ").title()
                drivers.append(
                    f"{formatted_feat} ({val}): {sign}₹{abs(impact):,.0f} impact on fair rent"
                )
        except Exception as e:
            logger.debug(f"SHAP explanation computation error: {e}")
            drivers.append(f"Fair market median computed at ₹{p50:,.0f}/month.")

        return {
            "actual_rent": actual_price,
            "fair_market_rent_p50": round(p50),
            "bargain_rent_p10": round(p10),
            "premium_rent_p90": round(p90),
            "valuation_verdict": verdict,
            "savings_or_premium": round(savings),
            "price_delta_percent": round(pct_diff, 1),
            "explanation_drivers": drivers,
        }
