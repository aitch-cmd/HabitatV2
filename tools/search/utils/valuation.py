"""
Stage 4: Quantile Rent Valuation & SHAP Explainer.
Predicts P10, P50, and P90 rent benchmarks and generates explainable pricing insights.
Supports both log-transformed quantile regressors and real-time TreeSHAP attributions.
"""

import os
import re
import pickle
import logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger("rent_valuation")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PRIMARY_MODEL_PATH = BASE_DIR / "models" / "quantile_rent_regressor.pkl"
FALLBACK_MODEL_PATH = BASE_DIR / "models" / "valuation_models.pkl"


def parse_clean_deposit(val) -> float:
    """Parse security deposit into float INR."""
    if not val or pd.isna(val):
        return 0.0
    s = str(val).strip().replace(",", "")
    if s in ["No", "No deposit", "nan", "None", "", "0", "unknown"]:
        return 0.0
    if s.endswith("L"):
        try:
            return float(s[:-1].strip()) * 100000.0
        except ValueError:
            pass
    elif s.endswith("Cr"):
        try:
            return float(s[:-2].strip()) * 10000000.0
        except ValueError:
            pass
    digits = re.sub(r"[^\d.]", "", s)
    try:
        return float(digits) if digits else 0.0
    except ValueError:
        return 0.0


class QuantileRentValuator:
    def __init__(self, model_path: Path | None = None):
        self.models = None
        self.explainer = None
        self.feature_names = []
        self.categorical_features = []
        self.categories = {}
        self.is_log_target = False

        target_path = model_path or PRIMARY_MODEL_PATH
        if not target_path.exists():
            target_path = FALLBACK_MODEL_PATH

        if target_path.exists():
            try:
                # Load with joblib or pickle
                try:
                    data = joblib.load(target_path)
                except Exception:
                    with open(target_path, "rb") as f:
                        data = pickle.load(f)

                self.models = data.get("models", {})
                self.explainer = data.get("explainer")
                self.feature_names = data.get("features") or data.get("feature_names", [])
                self.categorical_features = data.get("cat_cols") or data.get("categorical_features", [])
                self.categories = data.get("categories", {})
                self.is_log_target = data.get("is_log_target", False)

                logger.info(f"Loaded Quantile Rent Valuator from {target_path} (log_target={self.is_log_target}, features={len(self.feature_names)})")
            except Exception as e:
                logger.warning(f"Failed to load valuation models from {target_path}: {e}")
        else:
            logger.warning(f"Valuation model file not found at {target_path}")

    def evaluate_listing(self, listing: dict) -> dict:
        """
        Evaluate listing rent against P10, P50, P90 market quantiles and compute SHAP drivers.
        """
        actual_price = float(listing.get("price") or listing.get("rent_price") or 0.0)

        if not self.models:
            return {
                "actual_rent": actual_price,
                "fair_market_rent_p50": actual_price,
                "bargain_rent_p10": actual_price * 0.85,
                "premium_rent_p90": actual_price * 1.25,
                "valuation_verdict": "Fair Market Price",
                "savings_or_premium": 0.0,
                "price_delta_percent": 0.0,
                "explanation_drivers": ["Market valuation model unavailable."],
            }

        # Identify which model keys are available (0.10 or "p10", etc.)
        m10 = self.models.get(0.10) or self.models.get("p10")
        m50 = self.models.get(0.50) or self.models.get("p50")
        m90 = self.models.get(0.90) or self.models.get("p90")

        if not m50:
            return {
                "actual_rent": actual_price,
                "fair_market_rent_p50": actual_price,
                "bargain_rent_p10": actual_price * 0.85,
                "premium_rent_p90": actual_price * 1.25,
                "valuation_verdict": "Fair Market Price",
                "savings_or_premium": 0.0,
                "price_delta_percent": 0.0,
                "explanation_drivers": ["P50 model not found in artifact."],
            }

        # Build feature DataFrame
        bhk = int(listing.get("bedrooms") or listing.get("bhk") or 1)
        bathrooms = int(listing.get("bathrooms") or bhk)
        area_sqft = float(listing.get("area_sqft") or (bhk * 500))
        locality = str(listing.get("locality") or listing.get("location") or "Other").strip()
        city = str(listing.get("city") or "Delhi").strip()
        furnishing = str(listing.get("furnishing_status") or listing.get("status") or "Semi-Furnished").strip()
        ptype = str(listing.get("property_type") or "Apartment").strip()
        facing = str(listing.get("facing_direction") or "Not specified").strip()
        contact = str(listing.get("contact") or listing.get("seller_type") or "Agent").strip()
        seller_type = contact.split(" (")[0].strip()
        deposit = parse_clean_deposit(listing.get("security_deposit"))

        # Check whether trained model uses 10-feature schema or legacy schema
        if "sqft_per_bhk" in self.feature_names or "location" in self.feature_names:
            sqft_per_bhk = area_sqft / max(bhk, 1)
            row = {
                "bhk": bhk,
                "area_sqft": area_sqft,
                "bathrooms": bathrooms,
                "security_deposit": deposit,
                "sqft_per_bhk": sqft_per_bhk,
                "location": locality,
                "property_type": ptype,
                "status": furnishing,
                "facing_direction": facing,
                "seller_type": seller_type,
            }
        else:
            # Legacy 7-feature fallback
            row = {
                "bedrooms": bhk,
                "bathrooms": bathrooms,
                "area_sqft": int(area_sqft),
                "locality_freq": 10,
                "city": city,
                "furnishing_status": furnishing,
                "property_type": ptype,
            }

        df = pd.DataFrame([row])
        # Re-order and cast categoricals according to trained features
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].astype("category")

        # Select only expected features
        expected_cols = [c for c in self.feature_names if c in df.columns]
        df_feat = df[expected_cols] if expected_cols else df

        # Quantile predictions
        try:
            pred10 = float(m10.predict(df_feat)[0]) if m10 else actual_price * 0.85
            pred50 = float(m50.predict(df_feat)[0])
            pred90 = float(m90.predict(df_feat)[0]) if m90 else actual_price * 1.25

            if self.is_log_target:
                p10 = float(np.expm1(pred10))
                p50 = float(np.expm1(pred50))
                p90 = float(np.expm1(pred90))
            else:
                p10, p50, p90 = pred10, pred50, pred90

        except Exception as e:
            logger.warning(f"Prediction error in QuantileRentValuator: {e}")
            p10 = actual_price * 0.85
            p50 = actual_price
            p90 = actual_price * 1.25

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

        # Compute SHAP explanation drivers
        drivers = []
        if self.explainer is not None:
            try:
                shap_res = self.explainer(df_feat)
                vals = shap_res.values[0]
                feature_list = list(df_feat.columns)

                # Rank by absolute impact
                impacts = []
                for fname, val in zip(feature_list, vals):
                    abs_val = abs(val)
                    if self.is_log_target:
                        # Log impact converted to rupee delta: p50 * (exp(val) - 1)
                        inr_impact = p50 * (np.exp(val) - 1.0)
                    else:
                        inr_impact = val
                    impacts.append((fname, inr_impact, abs_val, row.get(fname, "")))

                impacts.sort(key=lambda x: x[2], reverse=True)

                for fname, inr_val, _, raw_v in impacts[:3]:
                    sign = "+" if inr_val >= 0 else "-"
                    label = fname.replace("_", " ").title()
                    drivers.append(
                        f"{label} ({raw_v}): {sign}₹{abs(inr_val):,.0f} impact on fair rent"
                    )
            except Exception as e:
                logger.debug(f"SHAP explanation computation error: {e}")
                drivers.append(f"Fair market median predicted at ₹{p50:,.0f}/month.")

        if not drivers:
            drivers.append(f"Fair market median predicted at ₹{p50:,.0f}/month.")

        return {
            "actual_rent": round(actual_price),
            "fair_market_rent_p50": round(p50),
            "bargain_rent_p10": round(p10),
            "premium_rent_p90": round(p90),
            "valuation_verdict": verdict,
            "savings_or_premium": round(savings),
            "price_delta_percent": round(pct_diff, 1),
            "explanation_drivers": drivers,
        }
