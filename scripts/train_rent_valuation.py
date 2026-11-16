"""
ML Model #2: LightGBM Quantile Rent Regressor (FastMCP Deal Tool)
Trained on: Area_sqft, BHK, Location, Property_type, Status, Facing, Seller_type, Security_deposit.
Predicts fair rent range [P10, P50, P90] + real-time SHAP drivers.
"""

import os
import sys
import re
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psycopg
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_rent_valuation")

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin123")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "habitatos")

DB_DSN = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUT = MODEL_DIR / "quantile_rent_regressor.pkl"
ALT_MODEL_OUT = MODEL_DIR / "valuation_models.pkl"

CSV_FALLBACK = Path.home() / ".cache/kagglehub/datasets/divyanshug40/data-for-houses-available-for-rent/versions/1/Makaan_data_700pages.csv"

CAT_COLS = ["location", "property_type", "status", "facing_direction", "seller_type"]
NUM_COLS = ["bhk", "area_sqft", "bathrooms", "security_deposit", "sqft_per_bhk"]
FEATURES = NUM_COLS + CAT_COLS


def detect_device() -> tuple[str, dict]:
    """Check device acceleration (GPU / MPS / CUDA / CPU)."""
    try:
        import torch
        if torch.backends.mps.is_available():
            logger.info("Apple Silicon Metal GPU (MPS) active for PyTorch / Voice Engine.")
    except ImportError:
        pass

    try:
        dummy_X = np.array([[1.0, 2.0], [3.0, 4.0]])
        dummy_y = np.array([1.0, 2.0])
        test_m = lgb.LGBMRegressor(device="gpu", n_estimators=1, verbose=-1)
        test_m.fit(dummy_X, dummy_y)
        logger.info("LightGBM OpenCL GPU tree learner active.")
        return "gpu", {"device": "gpu"}
    except Exception:
        pass

    try:
        dummy_X = np.array([[1.0, 2.0], [3.0, 4.0]])
        dummy_y = np.array([1.0, 2.0])
        test_m = lgb.LGBMRegressor(device="cuda", n_estimators=1, verbose=-1)
        test_m.fit(dummy_X, dummy_y)
        logger.info("LightGBM CUDA GPU tree learner active.")
        return "cuda", {"device": "cuda"}
    except Exception:
        pass

    logger.info("LightGBM running on high-throughput multi-core CPU (n_jobs=-1, OpenMP).")
    return "cpu", {"device": "cpu", "n_jobs": -1}


def clean_deposit_val(val) -> float:
    """Parse security deposit string into integer/float in INR."""
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


def load_dataset() -> pd.DataFrame:
    """Load and normalize rental dataset from PostgreSQL or Kaggle CSV."""
    logger.info(f"Connecting to PostgreSQL: {DB_DSN.split('@')[-1] if '@' in DB_DSN else DB_DSN}...")
    try:
        with psycopg.connect(DB_DSN) as conn:
            # Check if source column exists in listings
            with conn.cursor() as cur:
                cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'listings';")
                existing_cols = {row[0] for row in cur.fetchall()}

            # Adapt query to match listings schema
            price_col = "price" if "price" in existing_cols else "rent_price"
            bhk_col = "bedrooms" if "bedrooms" in existing_cols else "bhk"
            loc_col = "locality" if "locality" in existing_cols else "location"
            status_col = "furnishing_status" if "furnishing_status" in existing_cols else "status"
            seller_col = "contact" if "contact" in existing_cols else "seller_type"

            query = f"""
                SELECT 
                    {price_col} AS rent_price,
                    {bhk_col} AS bhk,
                    bathrooms,
                    area_sqft,
                    {loc_col} AS location,
                    property_type,
                    {status_col} AS status,
                    facing_direction,
                    {seller_col} AS seller_type,
                    security_deposit
                FROM listings
                {"WHERE source = 'kaggle_makaan_delhi'" if "source" in existing_cols else ""};
            """
            logger.info("Executing listings extraction query...")
            df = pd.read_sql(query, conn)
            if len(df) > 0:
                logger.info(f"Loaded {len(df)} listings from PostgreSQL.")
                # Clean seller_type if extracted from contact string (e.g. 'Agent (XYZ)')
                df["seller_type"] = df["seller_type"].apply(
                    lambda s: str(s).split(" (")[0].strip() if s else "Unknown"
                )
                df["security_deposit"] = df["security_deposit"].apply(clean_deposit_val)
                return df
    except Exception as e:
        logger.warning(f"PostgreSQL load failed ({e}). Checking local CSV fallback...")

    if CSV_FALLBACK.exists():
        logger.info(f"Reading from local Kaggle dataset: {CSV_FALLBACK}")
        raw = pd.read_csv(CSV_FALLBACK)
        df = pd.DataFrame()
        df["bhk"] = raw["Size"].astype(int)
        df["bathrooms"] = raw["Bathroom"].fillna(df["bhk"]).astype(int)
        df["rent_price"] = raw["Rent_price"].astype(str).str.replace(",", "").str.strip().astype(float)
        df["area_sqft"] = raw["Area_sqft"].astype(int)
        df["location"] = raw["Location"].str.strip()
        df["property_type"] = raw["Property_type"].str.strip()
        df["status"] = raw["Status"].str.strip()
        df["facing_direction"] = raw["Facing_direction"].fillna("Not specified").str.strip()
        df["seller_type"] = raw["Seller_type"].str.strip()
        df["security_deposit"] = raw["Security_deposit"].apply(clean_deposit_val)
        return df

    raise RuntimeError("Could not load data from database or CSV fallback.")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer feature set for LightGBM quantile regression."""
    X = df.copy()
    X["bhk"] = pd.to_numeric(X["bhk"], errors="coerce").fillna(1).astype(int)
    X["area_sqft"] = pd.to_numeric(X["area_sqft"], errors="coerce").fillna(500).astype(float)
    X["bathrooms"] = pd.to_numeric(X["bathrooms"], errors="coerce").fillna(X["bhk"]).astype(int)
    X["security_deposit"] = pd.to_numeric(X["security_deposit"], errors="coerce").fillna(0.0).astype(float)
    X["sqft_per_bhk"] = X["area_sqft"] / X["bhk"].clip(lower=1)

    for col in CAT_COLS:
        X[col] = X[col].fillna("Unknown").astype(str).str.strip().astype("category")

    return X[FEATURES]


def main():
    print("=" * 65)
    print("STAGE 4: LIGHTGBM QUANTILE RENT VALUATION TRAINING")
    print("Dataset: divyanshug40/data-for-houses-available-for-rent (14K rows)")
    print("=" * 65)

    device_name, device_params = detect_device()
    df = load_dataset()

    # Filter invalid records
    df = df[(df["rent_price"] >= 2000) & (df["rent_price"] <= 1500000) & (df["area_sqft"] >= 100)].copy()
    logger.info(f"Cleaned dataset: {len(df)} listings available for training.")

    X = prepare_features(df)
    # Log-transform skewed Indian rental prices for stable quantile intervals
    y_log = np.log1p(df["rent_price"].to_numpy())

    # Stratify split by top localities so validation covers all Delhi NCR micro-markets
    top_locs = df["location"].value_counts()
    strat_col = df["location"].where(df["location"].isin(top_locs[top_locs >= 10].index), "Other")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42, stratify=strat_col
    )
    logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_test)}")

    quantiles = [0.10, 0.50, 0.90]
    models = {}
    preds_inr = {}

    for alpha in quantiles:
        logger.info(f"Fitting Quantile Regressor for alpha={alpha:.2f}...")
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample_for_bin=50000,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42,
            verbose=-1,
            **device_params,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        models[alpha] = model
        # Store string key alias for backward compatibility
        models[f"p{int(alpha*100)}"] = model

        # Inverse log1p transform back to INR
        preds_inr[alpha] = np.expm1(model.predict(X_test))

    y_test_inr = np.expm1(y_test)
    mae_50 = mean_absolute_error(y_test_inr, preds_inr[0.50])
    pinball_50 = mean_pinball_loss(y_test_inr, preds_inr[0.50], alpha=0.50)
    coverage = np.mean((y_test_inr >= preds_inr[0.10]) & (y_test_inr <= preds_inr[0.90]))

    print("-" * 65)
    print(f"Hardware Acceleration : {device_name.upper()} (Multi-Core OpenMP)")
    print(f"Median (P50) MAE      : INR {mae_50:,.0f}")
    print(f"Median Pinball Loss   : INR {pinball_50:,.0f}")
    print(f"80% Interval Coverage : {coverage * 100:.1f}% (Target: ~80.0%)")
    print("-" * 65)

    # Build SHAP TreeExplainer on the median (P50) model for real-time deal explanations
    logger.info("Building SHAP TreeExplainer on median (P50) model...")
    explainer = shap.TreeExplainer(models[0.50])

    artifacts = {
        "models": models,
        "explainer": explainer,
        "features": FEATURES,
        "feature_names": FEATURES,
        "cat_cols": CAT_COLS,
        "categorical_features": CAT_COLS,
        "categories": {c: X_train[c].cat.categories.tolist() for c in CAT_COLS},
        "is_log_target": True,
        "metrics": {
            "mae_p50": float(mae_50),
            "pinball_loss_p50": float(pinball_50),
            "coverage_80": float(coverage),
        },
    }

    joblib.dump(artifacts, MODEL_OUT)
    print(f"Saved Quantile Rent Regressor + SHAP explainer to {MODEL_OUT}")

    # Also save to valuation_models.pkl so existing pipeline loads seamlessly
    joblib.dump(artifacts, ALT_MODEL_OUT)
    print(f"Mirrored model artifact to {ALT_MODEL_OUT}")

    # Test sample evaluation with SHAP output
    sample_df = X_test.iloc[[0]]
    sample_p10 = np.expm1(models[0.10].predict(sample_df)[0])
    sample_p50 = np.expm1(models[0.50].predict(sample_df)[0])
    sample_p90 = np.expm1(models[0.90].predict(sample_df)[0])
    sample_actual = y_test_inr[0]

    shap_vals = explainer(sample_df)
    top_driver_idx = np.argmax(np.abs(shap_vals.values[0]))
    top_driver_feat = FEATURES[top_driver_idx]
    top_driver_impact = shap_vals.values[0][top_driver_idx]

    print("\nSAMPLE TEST INFERENCE & SHAP EXPLANATION:")
    print(f"Actual Listing Rent : INR {sample_actual:,.0f}")
    print(f"Predicted P10 (Low) : INR {sample_p10:,.0f}")
    print(f"Predicted P50 (Med) : INR {sample_p50:,.0f}")
    print(f"Predicted P90 (High): INR {sample_p90:,.0f}")
    print(f"Key SHAP Driver     : {top_driver_feat} -> {top_driver_impact:+.3f} log impact")
    print("=" * 65)


if __name__ == "__main__":
    main()
