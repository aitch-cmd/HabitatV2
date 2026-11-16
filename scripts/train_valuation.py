"""
Train Quantile Rent Valuation Models (P10, P50, P90) with LightGBM and SHAP explainer.
Wraps and executes scripts/train_rent_valuation.py.
"""

from scripts.train_rent_valuation import main

if __name__ == "__main__":
    main()
