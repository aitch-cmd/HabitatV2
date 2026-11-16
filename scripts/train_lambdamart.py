"""
Train custom LambdaMART Ranker (LightGBM LGBMRanker) for Stage 3 Reranking.
Wraps and executes scripts/train_reranker.py.
"""

from scripts.train_reranker import main

if __name__ == "__main__":
    main()
