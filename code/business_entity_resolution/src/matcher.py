"""
Supervised Gradient Boosted Entity Matching Model.
Uses LightGBM with tree-based interaction modeling and probability calibration.
"""

import time
import os
import joblib
import numpy as np
import lightgbm as lgb
from typing import List, Tuple, Dict, Any, Optional

from src.features import FEATURE_NAMES

class EntityMatcher:
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 127,
        min_child_samples: int = 25,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            importance_type='gain',
            verbose=-1
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """Train LightGBM model on feature matrix X and binary labels y."""
        t0 = time.time()
        print(f"Training LightGBM on {X.shape[0]} pairs with {X.shape[1]} features...")
        print(f"  Positive pairs: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%) | Negative pairs: {(y == 0).sum()}")
        
        eval_data = [eval_set] if eval_set is not None else None
        self.model.fit(
            X, y,
            eval_set=eval_data,
            callbacks=[lgb.log_evaluation(period=50)] if eval_set is not None else None
        )
        self.is_fitted = True
        print(f"Model trained in {time.time()-t0:.2f}s.")
        
        # Print top feature importances
        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print("\nTop 10 Feature Importances (Gain):")
        for rank, idx in enumerate(sorted_idx[:10], 1):
            print(f"  {rank}. {FEATURE_NAMES[idx]}: {importances[idx]:.2f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probabilities (returns 1D array of positive class probabilities)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        if len(X) == 0:
            return np.array([])
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        self.model = joblib.load(path)
        self.is_fitted = True
        print(f"Model loaded from {path}")
