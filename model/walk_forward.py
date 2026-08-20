"""
Walk-Forward Validation
========================
Expanding-window walk-forward cross-validation for time-series models.
Avoids look-ahead bias by always training on past data and testing on the
immediate next period.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    min_train_size: int = 500,
    xgb_params: dict | None = None,
) -> dict:
    """
    Expanding-window walk-forward validation.

    The data is split into n_splits+1 consecutive blocks. For each fold k:
      - Train on blocks 0..k
      - Test on block k+1

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (must be time-ordered).
    y : pd.Series
        Labels aligned with X.
    n_splits : int
        Number of test folds.
    min_train_size : int
        Minimum number of training samples for the first fold.
    xgb_params : dict, optional
        XGBClassifier parameters.

    Returns
    -------
    dict with keys:
        - fold_results: list of dicts with metrics per fold
        - mean_accuracy: float
        - all_predictions: np.ndarray (predictions aligned to test indices)
    """
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
        }

    n_samples = len(X)
    # Calculate fold size ensuring minimum training data
    fold_size = (n_samples - min_train_size) // n_splits
    if fold_size < 10:
        raise ValueError(
            f"Not enough data for {n_splits} splits with min_train_size={min_train_size}. "
            f"Total samples: {n_samples}"
        )

    fold_results = []
    all_preds = pd.Series(np.nan, index=X.index)

    for fold in range(n_splits):
        test_start = min_train_size + fold * fold_size
        test_end = test_start + fold_size
        if fold == n_splits - 1:
            test_end = n_samples  # last fold takes remaining data

        X_train = X.iloc[:test_start]
        y_train = y.iloc[:test_start]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model = XGBClassifier(**xgb_params)
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)
        all_preds.iloc[test_start:test_end] = y_pred

        acc = accuracy_score(y_test, y_pred)
        fold_results.append({
            "fold": fold + 1,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "accuracy": acc,
            "test_start": X_test.index[0],
            "test_end": X_test.index[-1],
        })

    mean_acc = np.mean([r["accuracy"] for r in fold_results])

    return {
        "fold_results": fold_results,
        "mean_accuracy": mean_acc,
        "all_predictions": all_preds,
    }
