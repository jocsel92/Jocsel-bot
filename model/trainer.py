"""
Model Module — XGBoost + StandardScaler
=========================================
- Entrena XGBoost con las features generadas.
- Guarda modelo como xgboost_model.joblib
- Guarda scaler como scaler.joblib
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


DEFAULT_MODEL_PATH = "xgboost_model.joblib"
DEFAULT_SCALER_PATH = "scaler.joblib"


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
    xgb_params: dict | None = None,
) -> tuple[XGBClassifier, StandardScaler]:
    """
    Escala features con StandardScaler, entrena XGBoost y guarda ambos artefactos.

    Parameters
    ----------
    X : pd.DataFrame
        Features (sin la columna label).
    y : pd.Series
        Labels (0 o 1).
    model_path : str
        Ruta para guardar el modelo entrenado.
    scaler_path : str
        Ruta para guardar el scaler.
    xgb_params : dict, optional
        Parámetros personalizados para XGBClassifier.

    Returns
    -------
    tuple[XGBClassifier, StandardScaler]
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

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenar
    model = XGBClassifier(**xgb_params)
    model.fit(X_scaled, y)

    # Guardar artefactos
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler


def load_model(
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
) -> tuple[XGBClassifier, StandardScaler]:
    """Carga modelo y scaler desde disco."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict(X: pd.DataFrame, model_path: str = DEFAULT_MODEL_PATH, scaler_path: str = DEFAULT_SCALER_PATH) -> np.ndarray:
    """Carga artefactos y predice."""
    model, scaler = load_model(model_path, scaler_path)
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)
