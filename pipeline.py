"""
Pipeline Principal — Jocsel Bot
=================================
Conecta los 5 módulos para entrenar el modelo:
1. data/       → Carga CSV de EURUSD/GBPUSD 5min desde 2020
2. labeling/   → Etiqueta con lógica institucional VWAP
3. features/   → Calcula spread y features adicionales
4. model/      → Entrena XGBoost, guarda xgboost_model.joblib y scaler.joblib
5. strategy/   → Genera señales con gestión de riesgo

Uso:
    python pipeline.py --pair EURUSD --data path/to/eurusd_5min.csv
"""

import argparse
import sys

import pandas as pd

from data import load_csv, filter_date_range
from labeling import label_dataset
from features import compute_spread
from features.spread import compute_spread_features
from model import train_model


def run_pipeline(csv_path: str, pair: str = "EURUSD", start: str = "2020-01-01"):
    """Ejecuta el pipeline completo de entrenamiento."""

    print(f"[1/5] Cargando datos de {pair} desde {csv_path}...")
    df = load_csv(csv_path, pair=pair)
    df = filter_date_range(df, start=start)
    print(f"      → {len(df)} velas cargadas ({df.index.min()} a {df.index.max()})")

    print("[2/5] Etiquetando con lógica institucional VWAP...")
    df = label_dataset(df)
    n_valid = (df["label"] == 1).sum()
    print(f"      → {n_valid} setups válidos etiquetados ({n_valid/len(df)*100:.2f}%)")

    print("[3/5] Calculando features (spread, indicadores)...")
    spread_feats = compute_spread_features(df, method="hl_proxy")
    df = pd.concat([df, spread_feats], axis=1)

    # Features para el modelo
    feature_cols = [
        "vwap_anchored", "vwap_daily", "ema_200",
        "bias", "spread_pips", "spread_ma", "spread_ratio",
    ]
    # Agregar columnas numéricas disponibles
    available_features = [c for c in feature_cols if c in df.columns]

    print(f"      → Features: {available_features}")

    # Eliminar filas sin label o features NaN
    df_train = df.dropna(subset=available_features + ["label"])
    X = df_train[available_features]
    y = df_train["label"]

    print(f"[4/5] Entrenando XGBoost ({len(X)} muestras, {len(available_features)} features)...")
    model, scaler = train_model(X, y)
    print("      → Modelo guardado: xgboost_model.joblib")
    print("      → Scaler guardado: scaler.joblib")

    # Métricas básicas
    from sklearn.metrics import classification_report
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    print("\n[5/5] Métricas de entrenamiento:")
    print(classification_report(y, y_pred, target_names=["No Setup (0)", "Setup Válido (1)"]))

    return model, scaler, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jocsel Bot — Pipeline de entrenamiento")
    parser.add_argument("--data", required=True, help="Ruta al CSV de datos 5min")
    parser.add_argument("--pair", default="EURUSD", choices=["EURUSD", "GBPUSD"])
    parser.add_argument("--start", default="2020-01-01", help="Fecha inicio (YYYY-MM-DD)")
    args = parser.parse_args()

    run_pipeline(args.data, pair=args.pair, start=args.start)
