"""
Pipeline Principal — Jocsel Bot
=================================
Conecta los 5 módulos para entrenar el modelo:
1. data/       → Carga CSV de EURUSD/GBPUSD 5min desde 2020
2. labeling/   → Etiqueta con lógica institucional VWAP
3. features/   → Calcula spread y features adicionales
4. model/      → Walk-forward validation + entrena XGBoost final
5. strategy/   → Genera señales con gestión de riesgo + filtro de liquidez

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
from model.walk_forward import walk_forward_validate
from strategy.filters import is_liquid_session


def run_pipeline(csv_path: str, pair: str = "EURUSD", start: str = "2020-01-01"):
    """Ejecuta el pipeline completo de entrenamiento."""

    print(f"[1/6] Cargando datos de {pair} desde {csv_path}...")
    df = load_csv(csv_path, pair=pair)
    df = filter_date_range(df, start=start)
    print(f"      → {len(df)} velas cargadas ({df.index.min()} a {df.index.max()})")

    print("[2/6] Etiquetando con lógica institucional VWAP...")
    df = label_dataset(df)
    n_valid = (df["label"] == 1).sum()
    print(f"      → {n_valid} setups válidos etiquetados ({n_valid/len(df)*100:.2f}%)")

    print("[3/6] Calculando features (spread, indicadores)...")
    spread_feats = compute_spread_features(df, method="hl_proxy")
    df = pd.concat([df, spread_feats], axis=1)

    # Features para el modelo
    feature_cols = [
        "vwap_anchored", "vwap_daily", "ema_200",
        "bias", "spread_pips", "spread_ma", "spread_ratio",
    ]
    available_features = [c for c in feature_cols if c in df.columns]
    print(f"      → Features: {available_features}")

    # Eliminar filas sin label o features NaN
    df_train = df.dropna(subset=available_features + ["label"])
    X = df_train[available_features]
    y = df_train["label"]

    # --- 4. Filtro de liquidez: solo entrenar con datos en sesiones líquidas ---
    print("[4/6] Filtrando datos fuera de horarios de alta liquidez (London/NY)...")
    liquid_mask = is_liquid_session(X.index)
    X = X[liquid_mask]
    y = y[liquid_mask]
    print(f"      → {len(X)} muestras en sesiones líquidas (descartadas {(~liquid_mask).sum()})")

    # --- 5. Walk-forward validation ---
    print(f"[5/6] Walk-forward validation ({len(X)} muestras, {len(available_features)} features)...")
    wf_results = walk_forward_validate(X, y, n_splits=5, min_train_size=max(500, len(X) // 6))
    print(f"      → Mean accuracy (walk-forward): {wf_results['mean_accuracy']:.4f}")
    for fold in wf_results["fold_results"]:
        print(f"        Fold {fold['fold']}: train={fold['train_size']}, "
              f"test={fold['test_size']}, acc={fold['accuracy']:.4f} "
              f"({fold['test_start'].date()} → {fold['test_end'].date()})")

    # --- 6. Train final model on all liquid data ---
    print(f"[6/6] Entrenando modelo final XGBoost ({len(X)} muestras)...")
    model, scaler = train_model(X, y)
    print("      → Modelo guardado: xgboost_model.joblib")
    print("      → Scaler guardado: scaler.joblib")

    # Métricas del modelo final (in-sample, referencia)
    from sklearn.metrics import classification_report
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    print("\n      Métricas in-sample (referencia, usar walk-forward para evaluación real):")
    print(classification_report(y, y_pred, target_names=["No Setup (0)", "Setup Válido (1)"]))

    return model, scaler, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jocsel Bot — Pipeline de entrenamiento")
    parser.add_argument("--data", required=True, help="Ruta al CSV de datos 5min")
    parser.add_argument("--pair", default="EURUSD", choices=["EURUSD", "GBPUSD"])
    parser.add_argument("--start", default="2020-01-01", help="Fecha inicio (YYYY-MM-DD)")
    args = parser.parse_args()

    run_pipeline(args.data, pair=args.pair, start=args.start)
