"""
=============================================================================
JOCSEL BOT — Trading Bot Institucional VWAP (Todo-en-uno)
=============================================================================
Bot de trading que replica lógica institucional usando VWAP, EMA 200,
detección de pivots y machine learning (XGBoost).

USO:
    python jocsel_bot.py --data tu_archivo.csv --pair EURUSD

REQUISITOS (instalar una vez):
    pip install pandas numpy xgboost scikit-learn joblib

TU CSV DEBE TENER ESTAS COLUMNAS:
    datetime, open, high, low, close, volume

EJEMPLO DE CSV:
    datetime,open,high,low,close,volume
    2020-01-02 08:00:00,1.12130,1.12145,1.12120,1.12140,1500
    2020-01-02 08:05:00,1.12140,1.12160,1.12135,1.12155,1200
=============================================================================
"""

import argparse
import pathlib
from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ===========================================================================
# MÓDULO 1: CARGA DE DATOS
# ===========================================================================

SUPPORTED_PAIRS = ("EURUSD", "GBPUSD")


def load_csv(
    filepath: Union[str, pathlib.Path],
    pair: str = "EURUSD",
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """Carga un CSV de datos OHLCV 5-min y lo prepara."""
    pair = pair.upper()
    if pair not in SUPPORTED_PAIRS:
        raise ValueError(f"Par no soportado: {pair}. Usa {SUPPORTED_PAIRS}")

    df = pd.read_csv(filepath, parse_dates=[datetime_col])
    df = df.rename(columns={datetime_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()

    # Normalizar nombres de columnas a minúsculas
    df.columns = df.columns.str.lower()

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes: {missing}")

    df = df.dropna(subset=list(required))
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required))
    df.index.name = "datetime"
    df.attrs["pair"] = pair

    return df


def filter_date_range(df: pd.DataFrame, start: str = "2020-01-01", end=None):
    """Filtra DataFrame por rango de fechas."""
    mask = df.index >= pd.Timestamp(start, tz="UTC")
    if end:
        mask &= df.index <= pd.Timestamp(end, tz="UTC")
    return df.loc[mask]


# ===========================================================================
# MÓDULO 2: LABELING — Estrategia Institucional VWAP
# ===========================================================================

def compute_anchored_vwap(df: pd.DataFrame, anchor_hour: int = 8) -> pd.Series:
    """VWAP anclado a una hora UTC específica cada día (default 8AM London)."""
    vwap = pd.Series(np.nan, index=df.index, name="vwap_anchored")
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]

    for date, group in df.groupby(df.index.date):
        anchor_time = pd.Timestamp(date, tz=df.index.tz).replace(hour=anchor_hour, minute=0)
        mask = (df.index.date == date) & (df.index >= anchor_time)
        cum_tp_vol = tp_vol.loc[mask].cumsum()
        cum_vol = df.loc[mask, "volume"].cumsum()
        vwap.loc[mask] = cum_tp_vol / cum_vol.replace(0, np.nan)

    return vwap


def compute_daily_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP diario estándar (se reinicia a medianoche UTC)."""
    vwap = pd.Series(np.nan, index=df.index, name="vwap_daily")
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]

    for date, group in df.groupby(df.index.date):
        mask = df.index.date == date
        cum_tp_vol = tp_vol.loc[mask].cumsum()
        cum_vol = df.loc[mask, "volume"].cumsum()
        vwap.loc[mask] = cum_tp_vol / cum_vol.replace(0, np.nan)

    return vwap


def compute_ema(series: pd.Series, period: int = 200) -> pd.Series:
    """Calcula EMA."""
    return series.ewm(span=period, adjust=False).mean()


def compute_session_pivots(df: pd.DataFrame, session_start: int, session_end: int):
    """Calcula high/low de una sesión para detectar pivots."""
    highs = pd.Series(np.nan, index=df.index, name="session_high")
    lows = pd.Series(np.nan, index=df.index, name="session_low")

    for date, _ in df.groupby(df.index.date):
        start = pd.Timestamp(date, tz=df.index.tz).replace(hour=session_start, minute=0)
        end = pd.Timestamp(date, tz=df.index.tz).replace(hour=session_end, minute=0)
        mask = (df.index >= start) & (df.index < end)
        if mask.any():
            h = df.loc[mask, "high"].max()
            lo = df.loc[mask, "low"].min()
            after_mask = (df.index.date == date) & (df.index >= end)
            highs.loc[after_mask] = h
            lows.loc[after_mask] = lo

    return highs, lows


def label_dataset(
    df: pd.DataFrame,
    ema_period: int = 200,
    london_anchor_hour: int = 8,
    ny_open_hour: int = 13,
    retrace_window_hours: float = 3.0,
    rrr_min: float = 1.0,
) -> pd.DataFrame:
    """
    Etiqueta cada vela como 1 (setup válido) o 0 (no setup / trampa).

    Lógica institucional:
    1. Bias: precio vs VWAP anclado y EMA 200
    2. Filtro de dirección: VWAP diario vs VWAP anclado
    3. Entry: pivot break + retrace al VWAP anclado en ventana NY
    4. RRR: stop debajo del retrace low (longs) / arriba del retrace high (shorts)
    """
    df = df.copy()

    # Indicadores
    df["vwap_anchored"] = compute_anchored_vwap(df, anchor_hour=london_anchor_hour)
    df["vwap_daily"] = compute_daily_vwap(df)
    df["ema_200"] = compute_ema(df["close"], period=ema_period)

    # 1. Bias
    long_bias = (df["close"] > df["vwap_anchored"]) & (df["close"] > df["ema_200"])
    short_bias = (df["close"] < df["vwap_anchored"]) & (df["close"] < df["ema_200"])
    df["bias"] = 0
    df.loc[long_bias, "bias"] = 1
    df.loc[short_bias, "bias"] = -1

    # 2. Direction filter
    long_direction = df["vwap_daily"] > df["vwap_anchored"]
    short_direction = df["vwap_daily"] < df["vwap_anchored"]

    # 3. Session pivots
    london_highs, london_lows = compute_session_pivots(df, 8, 12)
    ny_highs, ny_lows = compute_session_pivots(df, 13, 17)
    df["pivot_high"] = np.maximum(london_highs.fillna(-np.inf), ny_highs.fillna(-np.inf))
    df["pivot_low"] = np.minimum(london_lows.fillna(np.inf), ny_lows.fillna(np.inf))

    # 4. Label logic
    df["label"] = 0
    df["stop"] = np.nan
    df["target"] = np.nan
    df["rrr"] = np.nan

    for date in df.index.normalize().unique():
        ny_open = date + pd.Timedelta(hours=ny_open_hour)
        window_end = ny_open + pd.Timedelta(hours=retrace_window_hours)
        day_mask = (df.index >= ny_open) & (df.index <= window_end)
        day_df = df.loc[day_mask]

        if day_df.empty:
            continue

        for i, (ts, row) in enumerate(day_df.iterrows()):
            # --- Long setup ---
            if row["bias"] == 1 and long_direction.get(ts, False):
                if row["high"] >= row["pivot_high"]:
                    future = day_df.iloc[i:]
                    touched_vwap = future["low"] <= future["vwap_anchored"]
                    if touched_vwap.any():
                        touch_idx = touched_vwap.idxmax()
                        retrace_low = day_df.loc[ts:touch_idx, "low"].min()
                        after_touch = day_df.loc[touch_idx:]
                        if not after_touch.empty:
                            close_after = after_touch.iloc[-1]["close"]
                            vwap_at_close = after_touch.iloc[-1]["vwap_anchored"]
                            if close_after > vwap_at_close:
                                stop = retrace_low
                                risk = row["close"] - stop
                                if risk > 0:
                                    target = row["close"] + risk * rrr_min
                                    df.at[ts, "label"] = 1
                                    df.at[ts, "stop"] = stop
                                    df.at[ts, "target"] = target
                                    df.at[ts, "rrr"] = rrr_min
                            else:
                                df.at[ts, "label"] = 0

            # --- Short setup ---
            elif row["bias"] == -1 and short_direction.get(ts, False):
                if row["low"] <= row["pivot_low"]:
                    future = day_df.iloc[i:]
                    touched_vwap = future["high"] >= future["vwap_anchored"]
                    if touched_vwap.any():
                        touch_idx = touched_vwap.idxmax()
                        retrace_high = day_df.loc[ts:touch_idx, "high"].max()
                        after_touch = day_df.loc[touch_idx:]
                        if not after_touch.empty:
                            close_after = after_touch.iloc[-1]["close"]
                            vwap_at_close = after_touch.iloc[-1]["vwap_anchored"]
                            if close_after < vwap_at_close:
                                stop = retrace_high
                                risk = stop - row["close"]
                                if risk > 0:
                                    target = row["close"] - risk * rrr_min
                                    df.at[ts, "label"] = 1
                                    df.at[ts, "stop"] = stop
                                    df.at[ts, "target"] = target
                                    df.at[ts, "rrr"] = rrr_min
                            else:
                                df.at[ts, "label"] = 0

    # Protección anti look-ahead: shift labels +1 bar
    df["label"] = df["label"].shift(1).fillna(0).astype(int)

    return df


# ===========================================================================
# MÓDULO 3: FEATURES — Spread
# ===========================================================================

def compute_spread(df: pd.DataFrame, method: str = "hl_proxy") -> pd.Series:
    """Calcula el spread en pips."""
    PIP_MULTIPLIER = 10_000

    if method == "bid_ask":
        if "ask" not in df.columns or "bid" not in df.columns:
            raise ValueError("Se requieren columnas 'ask' y 'bid' para method='bid_ask'")
        spread = (df["ask"] - df["bid"]) * PIP_MULTIPLIER
    elif method == "hl_proxy":
        spread = (df["high"] - df["low"]) * PIP_MULTIPLIER
    else:
        raise ValueError(f"Método no soportado: {method}")

    return spread.rename("spread_pips")


def compute_spread_features(df: pd.DataFrame, method: str = "hl_proxy", window: int = 20):
    """Genera features del spread: actual, media móvil, y ratio."""
    spread = compute_spread(df, method=method)
    spread_ma = spread.rolling(window).mean()
    spread_ratio = spread / spread_ma.replace(0, np.nan)

    return pd.DataFrame({
        "spread_pips": spread,
        "spread_ma": spread_ma,
        "spread_ratio": spread_ratio,
    }, index=df.index)


# ===========================================================================
# MÓDULO 4: FILTRO DE LIQUIDEZ — Solo London y NY
# ===========================================================================

LONDON_START = 8
LONDON_END = 12
NY_START = 13
NY_END = 17


def is_liquid_session(index: pd.DatetimeIndex) -> pd.Series:
    """True si el timestamp está en sesión London (8-12 UTC) o NY (13-17 UTC)."""
    hour = index.hour
    in_london = (hour >= LONDON_START) & (hour < LONDON_END)
    in_ny = (hour >= NY_START) & (hour < NY_END)
    in_overlap = hour == 12  # London-NY overlap
    return pd.Series(in_london | in_ny | in_overlap, index=index)


# ===========================================================================
# MÓDULO 5: WALK-FORWARD VALIDATION
# ===========================================================================

def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    min_train_size: int = 500,
    xgb_params: dict | None = None,
) -> dict:
    """
    Validación walk-forward con ventana expandible.
    Siempre entrena con datos pasados y testea con el periodo siguiente.
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
    fold_size = (n_samples - min_train_size) // n_splits
    if fold_size < 10:
        raise ValueError(
            f"No hay suficientes datos para {n_splits} splits con "
            f"min_train_size={min_train_size}. Total: {n_samples}"
        )

    fold_results = []
    all_preds = pd.Series(np.nan, index=X.index)

    for fold in range(n_splits):
        test_start = min_train_size + fold * fold_size
        test_end = test_start + fold_size
        if fold == n_splits - 1:
            test_end = n_samples

        X_train = X.iloc[:test_start]
        y_train = y.iloc[:test_start]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(**xgb_params)
        model.fit(X_train_scaled, y_train)

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


# ===========================================================================
# MÓDULO 6: ENTRENAMIENTO DEL MODELO
# ===========================================================================

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_path: str = "xgboost_model.joblib",
    scaler_path: str = "scaler.joblib",
    xgb_params: dict | None = None,
):
    """Escala features, entrena XGBoost y guarda modelo + scaler."""
    if xgb_params is None:
        xgb_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(**xgb_params)
    model.fit(X_scaled, y)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler


# ===========================================================================
# MÓDULO 7: SEÑALES Y GESTIÓN DE RIESGO
# ===========================================================================

def generate_signals(df: pd.DataFrame, predictions: np.ndarray):
    """Convierte predicciones en señales BUY/SELL."""
    signals = pd.DataFrame(index=df.index)
    signals["prediction"] = predictions
    signals["bias"] = df["bias"] if "bias" in df.columns else 0

    signals["signal"] = 0
    long_mask = (signals["prediction"] == 1) & (signals["bias"] == 1)
    short_mask = (signals["prediction"] == 1) & (signals["bias"] == -1)
    signals.loc[long_mask, "signal"] = 1
    signals.loc[short_mask, "signal"] = -1

    signals["entry_price"] = np.where(signals["signal"] != 0, df["close"], np.nan)

    return signals


def apply_risk_management(signals: pd.DataFrame, df: pd.DataFrame, risk_pct=0.01, rrr=1.5):
    """Aplica stops, targets, position sizing y límite diario de trades."""
    signals = signals.copy()

    if "stop" in df.columns:
        signals["stop"] = df["stop"]
        signals["target"] = df["target"]
    else:
        lookback = 6
        for i, (ts, row) in enumerate(signals.iterrows()):
            if row["signal"] == 1:
                start = max(0, i - lookback)
                signals.at[ts, "stop"] = df.iloc[start:i + 1]["low"].min()
                risk = row["entry_price"] - signals.at[ts, "stop"]
                signals.at[ts, "target"] = row["entry_price"] + risk * rrr
            elif row["signal"] == -1:
                start = max(0, i - lookback)
                signals.at[ts, "stop"] = df.iloc[start:i + 1]["high"].max()
                risk = signals.at[ts, "stop"] - row["entry_price"]
                signals.at[ts, "target"] = row["entry_price"] - risk * rrr

    signals["risk_distance"] = np.abs(signals["entry_price"] - signals["stop"])
    signals["position_size_pct"] = np.where(
        signals["risk_distance"] > 0,
        risk_pct / signals["risk_distance"] * signals["entry_price"],
        0,
    )

    # Máximo 3 trades por día
    signals["daily_group"] = signals.index.date
    daily_trades = signals.groupby("daily_group")["signal"].transform(
        lambda x: (x != 0).cumsum()
    )
    signals.loc[daily_trades > 3, "signal"] = 0
    signals = signals.drop(columns=["daily_group"], errors="ignore")

    return signals


# ===========================================================================
# PIPELINE PRINCIPAL — Todo junto
# ===========================================================================

def run_pipeline(csv_path: str, pair: str = "EURUSD", start: str = "2020-01-01"):
    """
    Ejecuta el pipeline completo:
    1. Carga datos CSV
    2. Etiqueta con lógica institucional VWAP
    3. Calcula features (spread)
    4. Filtra horarios de baja liquidez
    5. Walk-forward validation
    6. Entrena modelo final XGBoost
    """

    print("=" * 60)
    print("  JOCSEL BOT — Pipeline de Entrenamiento")
    print("=" * 60)

    # --- PASO 1: Cargar datos ---
    print(f"\n[1/6] Cargando datos de {pair} desde {csv_path}...")
    df = load_csv(csv_path, pair=pair)
    df = filter_date_range(df, start=start)
    print(f"      → {len(df)} velas cargadas ({df.index.min()} a {df.index.max()})")

    # --- PASO 2: Labeling ---
    print("\n[2/6] Etiquetando con lógica institucional VWAP...")
    df = label_dataset(df)
    n_valid = (df["label"] == 1).sum()
    print(f"      → {n_valid} setups válidos etiquetados ({n_valid / len(df) * 100:.2f}%)")

    # --- PASO 3: Features ---
    print("\n[3/6] Calculando features (spread, indicadores)...")
    spread_feats = compute_spread_features(df, method="hl_proxy")
    df = pd.concat([df, spread_feats], axis=1)

    feature_cols = [
        "vwap_anchored", "vwap_daily", "ema_200",
        "bias", "spread_pips", "spread_ma", "spread_ratio",
    ]
    available_features = [c for c in feature_cols if c in df.columns]
    print(f"      → Features: {available_features}")

    df_train = df.dropna(subset=available_features + ["label"])
    X = df_train[available_features]
    y = df_train["label"]

    # --- PASO 4: Filtro de liquidez ---
    print("\n[4/6] Filtrando datos fuera de horarios de alta liquidez (London/NY)...")
    liquid_mask = is_liquid_session(X.index)
    X = X[liquid_mask]
    y = y[liquid_mask]
    print(f"      → {len(X)} muestras en sesiones líquidas (descartadas {(~liquid_mask).sum()})")

    # --- PASO 5: Walk-forward validation ---
    print(f"\n[5/6] Walk-forward validation ({len(X)} muestras, {len(available_features)} features)...")
    wf_results = walk_forward_validate(X, y, n_splits=5, min_train_size=max(500, len(X) // 6))
    print(f"      → Mean accuracy (walk-forward): {wf_results['mean_accuracy']:.4f}")
    for fold in wf_results["fold_results"]:
        print(f"        Fold {fold['fold']}: train={fold['train_size']}, "
              f"test={fold['test_size']}, acc={fold['accuracy']:.4f} "
              f"({fold['test_start'].date()} → {fold['test_end'].date()})")

    # --- PASO 6: Entrenar modelo final ---
    print(f"\n[6/6] Entrenando modelo final XGBoost ({len(X)} muestras)...")
    model, scaler = train_model(X, y)
    print("      → Modelo guardado: xgboost_model.joblib")
    print("      → Scaler guardado: scaler.joblib")

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    print("\n      Métricas in-sample (referencia):")
    print(classification_report(y, y_pred, target_names=["No Setup (0)", "Setup Válido (1)"]))

    print("=" * 60)
    print("  ✅ ENTRENAMIENTO COMPLETO")
    print("  Archivos generados:")
    print("    • xgboost_model.joblib (modelo)")
    print("    • scaler.joblib (normalizador)")
    print("=" * 60)

    return model, scaler, df


# ===========================================================================
# PUNTO DE ENTRADA — Ejecutar con: python jocsel_bot.py --data tu_csv.csv
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Jocsel Bot — Trading Bot Institucional VWAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo:
    python jocsel_bot.py --data datos/eurusd_5min.csv --pair EURUSD

Tu CSV debe tener columnas: datetime, open, high, low, close, volume
        """,
    )
    parser.add_argument("--data", required=True, help="Ruta al CSV de datos 5min")
    parser.add_argument("--pair", default="EURUSD", choices=["EURUSD", "GBPUSD"],
                        help="Par de divisas (default: EURUSD)")
    parser.add_argument("--start", default="2020-01-01",
                        help="Fecha inicio YYYY-MM-DD (default: 2020-01-01)")
    args = parser.parse_args()

    run_pipeline(args.data, pair=args.pair, start=args.start)
