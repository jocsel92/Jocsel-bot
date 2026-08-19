"""
Strategy Module — Ejecución de señales y gestión de riesgo
============================================================
Genera señales de trading basadas en las predicciones del modelo,
aplica stops, targets y reglas de gestión de riesgo.
"""

import numpy as np
import pandas as pd


def generate_signals(
    df: pd.DataFrame,
    predictions: np.ndarray,
    bias_col: str = "bias",
) -> pd.DataFrame:
    """
    Convierte predicciones del modelo en señales de trading.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas de indicators y bias.
    predictions : np.ndarray
        Predicciones del modelo (1=setup válido, 0=no).
    bias_col : str
        Columna con el bias (1=long, -1=short, 0=neutral).

    Returns
    -------
    pd.DataFrame
        Columnas: signal (1=buy, -1=sell, 0=flat), entry_price, stop, target
    """
    signals = pd.DataFrame(index=df.index)
    signals["prediction"] = predictions
    signals["bias"] = df[bias_col] if bias_col in df.columns else 0

    # Solo generar señal cuando modelo predice 1 Y hay bias definido
    signals["signal"] = 0
    long_mask = (signals["prediction"] == 1) & (signals["bias"] == 1)
    short_mask = (signals["prediction"] == 1) & (signals["bias"] == -1)
    signals.loc[long_mask, "signal"] = 1
    signals.loc[short_mask, "signal"] = -1

    signals["entry_price"] = np.where(signals["signal"] != 0, df["close"], np.nan)

    return signals


def apply_risk_management(
    signals: pd.DataFrame,
    df: pd.DataFrame,
    risk_pct: float = 0.01,
    rrr: float = 1.5,
    max_daily_loss_pct: float = 0.03,
) -> pd.DataFrame:
    """
    Aplica gestión de riesgo a las señales.

    Parameters
    ----------
    signals : pd.DataFrame
        DataFrame con señales generadas.
    df : pd.DataFrame
        DataFrame original con stop/target del labeling si existen.
    risk_pct : float
        Porcentaje de la cuenta a arriesgar por trade.
    rrr : float
        Reward-to-risk ratio objetivo.
    max_daily_loss_pct : float
        Máxima pérdida diaria permitida (% de la cuenta).

    Returns
    -------
    pd.DataFrame
        signals con columnas adicionales: stop, target, position_size_pct
    """
    signals = signals.copy()

    # Usar stops del labeling si existen, sino calcular con ATR proxy
    if "stop" in df.columns:
        signals["stop"] = df["stop"]
        signals["target"] = df["target"]
    else:
        # Fallback: stop basado en el mínimo/máximo de las últimas 6 velas
        lookback = 6
        for i, (ts, row) in enumerate(signals.iterrows()):
            if row["signal"] == 1:  # Long
                start = max(0, i - lookback)
                signals.at[ts, "stop"] = df.iloc[start:i + 1]["low"].min()
                risk = row["entry_price"] - signals.at[ts, "stop"]
                signals.at[ts, "target"] = row["entry_price"] + risk * rrr
            elif row["signal"] == -1:  # Short
                start = max(0, i - lookback)
                signals.at[ts, "stop"] = df.iloc[start:i + 1]["high"].max()
                risk = signals.at[ts, "stop"] - row["entry_price"]
                signals.at[ts, "target"] = row["entry_price"] - risk * rrr

    # Position sizing: riesgo fijo por trade
    signals["risk_distance"] = np.abs(signals["entry_price"] - signals["stop"])
    signals["position_size_pct"] = np.where(
        signals["risk_distance"] > 0,
        risk_pct / signals["risk_distance"] * signals["entry_price"],
        0,
    )

    # Límite de pérdida diaria
    signals["daily_group"] = signals.index.date
    daily_trades = signals.groupby("daily_group")["signal"].transform(
        lambda x: (x != 0).cumsum()
    )
    # Máximo 3 trades por día como filtro conservador
    signals.loc[daily_trades > 3, "signal"] = 0

    signals = signals.drop(columns=["daily_group"], errors="ignore")

    return signals
