"""
Features Module — Spread Calculation
======================================
Calcula el spread (ask - bid) para EURUSD y GBPUSD.
Si solo tienes OHLCV sin bid/ask, estima el spread con high - low del bar (proxy).
"""

import numpy as np
import pandas as pd


def compute_spread(df: pd.DataFrame, method: str = "bid_ask") -> pd.Series:
    """
    Calcula el spread de un par de divisas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas dependiendo del método:
        - method='bid_ask': requiere columnas 'ask' y 'bid'
        - method='hl_proxy': requiere columnas 'high' y 'low' (proxy con rango del bar)

    method : str
        'bid_ask' → spread real (ask - bid)
        'hl_proxy' → estimación con high - low del bar de 5 minutos

    Returns
    -------
    pd.Series
        Spread en pips (multiplicado por 10_000 para pares XXX/USD de 4 decimales).
    """
    PIP_MULTIPLIER = 10_000  # Para EURUSD y GBPUSD (4 decimales)

    if method == "bid_ask":
        if "ask" not in df.columns or "bid" not in df.columns:
            raise ValueError("Se requieren columnas 'ask' y 'bid' para method='bid_ask'")
        spread = (df["ask"] - df["bid"]) * PIP_MULTIPLIER
    elif method == "hl_proxy":
        spread = (df["high"] - df["low"]) * PIP_MULTIPLIER
    else:
        raise ValueError(f"Método no soportado: {method}. Usa 'bid_ask' o 'hl_proxy'")

    return spread.rename("spread_pips")


def compute_spread_features(df: pd.DataFrame, method: str = "hl_proxy", window: int = 20) -> pd.DataFrame:
    """
    Genera features basados en el spread: spread actual, media móvil del spread,
    y ratio spread/media (para detectar momentos de spread anormalmente alto).

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    method : str
        Método de cálculo del spread.
    window : int
        Ventana para la media móvil del spread.

    Returns
    -------
    pd.DataFrame
        Columnas: spread_pips, spread_ma, spread_ratio
    """
    spread = compute_spread(df, method=method)
    spread_ma = spread.rolling(window).mean()
    spread_ratio = spread / spread_ma.replace(0, np.nan)

    return pd.DataFrame({
        "spread_pips": spread,
        "spread_ma": spread_ma,
        "spread_ratio": spread_ratio,
    }, index=df.index)
