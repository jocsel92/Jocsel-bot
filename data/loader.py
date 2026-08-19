"""
Data Module — Ingesta y preparación de datos históricos
=========================================================
Carga datos OHLCV en 5 minutos para EURUSD y GBPUSD desde 2020.
Soporta CSV con columnas: datetime, open, high, low, close, volume
(opcionalmente bid, ask para spread real).
"""

import pathlib
from typing import Union

import pandas as pd


SUPPORTED_PAIRS = ("EURUSD", "GBPUSD")
DEFAULT_DATA_DIR = pathlib.Path(__file__).parent / "csv"


def load_csv(
    filepath: Union[str, pathlib.Path],
    pair: str = "EURUSD",
    datetime_col: str = "datetime",
    tz: str = "UTC",
) -> pd.DataFrame:
    """
    Carga un CSV de datos OHLCV 5-min y lo prepara para el pipeline.

    Parameters
    ----------
    filepath : str or Path
        Ruta al archivo CSV.
    pair : str
        Nombre del par (para validación).
    datetime_col : str
        Nombre de la columna de fecha/hora.
    tz : str
        Timezone de los datos (default UTC).

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (UTC), columnas: open, high, low, close, volume
    """
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

    # Eliminar filas con datos faltantes en OHLCV
    df = df.dropna(subset=list(required))

    # Asegurar tipos numéricos
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(required))
    df.index.name = "datetime"
    df.attrs["pair"] = pair

    return df


def filter_date_range(
    df: pd.DataFrame,
    start: str = "2020-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Filtra DataFrame por rango de fechas."""
    mask = df.index >= pd.Timestamp(start, tz="UTC")
    if end:
        mask &= df.index <= pd.Timestamp(end, tz="UTC")
    return df.loc[mask]


def resample_timeframe(df: pd.DataFrame, timeframe: str = "5min") -> pd.DataFrame:
    """
    Resamplea datos a un timeframe específico (por si vienen en 1 min).
    """
    return df.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
