"""
Strategy Filters — Liquidity Hours
====================================
Filtra señales fuera de horarios de alta liquidez (London y NY sessions).
Evita operar en horarios de baja liquidez donde los spreads son altos
y los movimientos son erráticos.
"""

import pandas as pd


# London session: 08:00-12:00 UTC
# New York session: 13:00-17:00 UTC (overlap London-NY: 12-13 UTC included)
LONDON_START = 8
LONDON_END = 12
NY_START = 13
NY_END = 17


def is_liquid_session(index: pd.DatetimeIndex) -> pd.Series:
    """
    Returns a boolean Series indicating whether each timestamp falls within
    a high-liquidity session (London or New York).

    Parameters
    ----------
    index : pd.DatetimeIndex
        DatetimeIndex in UTC.

    Returns
    -------
    pd.Series[bool]
        True if within London (08-12 UTC) or NY (13-17 UTC) session.
    """
    hour = index.hour
    in_london = (hour >= LONDON_START) & (hour < LONDON_END)
    in_ny = (hour >= NY_START) & (hour < NY_END)
    # Include London-NY overlap (12-13 UTC)
    in_overlap = hour == 12
    return pd.Series(in_london | in_ny | in_overlap, index=index)


def filter_signals_by_session(
    signals: pd.DataFrame,
    signal_col: str = "signal",
) -> pd.DataFrame:
    """
    Zeroes out signals that fall outside high-liquidity sessions.

    Parameters
    ----------
    signals : pd.DataFrame
        Must have a DatetimeIndex (UTC) and a signal column.
    signal_col : str
        Name of the signal column.

    Returns
    -------
    pd.DataFrame
        Copy with signals set to 0 outside London/NY hours.
    """
    signals = signals.copy()
    liquid = is_liquid_session(signals.index)
    signals.loc[~liquid, signal_col] = 0
    return signals
