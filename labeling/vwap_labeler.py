"""
Institutional VWAP Labeling Strategy
=====================================
Replicates institutional logic using:
- VWAP anchored to London 8:00 AM open
- EMA 200 on 5-minute candles
- Daily VWAP

Labels each candle as 1 (valid trade) or 0 (no trade / trap) based on:
1. Bias: price vs anchored VWAP and EMA 200
2. Direction filter: daily VWAP vs anchored VWAP
3. Entry: pivot break + retrace to anchored VWAP within 2-3h of NY open
4. RRR context: stop below retrace low (longs) / above retrace high (shorts)
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: compute VWAP from an anchor time onward
# ---------------------------------------------------------------------------

def compute_anchored_vwap(df: pd.DataFrame, anchor_hour: int = 8) -> pd.Series:
    """Compute VWAP anchored to a specific UTC hour each day (default 8 AM London)."""
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
    """Compute standard daily VWAP (resets each day at midnight UTC)."""
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
    """Compute EMA on a series."""
    return series.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Pivot detection (session high/low)
# ---------------------------------------------------------------------------

def compute_session_pivots(df: pd.DataFrame, session_start: int, session_end: int):
    """Return session high and low for each day between session_start and session_end (UTC hours)."""
    highs = pd.Series(np.nan, index=df.index, name="session_high")
    lows = pd.Series(np.nan, index=df.index, name="session_low")

    for date, _ in df.groupby(df.index.date):
        start = pd.Timestamp(date, tz=df.index.tz).replace(hour=session_start, minute=0)
        end = pd.Timestamp(date, tz=df.index.tz).replace(hour=session_end, minute=0)
        mask = (df.index >= start) & (df.index < end)
        if mask.any():
            h = df.loc[mask, "high"].max()
            lo = df.loc[mask, "low"].min()
            # Propagate to the rest of the day after session_end
            after_mask = (df.index.date == date) & (df.index >= end)
            highs.loc[after_mask] = h
            lows.loc[after_mask] = lo

    return highs, lows


# ---------------------------------------------------------------------------
# Main labeling function
# ---------------------------------------------------------------------------

def label_dataset(
    df: pd.DataFrame,
    ema_period: int = 200,
    london_anchor_hour: int = 8,
    ny_open_hour: int = 13,  # 8 AM NY = 13 UTC (summer)
    retrace_window_hours: float = 3.0,
    rrr_min: float = 1.0,
    rrr_max: float = 1.5,
) -> pd.DataFrame:
    """
    Label a 5-minute OHLCV DataFrame for institutional VWAP strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Must have DatetimeIndex (UTC) and columns: open, high, low, close, volume.
    ema_period : int
        EMA period (default 200).
    london_anchor_hour : int
        UTC hour to anchor VWAP (default 8 = London open).
    ny_open_hour : int
        UTC hour of NY open (default 13 = 8AM ET in summer).
    retrace_window_hours : float
        Hours after NY open to look for retrace to anchored VWAP.
    rrr_min : float
        Minimum reward-to-risk ratio for a valid label.
    rrr_max : float
        Maximum reward-to-risk ratio target.

    Returns
    -------
    pd.DataFrame
        Original df with added columns:
        - vwap_anchored, vwap_daily, ema_200
        - bias (1=long, -1=short, 0=neutral)
        - label (1=valid setup, 0=no setup or trap)
        - stop, target, rrr
    """
    df = df.copy()

    # Indicators
    df["vwap_anchored"] = compute_anchored_vwap(df, anchor_hour=london_anchor_hour)
    df["vwap_daily"] = compute_daily_vwap(df)
    df["ema_200"] = compute_ema(df["close"], period=ema_period)

    # --- 1. Bias ---
    long_bias = (df["close"] > df["vwap_anchored"]) & (df["close"] > df["ema_200"])
    short_bias = (df["close"] < df["vwap_anchored"]) & (df["close"] < df["ema_200"])
    df["bias"] = 0
    df.loc[long_bias, "bias"] = 1
    df.loc[short_bias, "bias"] = -1

    # --- 2. Direction filter ---
    long_direction = df["vwap_daily"] > df["vwap_anchored"]
    short_direction = df["vwap_daily"] < df["vwap_anchored"]

    # --- 3. Session pivots (London 8-12 UTC, NY from 13 UTC) ---
    london_highs, london_lows = compute_session_pivots(df, 8, 12)
    ny_highs, ny_lows = compute_session_pivots(df, 13, 17)
    df["pivot_high"] = np.maximum(
        london_highs.fillna(-np.inf), ny_highs.fillna(-np.inf)
    )
    df["pivot_low"] = np.minimum(
        london_lows.fillna(np.inf), ny_lows.fillna(np.inf)
    )

    # --- 4. Label logic ---
    df["label"] = 0
    df["stop"] = np.nan
    df["target"] = np.nan
    df["rrr"] = np.nan

    retrace_bars = int(retrace_window_hours * 12)  # 5-min bars per hour = 12

    for date in df.index.normalize().unique():
        ny_open = date + pd.Timedelta(hours=ny_open_hour)
        window_end = ny_open + pd.Timedelta(hours=retrace_window_hours)
        day_mask = (df.index >= ny_open) & (df.index <= window_end)
        day_df = df.loc[day_mask]

        if day_df.empty:
            continue

        for i, (ts, row) in enumerate(day_df.iterrows()):
            # --- Long setup ---
            if (row["bias"] == 1 and long_direction.get(ts, False)):
                # Price broke above pivot high earlier
                if row["high"] >= row["pivot_high"]:
                    # Look for retrace to anchored VWAP
                    future = day_df.iloc[i:]
                    touched_vwap = future["low"] <= future["vwap_anchored"]
                    if touched_vwap.any():
                        touch_idx = touched_vwap.idxmax()
                        retrace_low = day_df.loc[ts:touch_idx, "low"].min()
                        # Check close after touch
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
                                # Trap: crossed back below
                                df.at[ts, "label"] = 0

            # --- Short setup ---
            elif (row["bias"] == -1 and short_direction.get(ts, False)):
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

    return df
