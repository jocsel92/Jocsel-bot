"""
Risk Manager — TP, trailing stop, EMA-200 TP adjustment, 1 trade/day.
======================================================================
Manages open positions: trailing stop logic, TP adjustment near EMA 200
on the 15-minute timeframe, and daily trade limit enforcement.
"""

import MetaTrader5 as mt5
import numpy as np

from broker import mt5_connector
from config import TP_PIPS, TRAILING_STEPS, EMA_15M_PERIOD, MAX_TRADES_PER_DAY


def compute_tp_sl(symbol: str, action: str, entry_price: float):
    """
    Compute initial SL and TP for a new trade.

    TP = 20 pips from entry.
    SL = 20 pips on the opposite side (1:1 initial, trailing will protect).

    If the EMA 200 on the 15-min chart is between entry and the 20-pip TP,
    the TP is adjusted to just before the EMA 200.

    Returns
    -------
    (sl, tp) : tuple[float, float]
    """
    pip = mt5_connector.pip_value(symbol)
    tp_distance = TP_PIPS * pip
    sl_distance = TP_PIPS * pip  # 1:1 initially

    if action == "buy":
        tp = entry_price + tp_distance
        sl = entry_price - sl_distance
    else:
        tp = entry_price - tp_distance
        sl = entry_price + sl_distance

    # ── Adjust TP if EMA 200 (15 min) is in the way ──
    ema_200_value = _get_ema_200_15m(symbol)
    if ema_200_value is not None:
        if action == "buy" and entry_price < ema_200_value < tp:
            # EMA is between entry and TP → pull TP back to 2 pips before EMA
            tp = ema_200_value - 2 * pip
            print(f"⚠️ TP ajustado por EMA200-15m: {mt5_connector.price_text(symbol, tp)}")
        elif action == "sell" and tp < ema_200_value < entry_price:
            tp = ema_200_value + 2 * pip
            print(f"⚠️ TP ajustado por EMA200-15m: {mt5_connector.price_text(symbol, tp)}")

    return sl, tp


def _get_ema_200_15m(symbol: str):
    """Get current EMA 200 value on the 15-minute chart."""
    try:
        df = mt5_connector.get_rates(symbol, mt5.TIMEFRAME_M15, EMA_15M_PERIOD + 50)
        if df is None or len(df) < EMA_15M_PERIOD:
            return None
        ema = df["close"].ewm(span=EMA_15M_PERIOD, adjust=False).mean()
        return float(ema.iloc[-1])
    except Exception:
        return None


def apply_trailing_stop(symbol: str, position):
    """
    Apply stepped trailing stop to an open position.

    Steps (configurable in config.TRAILING_STEPS):
        - When trade moves 75% toward TP → move SL to 25% profit
        - When trade moves 80-90%       → move SL to 80%
        - etc.

    Returns
    -------
    (new_sl, reason) or (None, None) if no change needed.
    """
    entry = position.price_open
    current_tp = position.tp
    current_sl = position.sl

    if current_tp == 0 or entry == 0:
        return None, None

    # Total risk/reward distance
    if position.type == mt5.ORDER_TYPE_BUY:
        total_distance = current_tp - entry
        current_price = mt5.symbol_info_tick(symbol).bid
        profit_distance = current_price - entry
    else:
        total_distance = entry - current_tp
        current_price = mt5.symbol_info_tick(symbol).ask
        profit_distance = entry - current_price

    if total_distance <= 0:
        return None, None

    profit_pct = profit_distance / total_distance

    # Walk through trailing steps from highest to lowest
    best_sl = None
    best_reason = None
    for trigger_pct, sl_pct in sorted(TRAILING_STEPS, reverse=True):
        if profit_pct >= trigger_pct:
            # Move SL to sl_pct of the total distance (in profit)
            if position.type == mt5.ORDER_TYPE_BUY:
                new_sl = entry + total_distance * sl_pct
                # Only move if it's better (higher) than current SL
                if new_sl > current_sl + 1e-10:
                    best_sl = new_sl
                    best_reason = f"Trailing {int(trigger_pct*100)}%→SL al {int(sl_pct*100)}%"
            else:
                new_sl = entry - total_distance * sl_pct
                # Only move if it's better (lower) than current SL
                if new_sl < current_sl - 1e-10:
                    best_sl = new_sl
                    best_reason = f"Trailing {int(trigger_pct*100)}%→SL al {int(sl_pct*100)}%"
            break  # Use the highest triggered step

    return best_sl, best_reason


def count_trades_today(now_utc) -> int:
    """Count how many trades the bot opened today (by magic number)."""
    start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    from datetime import timedelta
    end = now_utc + timedelta(hours=1)

    deals = mt5_connector.history_deals_get(start, end)
    if deals is None:
        return 0

    return sum(
        1 for d in deals
        if d.magic == mt5_connector.MAGIC and d.entry == mt5.DEAL_ENTRY_IN
    )


def can_open_trade(now_utc) -> bool:
    """Check if we can open a new trade (max 1 per day)."""
    # Also check if there's already an open position
    open_positions = mt5_connector.get_open_positions()
    if open_positions:
        return False
    return count_trades_today(now_utc) < MAX_TRADES_PER_DAY
