"""
=============================================================================
JOCSEL BOT — Live Trading Orchestrator
=============================================================================
Connects MT5 + TradeLocker + Telegram.

Runs a loop during London 8 AM – New York 12 PM:
  1. Check session hours
  2. Filter out news windows (±15 min high-impact)
  3. Look for signals from the ML model
  4. Execute on MT5 → copy to TradeLocker → notify Telegram
  5. Manage trailing stop on open positions
  6. Detect closed trades and notify

USAGE:
    # Set environment variables first (see config.py)
    python live_bot.py --pair EURUSD

REQUIRED ENV VARS:
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    TRADELOCKER_EMAIL, TRADELOCKER_PASSWORD, TRADELOCKER_SERVER
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
=============================================================================
"""

import argparse
import time
import warnings
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from joblib import load

import config
from broker import mt5_connector, tradelocker_connector
from notifications import telegram_notifier
from strategy.risk_manager import (
    apply_trailing_stop,
    can_open_trade,
    compute_tp_sl,
)
from strategy.news_filter import is_news_window

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Globals ───────────────────────────────────────────────────────────────
MODEL = None
SCALER = None


def load_model(model_path: str = "xgboost_model.joblib",
               scaler_path: str = "scaler.joblib"):
    """Load the trained XGBoost model and scaler."""
    global MODEL, SCALER
    MODEL = load(model_path)
    SCALER = load(scaler_path)
    print(f"✅ Modelo cargado: {model_path}")


def is_within_session(now_utc: datetime) -> bool:
    """Check if current time is within trading hours (8 AM London – 12 PM NY)."""
    hour = now_utc.hour
    # London 8 AM UTC = 8, NY 12 PM EDT = 16 UTC (summer) / 17 UTC (winter)
    # Using configurable bounds
    return config.LONDON_START_UTC <= hour < config.NY_END_UTC


def get_signal(symbol: str):
    """
    Get a trading signal from the ML model for the given symbol.

    Returns
    -------
    (action, confidence) : ('buy'|'sell'|None, float)
    """
    if MODEL is None or SCALER is None:
        return None, 0.0

    try:
        # Get 5-min data from MT5
        df = mt5_connector.get_rates(symbol, mt5.TIMEFRAME_M5, 300)
        if df is None or len(df) < 210:
            return None, 0.0

        # Compute features (same as training pipeline)
        from jocsel_bot import (
            compute_anchored_vwap,
            compute_daily_vwap,
            compute_ema,
        )

        df = df.rename(columns={"tick_volume": "volume"})
        if "volume" not in df.columns and "real_volume" in df.columns:
            df["volume"] = df["real_volume"]

        df["vwap_anchored"] = compute_anchored_vwap(df, anchor_hour=8)
        df["vwap_daily"] = compute_daily_vwap(df)
        df["ema_200"] = compute_ema(df["close"], period=200)

        # Bias
        long_bias = (df["close"] > df["vwap_anchored"]) & (
            df["close"] > df["ema_200"]
        )
        short_bias = (df["close"] < df["vwap_anchored"]) & (
            df["close"] < df["ema_200"]
        )
        df["bias"] = 0
        df.loc[long_bias, "bias"] = 1
        df.loc[short_bias, "bias"] = -1

        # Spread features (proxy)
        df["spread_pips"] = (df["high"] - df["low"]) / mt5_connector.pip_value(symbol)
        df["spread_ma"] = df["spread_pips"].rolling(20).mean()
        df["spread_ratio"] = df["spread_pips"] / df["spread_ma"].replace(0, np.nan)

        feature_cols = [
            "vwap_anchored", "vwap_daily", "ema_200",
            "bias", "spread_pips", "spread_ma", "spread_ratio",
        ]
        available = [c for c in feature_cols if c in df.columns]
        row = df[available].iloc[-1:]

        if row.isna().any(axis=1).iloc[0]:
            return None, 0.0

        X = SCALER.transform(row)
        pred = MODEL.predict(X)[0]
        proba = MODEL.predict_proba(X)[0]
        confidence = float(max(proba))

        if pred == 1:
            bias = int(df["bias"].iloc[-1])
            if bias == 1:
                return "buy", confidence
            elif bias == -1:
                return "sell", confidence

        return None, 0.0

    except Exception as e:
        print(f"⚠️ Error obteniendo señal {symbol}: {e}")
        return None, 0.0


def manage_open_positions():
    """Apply trailing stop to all open positions and notify changes."""
    positions = mt5_connector.get_open_positions()

    for pos in positions:
        symbol = pos.symbol
        new_sl, reason = apply_trailing_stop(symbol, pos)

        if new_sl is not None:
            old_sl = pos.sl
            old_tp = pos.tp
            result = mt5_connector.modify_position(
                pos.ticket, symbol, new_sl=new_sl
            )
            if result:
                print(f"🛡 {symbol} SL movido: {reason}")
                telegram_notifier.send_sltp_update(
                    symbol, pos.ticket,
                    old_sl, old_tp,
                    new_sl, old_tp,
                    reason,
                )


def execute_trade(symbol: str, action: str, confidence: float):
    """Execute trade on MT5, copy to TradeLocker, notify Telegram."""
    info = mt5_connector.symbol_info(symbol)
    if info is None:
        return

    entry_price = info.ask if action == "buy" else info.bid
    sl, tp = compute_tp_sl(symbol, action, entry_price)

    # Default lot (can be made configurable)
    lot = 0.01

    # ── MT5 ──
    result = mt5_connector.open_order(symbol, action, lot, sl, tp)
    if result is None:
        return

    # ── TradeLocker (copy) ──
    tradelocker_connector.execute(symbol, action, lot, sl, tp)

    # ── Telegram ──
    bias_text = "LONG" if action == "buy" else "SHORT"
    telegram_notifier.send_trade_opened(
        symbol, action, lot, entry_price, sl, tp,
        confidence=confidence, trend=bias_text, result=result,
    )


def main_loop(pairs: tuple):
    """Main trading loop."""
    print("=" * 60)
    print("  JOCSEL BOT — Live Trading")
    print(f"  Pares: {', '.join(pairs)}")
    print(f"  Horario: {config.LONDON_START_UTC}:00 – {config.NY_END_UTC}:00 UTC")
    print(f"  TP: {config.TP_PIPS} pips | Max trades/día: {config.MAX_TRADES_PER_DAY}")
    print("=" * 60)

    while True:
        try:
            now_utc = datetime.now(timezone.utc)

            # 1. Check session hours
            if not is_within_session(now_utc):
                time.sleep(30)
                continue

            # 2. Check for closed trades and notify
            telegram_notifier.check_closed_trades(now_utc, config.MAGIC)

            # 3. Manage trailing stop on open positions
            manage_open_positions()

            # 4. Check if we can open a new trade
            if not can_open_trade(now_utc):
                time.sleep(10)
                continue

            # 5. Check news filter
            if is_news_window(now_utc):
                time.sleep(60)
                continue

            # 6. Look for signals
            for symbol in pairs:
                action, confidence = get_signal(symbol)
                if action is not None:
                    print(
                        f"📊 Señal: {symbol} {action.upper()} "
                        f"conf={confidence:.2f}"
                    )
                    execute_trade(symbol, action, confidence)
                    break  # Only 1 trade per cycle

            time.sleep(10)  # Check every 10 seconds

        except KeyboardInterrupt:
            print("\n🛑 Bot detenido por usuario")
            break
        except Exception as e:
            print(f"❌ Error en loop: {e}")
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description="Jocsel Bot — Live Trading")
    parser.add_argument(
        "--pair", default="EURUSD,GBPUSD",
        help="Comma-separated pairs (default: EURUSD,GBPUSD)",
    )
    parser.add_argument(
        "--model", default="xgboost_model.joblib",
        help="Path to trained model",
    )
    parser.add_argument(
        "--scaler", default="scaler.joblib",
        help="Path to scaler",
    )
    args = parser.parse_args()

    pairs = tuple(p.strip().upper() for p in args.pair.split(","))

    # ── Connect ──
    if not mt5_connector.connect():
        return

    tradelocker_connector.init()
    load_model(args.model, args.scaler)

    # ── Telegram startup ──
    telegram_notifier.send_startup(pairs)

    try:
        main_loop(pairs)
    finally:
        mt5_connector.shutdown()


if __name__ == "__main__":
    main()
