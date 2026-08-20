"""
Configuration — All credentials via environment variables.
==========================================================
Set these environment variables before running the bot:

    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    TRADELOCKER_EMAIL, TRADELOCKER_PASSWORD, TRADELOCKER_SERVER,
    TRADELOCKER_ENV, TRADELOCKER_ACCOUNT, TRADELOCKER_ENABLED
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import os

# ── MetaTrader 5 ──────────────────────────────────────────────────────────
MT5_LOGIN = int(os.environ.get("MT5_LOGIN", "0"))
MT5_PASSWORD = os.environ.get("MT5_PASSWORD", "")
MT5_SERVER = os.environ.get("MT5_SERVER", "")

# ── TradeLocker ───────────────────────────────────────────────────────────
TRADELOCKER_ENABLED = os.environ.get("TRADELOCKER_ENABLED", "1") == "1"
TRADELOCKER_EMAIL = os.environ.get("TRADELOCKER_EMAIL", "")
TRADELOCKER_PASSWORD = os.environ.get("TRADELOCKER_PASSWORD", "")
TRADELOCKER_SERVER = os.environ.get("TRADELOCKER_SERVER", "")
TRADELOCKER_ENV = os.environ.get("TRADELOCKER_ENV", "demo")
TRADELOCKER_ACCOUNT = os.environ.get("TRADELOCKER_ACCOUNT", "")
TRADELOCKER_ENV_URLS = {
    "demo": "https://demo.tradelocker.com",
    "live": "https://live.tradelocker.com",
}

# ── Telegram ──────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── Trading parameters ───────────────────────────────────────────────────
MAGIC = 202507
SUPPORTED_PAIRS = ("EURUSD", "GBPUSD")
TP_PIPS = 20
MAX_TRADES_PER_DAY = 1

# Trailing stop thresholds (profit_pct_trigger, move_sl_to_pct)
TRAILING_STEPS = [
    (0.75, 0.25),
    (0.80, 0.50),
    (0.85, 0.60),
    (0.90, 0.80),
    (0.95, 0.90),
]

# EMA 200 on 15-min — adjust TP if EMA is between entry and 20-pip TP
EMA_15M_PERIOD = 200

# News filter: skip trading ±15 minutes around high-impact news
NEWS_BUFFER_MINUTES = 15

# Session hours (UTC)
LONDON_START_UTC = 8   # 8 AM London
NY_END_UTC = 16        # 12 PM New York = 16:00 UTC (EDT) / 17:00 UTC (EST)
