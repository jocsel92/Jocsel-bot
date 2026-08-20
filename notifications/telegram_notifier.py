"""
Telegram Notifier — Send trade notifications to Telegram.
==========================================================
"""

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# De-duplication state
_last_open_ticket: dict = {}
_last_sltp_sent: dict = {}
_notified_closed_deals: set = set()


def _send(msg: str) -> bool:
    """Send a message to the configured Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram no configurado (falta token o chat_id)")
        return False
    try:
        payload = urlencode({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg,
        }).encode()
        req = Request(
            url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data=payload,
            method="POST",
        )
        with urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode()).get("ok", False)
    except Exception as e:
        print(f"Telegram error: {e}")
        return False


def _price_text(symbol: str, value: float) -> str:
    """Format price using MT5 symbol digits."""
    try:
        from broker import mt5_connector
        info = mt5_connector.symbol_info(symbol)
        digits = info.digits if info else 5
    except Exception:
        digits = 5
    return f"{float(value):.{digits}f}"


def send_startup(pairs: tuple):
    """Send bot startup message."""
    _send(
        f"🤖 BOT INICIADO - Buscando operaciones\n"
        f"⏰ Horario: 8:00 AM Londres — 12:00 PM New York\n"
        f"📍 Pares: {', '.join(pairs)}"
    )
    print("Telegram startup enviado")


def send_trade_opened(symbol, action, lot, entry, sl, tp,
                      confidence=0.0, trend="", result=None):
    """Notify that a trade was opened."""
    ticket = getattr(result, "order", None)
    if ticket and _last_open_ticket.get(symbol) == ticket:
        return  # Already notified

    side = "BUY" if action == "buy" else "SELL"
    fill_price = getattr(result, "price", 0.0) or entry

    msg = (
        f"🚨 OPERACION ABIERTA\n"
        f"Par: {symbol} {side}\n"
        f"Entrada: {_price_text(symbol, fill_price)}\n"
        f"SL: {_price_text(symbol, sl)}\n"
        f"TP: {_price_text(symbol, tp)}\n"
        f"Lote: {lot}\n"
        f"Conf: {confidence:.2f} | Trend: {trend}"
    )
    if _send(msg) and ticket:
        _last_open_ticket[symbol] = ticket


def send_trade_closed(symbol, profit, close_price):
    """Notify that a trade was closed."""
    if profit > 0:
        msg = (
            f"✅ CIERRE POR TP\n"
            f"Par: {symbol}\n"
            f"Profit: +${profit:.2f}\n"
            f"Precio cierre: {_price_text(symbol, close_price)}\n"
            f"🎯 Objetivo alcanzado"
        )
    else:
        msg = (
            f"❌ CIERRE POR SL\n"
            f"Par: {symbol}\n"
            f"Profit: ${profit:.2f}\n"
            f"Precio cierre: {_price_text(symbol, close_price)}\n"
            f"🔴 Stop alcanzado"
        )
    _send(msg)


def send_sltp_update(symbol, ticket, old_sl, old_tp, new_sl, new_tp, reason):
    """Notify SL/TP modification."""
    old_sl_f = float(old_sl or 0)
    new_sl_f = float(new_sl or 0)
    old_tp_f = float(old_tp or 0)
    new_tp_f = float(new_tp or 0)
    sl_changed = abs(new_sl_f - old_sl_f) > 1e-10
    tp_changed = abs(new_tp_f - old_tp_f) > 1e-10

    if not sl_changed and not tp_changed:
        return

    if sl_changed:
        key = (ticket, round(new_sl_f, 10), reason)
        if _last_sltp_sent.get(ticket) == key:
            return
        msg = (
            f"🛡 SL MODIFICADO — {reason}\n"
            f"Par: {symbol}\n"
            f"Viejo SL: {_price_text(symbol, old_sl_f)}\n"
            f"Nuevo SL: {_price_text(symbol, new_sl_f)}\n"
            f"Ticket: {ticket}"
        )
        if _send(msg):
            _last_sltp_sent[ticket] = key

    if tp_changed:
        key_tp = (ticket, round(new_tp_f, 10), reason + "_TP")
        if _last_sltp_sent.get(key_tp) == key_tp:
            return
        msg = (
            f"🎯 TP MODIFICADO — {reason}\n"
            f"Par: {symbol}\n"
            f"Viejo TP: {_price_text(symbol, old_tp_f)}\n"
            f"Nuevo TP: {_price_text(symbol, new_tp_f)}\n"
            f"Ticket: {ticket}"
        )
        if _send(msg):
            _last_sltp_sent[key_tp] = key_tp


def check_closed_trades(now_utc, magic):
    """Check MT5 deal history and notify closed trades."""
    from broker import mt5_connector
    from datetime import timedelta

    start_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    end_utc = now_utc + timedelta(hours=1)

    deals = mt5_connector.history_deals_get(start_utc, end_utc)
    if deals is None:
        return

    import MetaTrader5 as mt5
    for d in deals:
        if d.magic != magic:
            continue
        if d.entry != mt5.DEAL_ENTRY_OUT:
            continue
        if d.ticket in _notified_closed_deals:
            continue

        send_trade_closed(d.symbol, float(d.profit), float(d.price))
        _notified_closed_deals.add(d.ticket)
        print(f"Telegram cierre notificado {d.symbol} profit={d.profit}")
